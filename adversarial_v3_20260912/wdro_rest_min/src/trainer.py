"""Unified resumable Actor-Critic trainer; stage 2 currently enables ERM."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .accounting import CallLedger
from .checkpoints import atomic_torch_save, capture_rng_state, load_checkpoint, restore_rng_state
from .geosteiner import GeoSteinerEvaluator, ParallelGeoSteinerEvaluator
from .geometry import NetTransform, inverse_net
from .io_utils import append_jsonl, atomic_write_json
from .job_control import utc_now
from .metrics import relative_regret
from .protocol import protocol_sha256
from .rest import RestTreeEvaluator, build_actor, build_critic, file_sha256, load_trusted_legacy_actor
from .training_data import EMPTY_STREAM_SHA256, extend_stream_hash, materialize_training_batch
from .validation import evaluate_validation, validation_manifest_hash


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_TRAINER_CONFIG = PROJECT_ROOT / "configs" / "stage2_trainer_v1.json"


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_trainer_config(path: str | Path = DEFAULT_TRAINER_CONFIG) -> dict[str, Any]:
    source = Path(path)
    config = json.loads(source.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or config.get("research_protocol_override") is not False:
        raise ValueError("invalid stage-2 trainer config")
    if config.get("status") != "frozen":
        raise ValueError("stage-2 trainer config must be frozen before formal runs")
    if config["loss"] != "per_net_relative_regret":
        raise ValueError("stage-2 trainer loss drifted from protocol_v1")
    return config


def derived_seed(protocol_id: str, degree: int, training_seed: int, namespace: str) -> int:
    identity = f"{protocol_id}\0{degree}\0{training_seed}\0{namespace}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(identity).digest()[:4], "little")


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_determinism(config: dict[str, Any]) -> None:
    deterministic = config["determinism"]
    torch.use_deterministic_algorithms(bool(deterministic["torch_deterministic_algorithms"]))
    torch.backends.cudnn.benchmark = bool(deterministic["cudnn_benchmark"])


def _ledger_from_dict(value: dict[str, int]) -> CallLedger:
    ledger = CallLedger()
    for field in ledger.__dataclass_fields__:
        if field not in value:
            raise ValueError(f"checkpoint call ledger is missing {field}")
        ledger.add(field, int(value[field]))
    return ledger


def _checkpoint_state(
    *,
    identity: dict[str, Any],
    update: int,
    actor: torch.nn.Module,
    critic: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    best: dict[str, Any],
    ledger: CallLedger,
    stream_hash: str,
    initialization: dict[str, Any],
    inverse_audit: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "saved_at_utc": utc_now(),
        "run_identity": identity,
        "update": update,
        "actor_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "rng_state": capture_rng_state(),
        "best": best,
        "call_ledger": ledger.to_dict(),
        "training_stream_hash": stream_hash,
        "initialization": initialization,
        "inverse_length_audit": inverse_audit,
    }


def _save_checkpoint_pair(
    *,
    latest_path: Path,
    best_path: Path,
    save_best: bool,
    state: dict[str, Any],
) -> None:
    atomic_torch_save(state, latest_path)
    if save_best:
        atomic_torch_save(state, best_path)


def _inverse_length_audit(
    *,
    actor: torch.nn.Module,
    rest: RestTreeEvaluator,
    geosteiner: GeoSteinerEvaluator,
    batch: Any,
    ledger: CallLedger,
) -> dict[str, Any]:
    actor.eval()
    with torch.no_grad():
        topology, _ = actor(batch.pins[:1], True)
    actor.train()
    output = topology[0].detach().cpu().numpy()
    normalized = np.asarray(batch.pins[0], dtype=np.float64)
    transform = NetTransform(
        offset_xy=np.asarray(batch.offsets[0], dtype=np.float64),
        scale=float(batch.scales[0]),
        original_degree=len(normalized),
        unique_degree=len(normalized),
        duplicate_count=0,
    )
    original = inverse_net(normalized, transform)
    rest_normalized = rest.length(normalized, output)
    rest_original = rest.length(original, output)
    lstar_normalized = geosteiner.length(normalized)
    lstar_original = geosteiner.length(original)
    ledger.add("actor_forward_nets", 1)
    ledger.add("exact_rest_evaluator_nets", 2)
    ledger.add("geosteiner_nets", 2)
    rest_error = abs(rest_original - rest_normalized / transform.scale)
    lstar_error = abs(lstar_original - lstar_normalized / transform.scale)
    rest_tolerance = 1e-8 * max(1.0, abs(rest_original))
    lstar_tolerance = 1e-8 * max(1.0, abs(lstar_original))
    passed = rest_error <= rest_tolerance and lstar_error <= lstar_tolerance
    result = {
        "status": "pass" if passed else "fail",
        "net_id": batch.net_ids[0].decode("utf-8"),
        "scale": transform.scale,
        "rest_normalized": rest_normalized,
        "rest_original": rest_original,
        "rest_absolute_error": rest_error,
        "lstar_normalized": lstar_normalized,
        "lstar_original": lstar_original,
        "lstar_absolute_error": lstar_error,
    }
    if not passed:
        raise RuntimeError(f"inverse length audit failed: {result}")
    return result


def train_erm(
    *,
    protocol: dict[str, Any],
    trainer_config: dict[str, Any],
    degree: int,
    training_seed: int,
    expected_updates: int,
    stop_after: int,
    device: torch.device,
    run_id: str,
    checkpoint_dir: str | Path,
    log_dir: str | Path,
    resume: bool,
    heartbeat_callback: Any | None = None,
    physical_gpu_index: int | None = None,
    host_runtime_manifest: str | None = None,
    geosteiner_workers: int = 1,
) -> dict[str, Any]:
    if degree not in protocol["scope"]["degrees"]:
        raise ValueError("degree is outside protocol_v1")
    if training_seed not in protocol["data"]["seeds"]["training"]:
        raise ValueError("training seed is outside protocol_v1")
    if expected_updates <= 0 or stop_after <= 0 or stop_after > expected_updates:
        raise ValueError("invalid update bounds")
    batch_size = int(protocol["training"]["batch_size"])
    config_hash = canonical_sha256(trainer_config)
    identity = {
        "method": "erm",
        "degree": degree,
        "training_seed": training_seed,
        "expected_updates": expected_updates,
        "batch_size": batch_size,
        "protocol_canonical_sha256": protocol_sha256(protocol),
        "trainer_config_sha256": config_hash,
    }
    from .geosteiner import reference_identity
    identity['reference_solver'] = reference_identity()
    checkpoints = Path(checkpoint_dir)
    logs = Path(log_dir)
    checkpoints.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    latest_path = checkpoints / "latest.pt"
    best_path = checkpoints / "best.pt"
    event_log = logs / "events.jsonl"
    result_path = logs / "result.json"
    configure_determinism(trainer_config)

    init_seed = derived_seed(protocol["protocol_id"], degree, training_seed, "actor-construction")
    set_all_seeds(init_seed)
    actor = build_actor(degree, device)
    template = protocol["training"]["initial_actor_checkpoint_template"]
    legacy_path = WORKSPACE_ROOT / template.format(degree=degree)
    initialization = load_trusted_legacy_actor(actor, legacy_path, map_location=device)
    critic_seed = derived_seed(protocol["protocol_id"], degree, training_seed, "critic-initialization")
    set_all_seeds(critic_seed)
    critic = build_critic(degree, device)
    optimizer_config = trainer_config["optimizer"]
    optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()),
        lr=float(optimizer_config["learning_rate"]),
        eps=float(optimizer_config["epsilon"]),
        weight_decay=float(optimizer_config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    rest = RestTreeEvaluator()
    if geosteiner_workers <= 0:
        raise ValueError("geosteiner_workers must be positive")
    geosteiner = ParallelGeoSteinerEvaluator(geosteiner_workers)
    ledger = CallLedger()
    best: dict[str, Any] = {
        "update": -1,
        "selection_objective": math.inf,
        "nominal_mean": math.inf,
        "validation_manifest_sha256": None,
    }
    stream_hash = EMPTY_STREAM_SHA256
    inverse_audit: dict[str, Any] = {}
    start_update = 1
    resumed_from: str | None = None

    if resume:
        checkpoint = load_checkpoint(latest_path, map_location=device)
        if checkpoint["run_identity"] != identity:
            raise ValueError("resume checkpoint run identity differs from the requested run")
        actor.load_state_dict(checkpoint["actor_state_dict"], strict=True)
        critic.load_state_dict(checkpoint["critic_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        best = checkpoint["best"]
        ledger = _ledger_from_dict(checkpoint["call_ledger"])
        stream_hash = checkpoint["training_stream_hash"]
        inverse_audit = checkpoint["inverse_length_audit"]
        start_update = int(checkpoint["update"]) + 1
        restore_rng_state(checkpoint["rng_state"])
        resumed_from = str(latest_path)
        append_jsonl(event_log, {"event": "resume", "at_utc": utc_now(), "from_update": start_update - 1})
    else:
        if latest_path.exists() or best_path.exists():
            raise FileExistsError(f"run already has checkpoint(s); pass --resume: {checkpoints}")
        policy_seed = derived_seed(protocol["protocol_id"], degree, training_seed, "policy-training")
        set_all_seeds(policy_seed)
        audit_batch = materialize_training_batch(
            protocol=protocol,
            degree=degree,
            training_seed=training_seed,
            update=1,
            batch_size=batch_size,
        )
        inverse_audit = _inverse_length_audit(
            actor=actor, rest=rest, geosteiner=geosteiner, batch=audit_batch, ledger=ledger
        )
        validation = evaluate_validation(
            actor,
            rest,
            protocol,
            degree=degree,
            device=device,
            batch_size=int(trainer_config["validation_batch_size"]),
            ledger=ledger,
        )
        best = {
            "update": 0,
            "selection_objective": validation["selection"]["worst_group_mean"],
            "nominal_mean": validation["selection"]["nominal_mean"],
            "validation_manifest_sha256": validation_manifest_hash(validation),
        }
        state = _checkpoint_state(
            identity=identity,
            update=0,
            actor=actor,
            critic=critic,
            optimizer=optimizer,
            scheduler=scheduler,
            best=best,
            ledger=ledger,
            stream_hash=stream_hash,
            initialization=initialization,
            inverse_audit=inverse_audit,
        )
        _save_checkpoint_pair(
            latest_path=latest_path, best_path=best_path, save_best=True, state=state
        )
        append_jsonl(
            event_log,
            {"event": "validation", "at_utc": utc_now(), "update": 0, "is_best": True, "result": validation},
        )

    if start_update > stop_after:
        return {
            "status": "already_at_or_beyond_requested_stop",
            "run_id": run_id,
            "completed_updates": start_update - 1,
            "expected_updates": expected_updates,
            "best": best,
            "latest_checkpoint": str(latest_path),
            "best_checkpoint": str(best_path),
        }

    actor.train()
    critic.train()
    mse_loss = torch.nn.MSELoss()
    validation_interval = int(trainer_config["validation_interval_updates"])
    checkpoint_interval = int(trainer_config["checkpoint_interval_updates"])
    structured_interval = int(trainer_config["structured_log_interval_updates"])
    clip = float(trainer_config["gradient_clip_l2"])
    started = time.monotonic()
    last_validation: dict[str, Any] | None = None
    peak_allocated = 0
    peak_reserved = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for update in range(start_update, stop_after + 1):
        batch = materialize_training_batch(
            protocol=protocol,
            degree=degree,
            training_seed=training_seed,
            update=update,
            batch_size=batch_size,
        )
        outputs, log_probs = actor(batch.pins, False)
        ledger.add("actor_forward_nets", batch_size)
        output_array = outputs.detach().cpu().numpy()
        rest_lengths = rest.lengths(batch.pins, output_array)
        lstar = geosteiner.lengths(batch.pins)
        ledger.add("exact_rest_evaluator_nets", batch_size)
        ledger.add("geosteiner_nets", batch_size)
        regrets = relative_regret(rest_lengths, lstar)
        if not np.isfinite(regrets).all():
            raise RuntimeError(f"non-finite training regret at update {update}")
        target = torch.as_tensor(regrets, dtype=torch.float32, device=device)
        predictions = critic(batch.pins)
        actor_loss = torch.mean((target - predictions).detach() * log_probs)
        critic_loss = mse_loss(predictions, target)
        loss = actor_loss + critic_loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at update {update}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), clip)
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(critic.parameters(), clip)
        if not (torch.isfinite(actor_grad_norm) and torch.isfinite(critic_grad_norm)):
            raise RuntimeError(f"non-finite gradient norm at update {update}")
        optimizer.step()
        scheduler.step()
        stream_hash = extend_stream_hash(stream_hash, batch.bundle_sha256)

        if device.type == "cuda":
            peak_allocated = max(peak_allocated, int(torch.cuda.max_memory_allocated(device)))
            peak_reserved = max(peak_reserved, int(torch.cuda.max_memory_reserved(device)))
        if update % structured_interval == 0 or update == start_update or update == stop_after:
            append_jsonl(
                event_log,
                {
                    "event": "train",
                    "at_utc": utc_now(),
                    "update": update,
                    "actor_loss": float(actor_loss.detach().cpu()),
                    "critic_loss": float(critic_loss.detach().cpu()),
                    "mean_regret": float(regrets.mean()),
                    "p95_regret": float(np.quantile(regrets, 0.95)),
                    "actor_grad_norm_before_clip": float(actor_grad_norm.detach().cpu()),
                    "critic_grad_norm_before_clip": float(critic_grad_norm.detach().cpu()),
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "batch_bundle_sha256": batch.bundle_sha256,
                    "training_stream_hash": stream_hash,
                    "call_ledger": ledger.to_dict(),
                },
            )

        save_best = False
        validation_due = update % validation_interval == 0 or update == expected_updates
        if validation_due:
            last_validation = evaluate_validation(
                actor,
                rest,
                protocol,
                degree=degree,
                device=device,
                batch_size=int(trainer_config["validation_batch_size"]),
                ledger=ledger,
            )
            objective = float(last_validation["selection"]["worst_group_mean"])
            if objective < float(best["selection_objective"]):
                best = {
                    "update": update,
                    "selection_objective": objective,
                    "nominal_mean": float(last_validation["selection"]["nominal_mean"]),
                    "validation_manifest_sha256": validation_manifest_hash(last_validation),
                }
                save_best = True
            append_jsonl(
                event_log,
                {
                    "event": "validation",
                    "at_utc": utc_now(),
                    "update": update,
                    "is_best": save_best,
                    "result": last_validation,
                },
            )

        checkpoint_due = update % checkpoint_interval == 0 or update == stop_after or validation_due
        if checkpoint_due:
            state = _checkpoint_state(
                identity=identity,
                update=update,
                actor=actor,
                critic=critic,
                optimizer=optimizer,
                scheduler=scheduler,
                best=best,
                ledger=ledger,
                stream_hash=stream_hash,
                initialization=initialization,
                inverse_audit=inverse_audit,
            )
            _save_checkpoint_pair(
                latest_path=latest_path, best_path=best_path, save_best=save_best, state=state
            )
        if heartbeat_callback is not None and (
            update % structured_interval == 0 or update == start_update or update == stop_after
        ):
            heartbeat_callback(
                update,
                "running",
                {
                    "mean_regret": float(regrets.mean()),
                    "latest_checkpoint": str(latest_path),
                    "event_log": str(event_log),
                },
            )

    elapsed = time.monotonic() - started
    status = "complete" if stop_after == expected_updates else "paused_for_resume"
    if device.type == "cuda":
        peak_allocated = max(peak_allocated, int(torch.cuda.max_memory_allocated(device)))
        peak_reserved = max(peak_reserved, int(torch.cuda.max_memory_reserved(device)))
    result = {
        "schema_version": 1,
        "status": status,
        "run_id": run_id,
        "method": "erm",
        "degree": degree,
        "training_seed": training_seed,
        "completed_updates": stop_after,
        "expected_updates": expected_updates,
        "batch_size": batch_size,
        "policy_update_nets": stop_after * batch_size,
        "started_this_segment_at_update": start_update,
        "resumed_from": resumed_from,
        "finished_at_utc": utc_now(),
        "segment_wall_seconds": elapsed,
        "best": best,
        "last_validation": last_validation,
        "call_ledger": ledger.to_dict(),
        "training_stream_hash": stream_hash,
        "inverse_length_audit": inverse_audit,
        "initialization": initialization,
        "protocol_canonical_sha256": identity["protocol_canonical_sha256"],
        "trainer_config_sha256": config_hash,
        "trainer_config_path": str(DEFAULT_TRAINER_CONFIG.relative_to(WORKSPACE_ROOT)),
        "latest_checkpoint": str(latest_path),
        "latest_checkpoint_sha256": file_sha256(latest_path),
        "best_checkpoint": str(best_path),
        "best_checkpoint_sha256": file_sha256(best_path),
        "event_log": str(event_log),
        "device": str(device),
        "physical_gpu_index": physical_gpu_index,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "host_runtime_manifest": host_runtime_manifest,
        "geosteiner_workers": geosteiner_workers,
        "peak_memory_allocated_bytes": peak_allocated,
        "peak_memory_reserved_bytes": peak_reserved,
        "nan_detected": False,
        "test_or_ispd_used": False,
    }
    atomic_write_json(result_path, result)
    segment_path = logs / f"segment_u{start_update:06d}_to_u{stop_after:06d}.json"
    atomic_write_json(segment_path, result, mode=0o444)
    append_jsonl(event_log, {"event": "segment_end", "at_utc": utc_now(), "result": result})
    geosteiner.close()
    return result
