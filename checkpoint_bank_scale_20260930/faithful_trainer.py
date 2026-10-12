"""WDRO using unchanged stage-2 optimization and stage-3 search dependencies."""
from __future__ import annotations
import json
import math
import os
import signal
import time
from pathlib import Path
from typing import Any
import numpy as np
import torch
from src.trainer import (PROJECT_ROOT, WORKSPACE_ROOT, DEFAULT_TRAINER_CONFIG, canonical_sha256,
    configure_determinism, derived_seed, set_all_seeds, _ledger_from_dict,
    _checkpoint_state, _save_checkpoint_pair, _inverse_length_audit)
from src.accounting import CallLedger
from src.checkpoints import load_checkpoint, restore_rng_state
from src.geosteiner import ParallelGeoSteinerEvaluator
from src.io_utils import append_jsonl, atomic_write_json
from src.job_control import utc_now
from src.metrics import relative_regret
from src.protocol import protocol_sha256
from src.rest import RestTreeEvaluator, build_actor, build_critic, file_sha256, load_trusted_legacy_actor
from src.training_data import EMPTY_STREAM_SHA256, extend_stream_hash, materialize_training_batch, TrainingBatch
from src.validation import evaluate_validation, validation_manifest_hash
from src.datasets import array_bundle_sha256
from src.candidates import ReferenceCache, generate_support, exact_score_support
from stage4.dual import select_wdro, update_lambda
from stage5.search import transport_costs, search_batch

from adapters import *

def train_wdro(
    *,
    wdro_config: dict[str, Any],
    rho_mode: str,
    baseline_config: dict[str, Any],
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
    method = "wdro_rest"
    if degree not in protocol["scope"]["degrees"]:
        raise ValueError("degree is outside protocol_v1")
    if training_seed not in protocol["data"]["seeds"]["training"]:
        raise ValueError("training seed is outside protocol_v1")
    if expected_updates <= 0 or stop_after <= 0 or stop_after > expected_updates:
        raise ValueError("invalid update bounds")
    batch_size = int(protocol["training"]["batch_size"])
    config_hash = canonical_sha256(trainer_config)
    identity = {
        "method": method,
        "degree": degree,
        "training_seed": training_seed,
        "expected_updates": expected_updates,
        "batch_size": batch_size,
        "protocol_canonical_sha256": protocol_sha256(protocol),
        "trainer_config_sha256": config_hash,
    }
    selected = json.loads((PROJECT_ROOT / 'configs/stage5_selected_wdro_v1.json').read_text())
    if (expected_updates != 2000 or selected['status'] != 'frozen' or selected['test_or_ispd_used']
        or rho_mode != selected['selected_rho_mode'] or wdro_config["degree"] != degree):
        raise ValueError("stage 6 requires corrected validation-frozen degree calibration and 2000 updates")
    calibration_path = PROJECT_ROOT.parent / wdro_config["calibration_artifact"]
    calibration = json.loads(calibration_path.read_text())
    if (calibration["status"] != "pass" or False
        or False
        or calibration["source"] != "training_centers_only"):
        raise ValueError("calibration has not passed")
    assert calibration["degree"] == degree and calibration["seed"] == 101
    assert wdro_config["ground_cost"] == "hungarian_l1_mean" and wdro_config["lambda_mode"] == "learned"
    assert wdro_config["lambda_initial"] == 0 and wdro_config["lambda_learning_rate"] == 1
    assert wdro_config["search_overrides"]["searcher"] == "random_sobol" and wdro_config["search_overrides"]["candidate_count"] == 32
    assert wdro_config["search_overrides"]["exact_rescore_top_k"] == 4
    rho = calibration[wdro_config["ground_cost"]][rho_mode] if rho_mode != "zero" else 0.0
    search_config = dict(baseline_config["search"], **wdro_config["search_overrides"], ground_cost=wdro_config["ground_cost"])
    identity["wdro_config_sha256"] = canonical_sha256(wdro_config)
    identity["stage"] = 6
    identity["ablation_config"] = wdro_config
    identity["rho_mode"] = rho_mode
    identity["rho"] = rho
    identity["calibration_sha256"] = file_sha256(calibration_path)
    source_files = [*sorted((PROJECT_ROOT / "src").glob("*.py")), *sorted((PROJECT_ROOT / "stage4").glob("*.py")), *sorted((PROJECT_ROOT / "stage5").glob("*.py")), PROJECT_ROOT / "stage6" / "trainer.py", PROJECT_ROOT / "scripts" / "train_stage6.py"]
    identity["baseline_config_sha256"] = canonical_sha256(baseline_config)
    identity["implementation_sha256"] = canonical_sha256({str(p.relative_to(PROJECT_ROOT)): file_sha256(p) for p in source_files})
    identity["inference"] = protocol["inference"]
    reference_path = BASE / "experiments" / "logs" / "stage2" / f"formal_erm_d{degree}_s{training_seed}_u2000" / "result.json"
    reference = json.loads(reference_path.read_text())
    if reference["status"] != "complete" or reference["completed_updates"] != 2000 or reference["test_or_ispd_used"]:
        raise ValueError("ERM selection reference has not completed validation-only training")
    nominal_limit = reference["best"]["nominal_mean"] + protocol["training"]["selection"]["nominal_mean_regret_degradation_max_absolute"]
    identity["erm_reference_result_sha256"] = file_sha256(reference_path)
    identity["nominal_limit"] = nominal_limit
    from src.geosteiner import reference_identity
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
    rest = RestTreeEvaluator(LIB)
    if geosteiner_workers <= 0:
        raise ValueError("geosteiner_workers must be positive")
    geosteiner = make_geosteiner()
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
    method_state = {"lambda": wdro_config["lambda_initial"], "rho": rho, "cost_history": [], "budget": {},
                    "support_stream_hash": EMPTY_STREAM_SHA256, "selection_stream_hash": EMPTY_STREAM_SHA256}
    reference_cache = ResearchReference(Path(checkpoint_dir).parent / "reference_state")
    reference_cache.initialize(actor)
    def checkpoint_state(**kwargs):
        value = _checkpoint_state(**kwargs)
        value["method_state"] = method_state
        value["reference_state"] = reference_cache.state()
        return value
    def feasible(validation):
        return float(validation["selection"]["nominal_mean"]) <= nominal_limit
    stop_requested = {"value": False}
    def request_stop(signum, frame):
        stop_requested["value"] = True
    previous_sigterm = signal.signal(signal.SIGTERM, request_stop)

    if resume:
        checkpoint = load_checkpoint(latest_path, map_location=device)
        if checkpoint["run_identity"] != identity:
            raise ValueError("resume checkpoint run identity differs from the requested run")
        actor.load_state_dict(checkpoint["actor_state_dict"], strict=True)
        critic.load_state_dict(checkpoint["critic_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        best = checkpoint["best"]
        method_state = checkpoint["method_state"]
        reference_cache.restore(checkpoint["reference_state"])
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
        inverse_audit = {'status':'not_run','reason':'existing Skill forbids unsolicited automated inverse audit'}
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
            "feasible": feasible(validation),
            "selection_objective": validation["selection"]["worst_group_mean"],
            "nominal_mean": validation["selection"]["nominal_mean"],
            "validation_manifest_sha256": validation_manifest_hash(validation),
        }
        state = checkpoint_state(
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

    for update in range(start_update, stop_after + 1):
        batch = materialize_training_batch(
            protocol=protocol,
            degree=degree,
            training_seed=training_seed,
            update=update,
            batch_size=batch_size,
        )
        center_batch = batch
        method_detail = {}
        selected_lstar = None
        if method == "wdro_rest":
            support, indices, true_regrets, references, method_detail = search_batch(
                batch, training_seed=training_seed, update=update, config=search_config,
                price=method_state["lambda"], rho=rho, actor=actor, rest=rest, geosteiner=geosteiner, cache=reference_cache, ledger=ledger)
            shortlisted = support.pins[np.arange(batch_size)[:, None], indices]
            costs = transport_costs(center_batch.pins, shortlisted, wdro_config["ground_cost"])
            price_before = method_state["lambda"]
            selected_indices, columns = select_wdro(indices, true_regrets, costs, price_before, rho)
            selected_cost = costs[np.arange(batch_size), columns]
            mean_cost = float(selected_cost.mean())
            price_after = (update_lambda(price_before, mean_cost, rho, wdro_config["lambda_learning_rate"])
                           if wdro_config["lambda_mode"] == "learned" else price_before)
            method_state["lambda"] = price_after
            method_state["cost_history"] = (method_state["cost_history"] + [mean_cost])[-100:]
            method_detail.update({"lambda_before": price_before, "lambda_after": price_after,
                "rho": rho, "mean_selected_cost": mean_cost, "budget_gap": mean_cost-rho,
                "mean_cost_last100": float(np.mean(method_state["cost_history"])),
                "exact_transport_costs": costs.tolist(), "selected_costs": selected_cost.tolist(),
                "selection_mode": "zero_cost_feasible_limit" if rho == 0 else "regret_minus_lambda_cost",
                "legal_candidate_fraction": 1.0})
            if not np.isfinite(price_after) or price_after > wdro_config["stability_gate"]["maximum_lambda"]:
                raise RuntimeError("dual lambda divergence gate")
            chosen = support.pins[np.arange(batch_size), selected_indices]
            selected_lstar = references[np.arange(batch_size), columns]
            selected_ids = np.asarray([f"{n.decode()}:candidate{int(j)}:block{support.refresh_block}".encode() for n,j in zip(batch.net_ids, selected_indices)])
            selected_hash = array_bundle_sha256({"pins": chosen, "net_ids": selected_ids})
            batch = TrainingBatch(chosen, selected_ids, batch.offsets, batch.scales, batch.metadata, selected_hash)
            method_detail["exact_lstar"] = references.tolist()
            method_detail["selected_indices"] = selected_indices.tolist()
            method_detail["selected_true_regret_mean"] = float(true_regrets[np.arange(batch_size), columns].mean())
            method_detail["selected_bundle_sha256"] = selected_hash
            for key in ("centers", "perturbations", "surrogate_nets", "exact_rescore_nets", "reference_requests", "reference_cache_misses", "reference_cache_hits", "repaired_candidates"):
                method_state["budget"][key] = method_state["budget"].get(key, 0) + method_detail[key]
            method_state["support_stream_hash"] = extend_stream_hash(method_state["support_stream_hash"], support.sha256)
            method_state["selection_stream_hash"] = extend_stream_hash(method_state["selection_stream_hash"], selected_hash)
        outputs, log_probs = actor(batch.pins, False)
        ledger.add("actor_forward_nets", batch_size)
        output_array = outputs.detach().cpu().numpy()
        rest_lengths = rest.lengths(batch.pins, output_array)
        lstar = selected_lstar if selected_lstar is not None else geosteiner.lengths(batch.pins)
        ledger.add("exact_rest_evaluator_nets", batch_size)
        if selected_lstar is None:
            ledger.add("geosteiner_nets", batch_size)
        regrets = relative_regret(rest_lengths, lstar)
        if not np.isfinite(regrets).all():
            raise RuntimeError(f"non-finite training regret at update {update}")
        target = torch.as_tensor(regrets, dtype=torch.float32, device=device)
        predictions = critic(batch.pins)
        per_net_actor_loss = (target - predictions).detach() * log_probs
        per_net_critic_loss = (predictions - target).square()
        actor_loss = per_net_actor_loss.mean()
        critic_loss = per_net_critic_loss.mean()
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
        reference_cache.refresh(actor,update)
        stream_hash = extend_stream_hash(stream_hash, center_batch.bundle_sha256)

        if device.type == "cuda":
            peak_allocated = max(peak_allocated, int(torch.cuda.max_memory_allocated(device)))
            peak_reserved = max(peak_reserved, int(torch.cuda.max_memory_reserved(device)))
        if True:  # One auditable training/search-budget record per policy update.
            append_jsonl(
                event_log,
                {
                    "event": "train",
                    "at_utc": utc_now(),
                    "update": update,
                    "actor_loss": float(actor_loss.detach().cpu()),
                    "critic_loss": float(critic_loss.detach().cpu()),
                    "mean_regret": float(regrets.mean()),
                    "negative_target_fraction": float((regrets<0).mean()),
                    "reference_accounting": reference_cache.accounting.copy(),
                    "p95_regret": float(np.quantile(regrets, 0.95)),
                    "actor_grad_norm_before_clip": float(actor_grad_norm.detach().cpu()),
                    "critic_grad_norm_before_clip": float(critic_grad_norm.detach().cpu()),
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "batch_bundle_sha256": center_batch.bundle_sha256,
                    "method_detail": method_detail,
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
            if feasible(last_validation) and (not best.get("feasible", False) or objective < float(best["selection_objective"])):
                best = {
                    "update": update,
                    "feasible": True,
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

        checkpoint_due = update % checkpoint_interval == 0 or update == stop_after or validation_due or stop_requested["value"]
        if checkpoint_due:
            state = checkpoint_state(
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
                    "negative_target_fraction": float((regrets<0).mean()),
                    "reference_accounting": reference_cache.accounting.copy(),
                    "latest_checkpoint": str(latest_path),
                    "event_log": str(event_log),
                },
            )

        if stop_requested["value"]:
            stop_after = update
            break

    signal.signal(signal.SIGTERM, previous_sigterm)
    elapsed = time.monotonic() - started
    status = "complete" if stop_after == expected_updates else "paused_for_resume"
    if device.type == "cuda":
        peak_allocated = max(peak_allocated, int(torch.cuda.max_memory_allocated(device)))
        peak_reserved = max(peak_reserved, int(torch.cuda.max_memory_reserved(device)))
    result = {
        "schema_version": 1,
        "status": status,
        "run_id": run_id,
        "method": method,
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
        "method_state": method_state,
        "reference_cache": {"logical_path": str(reference_cache.path), "resolved_path": str(reference_cache.path.resolve())},
        "run_identity": identity,
        "nominal_limit": nominal_limit,
        "selection_feasible": best.get("feasible", False),
        "artifact_paths": {"checkpoints_logical": str(checkpoints), "checkpoints_resolved": str(checkpoints.resolve()), "logs_logical": str(logs), "logs_resolved": str(logs.resolve())},
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
    if reference_cache is not None:
        reference_cache.close()
    return result
