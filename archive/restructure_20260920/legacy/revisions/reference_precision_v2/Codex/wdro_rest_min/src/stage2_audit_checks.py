"""Read-only model checks and resumable reconstruction of the frozen ERM pool.

This module is an auditor, not part of the frozen training implementation.
Test coordinates are read only to check content separation; no model is run on test.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .datasets import load_dataset, make_net_id
from .geometry import pin_set_sha256
from .io_utils import atomic_write_json
from .job_control import utc_now
from .rest import file_sha256
from .training_data import EMPTY_STREAM_SHA256, extend_stream_hash, materialize_training_batch
from .validation import load_validation_group, validation_manifest_hash


def all_finite(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, np.ndarray):
        return bool(np.isfinite(value).all()) if value.dtype.kind in "fciub" else True
    if isinstance(value, dict):
        return all(all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(all_finite(item) for item in value)
    if isinstance(value, (float, int)):
        return math.isfinite(value)
    return True


def validation_checks(events, best, protocol, trainer, degree):
    by_update = {}
    duplicates_agree = True
    for event in events:
        update = event["update"]
        if update in by_update:
            duplicates_agree &= by_update[update]["result"] == event["result"]
        by_update[update] = event
    expected = list(range(0, protocol["training"]["policy_updates"] + 1,
                          trainer["validation_interval_updates"]))
    ordered = [by_update[key] for key in sorted(by_update)]
    selected = min(ordered, key=lambda event: event["result"]["selection"]["worst_group_mean"])
    dataset_hashes = {}
    for group in protocol["data"]["distributions"]:
        _, _, _, metadata = load_validation_group(protocol, degree=degree, distribution=group)
        dataset_hashes[group] = metadata["array_bundle_sha256"]
    identities = True
    objectives = True
    for event in ordered:
        result = event["result"]
        scope = protocol["training"]["selection"]["worst_group_scope"]
        identities &= (
            result["split"] == "validation" and result["seed"] == protocol["data"]["seeds"]["validation"]
            and result["degree"] == degree and set(result["groups"]) == set(dataset_hashes)
            and result["inference"] == {"mode": "deterministic_greedy", "symmetry_transforms": 1,
                                         "batch_size": trainer["validation_batch_size"]}
            and all(result["groups"][group]["dataset_array_bundle_sha256"] == digest
                    and result["groups"][group]["summary"]["count"] ==
                    protocol["data"]["sizes_per_degree_distribution"]["validation"]
                    for group, digest in dataset_hashes.items())
        )
        objectives &= (
            result["selection"]["scope"] == scope
            and result["selection"]["worst_group_mean"] ==
            max(result["groups"][group]["summary"]["mean"] for group in scope)
            and result["selection"]["nominal_mean"] == result["groups"]["uniform"]["summary"]["mean"]
            and result["selection"]["test_or_ispd_used"] is False
        )
    return {
        "all_21_validation_updates_present": sorted(by_update) == expected,
        "duplicate_validation_replays_agree": duplicates_agree,
        "all_validation_data_and_lstar_verified": identities,
        "ood_objective_independently_recomputed": objectives,
        "selected_best_is_earliest_global_validation_minimum": (
            best["update"] == selected["update"]
            and best["best"]["selection_objective"] == selected["result"]["selection"]["worst_group_mean"]
            and best["best"]["nominal_mean"] == selected["result"]["selection"]["nominal_mean"]
            and best["best"]["validation_manifest_sha256"] == validation_manifest_hash(selected["result"])
        ),
    }


def optimizer_checks(checkpoint, trainer):
    optimizer = checkpoint["optimizer_state_dict"]
    update = checkpoint["update"]
    return {
        "optimizer_steps_match_checkpoint": bool(optimizer["state"]) and all(
            int(state["step"]) == update for state in optimizer["state"].values()),
        "scheduler_steps_match_checkpoint": checkpoint["scheduler_state_dict"]["last_epoch"] == update,
        "optimizer_hyperparameters_match": all(
            group["lr"] == trainer["optimizer"]["learning_rate"]
            and group["eps"] == trainer["optimizer"]["epsilon"]
            and group["weight_decay"] == trainer["optimizer"]["weight_decay"]
            for group in optimizer["param_groups"]),
        "model_optimizer_rng_finite": all_finite(checkpoint),
        "rng_state_complete": set(checkpoint["rng_state"]) >= {"python", "numpy", "torch_cpu", "torch_cuda"},
    }


def rebuild_stream(*, protocol, degree, seed, result, root, dataset_root, heartbeat):
    """Resume every 100 batches; completed records are bound to source/data hashes."""
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"{result['run_id']}.json"
    source_root = Path(__file__).resolve().parent
    heldout = set()
    data_files = {}
    for split in ("validation", "test"):
        count = protocol["data"]["sizes_per_degree_distribution"][split]
        base_seed = protocol["data"]["seeds"][split]
        for group in protocol["data"]["distributions"]:
            path = dataset_root / split / f"{group}_d{degree}_s{base_seed}_n{count}.npz"
            arrays, metadata = load_dataset(path, verify=True)
            data_files[str(path)] = metadata["npz_sha256"]
            heldout.update(pin_set_sha256(pins) for pins in arrays["pins"])
    binding = {
        "protocol_hash": result["protocol_canonical_sha256"],
        "run_id": result["run_id"], "expected_stream_hash": result["training_stream_hash"],
        "expected_updates": result["expected_updates"], "batch_size": result["batch_size"],
        "source_hashes": {name: file_sha256(source_root / name) for name in
                          ("training_data.py", "datasets.py", "geometry.py", "stage2_audit_checks.py")},
        "heldout_dataset_files": data_files,
    }
    cursor = {"binding": binding, "completed_updates": 0, "stream_hash": EMPTY_STREAM_SHA256,
              "batches": [], "id_mismatches": 0, "heldout_content_intersections": 0,
              "status": "running"}
    if target.exists():
        cursor = json.loads(target.read_text())
        if cursor["binding"] != binding:
            raise ValueError(f"audit cursor identity/source changed: {target}")
        digest = EMPTY_STREAM_SHA256
        for index, batch in enumerate(cursor["batches"], 1):
            if batch["update"] != index:
                raise ValueError("non-contiguous audit cursor")
            digest = extend_stream_hash(digest, batch["bundle_sha256"])
        if digest != cursor["stream_hash"] or len(cursor["batches"]) != cursor["completed_updates"]:
            raise ValueError("audit cursor hash/count mismatch")
    batch_size = result["batch_size"]
    for update in range(cursor["completed_updates"] + 1, result["expected_updates"] + 1):
        batch = materialize_training_batch(protocol=protocol, degree=degree, training_seed=seed,
                                           update=update, batch_size=batch_size)
        cursor["id_mismatches"] += sum(
            item.decode() != make_net_id(f"train_u{update:06d}", "uniform", degree, seed, index)
            for index, item in enumerate(batch.net_ids))
        cursor["heldout_content_intersections"] += sum(pin_set_sha256(pins) in heldout for pins in batch.pins)
        if cursor["id_mismatches"] or cursor["heldout_content_intersections"]:
            cursor["status"] = "fail"
            atomic_write_json(target, cursor)
            raise RuntimeError(f"training/heldout separation failed: {target}")
        cursor["stream_hash"] = extend_stream_hash(cursor["stream_hash"], batch.bundle_sha256)
        cursor["completed_updates"] = update
        cursor["batches"].append({"update": update, "bundle_sha256": batch.bundle_sha256})
        if update % 100 == 0 or update == result["expected_updates"]:
            cursor["last_update_utc"] = utc_now()
            atomic_write_json(target, cursor)
            heartbeat(update, str(target))
    cursor["status"] = "pass" if cursor["stream_hash"] == result["training_stream_hash"] else "fail"
    cursor["logical_path"] = str(target)
    cursor["resolved_path"] = str(target.resolve())
    atomic_write_json(target, cursor, mode=0o444)
    return cursor
