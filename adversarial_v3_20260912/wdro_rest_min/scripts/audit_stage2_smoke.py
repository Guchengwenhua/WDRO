#!/usr/bin/env python3
"""Independently audit the stage-2 200-update ERM closed loop."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from src.checkpoints import load_checkpoint
from src.io_utils import atomic_write_json
from src.protocol import load_protocol, protocol_sha256
from src.rest import file_sha256
from src.trainer import canonical_sha256, load_trainer_config
from src.training_data import EMPTY_STREAM_SHA256, extend_stream_hash, materialize_training_batch


def all_finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(all_finite(item) for item in value)
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = load_protocol()
    trainer_config = load_trainer_config()
    checkpoint_root = Path(__file__).resolve().parents[2] / "experiments" / "checkpoints" / "stage2" / args.run_id
    log_root = Path(__file__).resolve().parents[2] / "experiments" / "logs" / "stage2" / args.run_id
    job_root = Path(__file__).resolve().parents[2] / "experiments" / "logs" / "jobs" / args.run_id
    result = json.loads((log_root / "result.json").read_text(encoding="utf-8"))
    latest = load_checkpoint(checkpoint_root / "latest.pt", map_location="cpu")
    best = load_checkpoint(checkpoint_root / "best.pt", map_location="cpu")
    events = [json.loads(line) for line in (log_root / "events.jsonl").read_text().splitlines() if line]
    job = json.loads((job_root / "job.json").read_text(encoding="utf-8"))
    task = json.loads((job_root / "task.json").read_text(encoding="utf-8"))
    updates = int(trainer_config["stage2_smoke_updates"])
    batch_size = int(protocol["training"]["batch_size"])
    validation_count = protocol["data"]["sizes_per_degree_distribution"]["validation"]
    group_count = len(protocol["data"]["distributions"])
    validations = updates // int(trainer_config["validation_interval_updates"]) + 1
    expected_ledger = {
        "actor_forward_nets": updates * batch_size + validations * group_count * validation_count + 1,
        "surrogate_evaluator_nets": 0,
        "exact_rest_evaluator_nets": updates * batch_size + validations * group_count * validation_count + 2,
        "geosteiner_nets": updates * batch_size + 2,
    }
    stream_hash = EMPTY_STREAM_SHA256
    train_ids_are_isolated = True
    for update in range(1, updates + 1):
        batch = materialize_training_batch(
            protocol=protocol,
            degree=int(result["degree"]),
            training_seed=int(result["training_seed"]),
            update=update,
            batch_size=batch_size,
        )
        stream_hash = extend_stream_hash(stream_hash, batch.bundle_sha256)
        train_ids_are_isolated &= all(
            not item.decode("utf-8").startswith(("validation:", "test:")) for item in batch.net_ids
        )
    validation_events = [event for event in events if event.get("event") == "validation"]
    event_updates = [event["update"] for event in validation_events]
    checks = {
        "result_complete": result["status"] == "complete" and result["completed_updates"] == updates,
        "latest_update_complete": latest["update"] == updates,
        "best_checkpoint_matches_result": best["update"] == result["best"]["update"],
        "checkpoint_identity_matches_frozen_inputs": (
            latest["run_identity"]["protocol_canonical_sha256"] == protocol_sha256(protocol)
            and latest["run_identity"]["trainer_config_sha256"] == canonical_sha256(trainer_config)
        ),
        "call_ledger_exact": result["call_ledger"] == expected_ledger == latest["call_ledger"],
        "policy_update_count_exact": result["policy_update_nets"] == updates * batch_size,
        "training_stream_hash_reproduced": stream_hash == result["training_stream_hash"] == latest["training_stream_hash"],
        "training_ids_exclude_validation_and_test": train_ids_are_isolated,
        "validation_schedule_exact": event_updates == [0, 100, 200],
        "validation_only_selection": all(
            event["result"]["split"] == "validation"
            and not event["result"]["selection"]["test_or_ispd_used"]
            for event in validation_events
        ),
        "inverse_length_audit_passed": result["inverse_length_audit"]["status"] == "pass",
        "all_structured_numeric_results_finite": all_finite(result),
        "no_nan_flag": result["nan_detected"] is False,
        "job_finished_complete": job["status"] == "complete" and task["status"] == "complete",
        "run_lock_released": not (job_root / "run.lock").exists(),
        "only_latest_and_best_checkpoints": sorted(path.name for path in checkpoint_root.glob("*.pt"))
        == ["best.pt", "latest.pt"],
        "no_partial_artifacts": not any(
            path.name.endswith(".partial")
            for root in (checkpoint_root, log_root, job_root)
            for path in root.rglob("*")
        ),
        "checkpoint_hashes_match_result": (
            file_sha256(checkpoint_root / "latest.pt") == result["latest_checkpoint_sha256"]
            and file_sha256(checkpoint_root / "best.pt") == result["best_checkpoint_sha256"]
        ),
    }
    audit = {
        "schema_version": 1,
        "kind": "stage2_erm_smoke_audit",
        "captured_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "run_id": args.run_id,
        "checks": checks,
        "expected_call_ledger": expected_ledger,
        "observed_call_ledger": result["call_ledger"],
        "validation_updates": event_updates,
        "initial_validation": validation_events[0]["result"]["selection"],
        "final_validation": validation_events[-1]["result"]["selection"],
        "latest_checkpoint": str((checkpoint_root / "latest.pt").resolve()),
        "best_checkpoint": str((checkpoint_root / "best.pt").resolve()),
        "event_log": str((log_root / "events.jsonl").resolve()),
        "status": "pass" if all(checks.values()) else "fail",
    }
    atomic_write_json(args.output, audit, mode=0o444)
    print(json.dumps(audit, sort_keys=True))
    return 0 if audit["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
