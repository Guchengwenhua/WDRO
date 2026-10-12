#!/usr/bin/env python3
"""Audit all 15 formal ERM runs and emit the stage-2 validation table."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
CODEX_ROOT = WORKSPACE_ROOT / "Codex"
EXPERIMENTS_ROOT = CODEX_ROOT / "experiments"
sys.path.insert(0, str(PROJECT_ROOT))

from src.checkpoints import load_checkpoint
from src.io_utils import atomic_write_json
from src.job_control import JobLease
from src.protocol import load_protocol, protocol_sha256
from src.rest import file_sha256
from src.trainer import canonical_sha256, load_trainer_config
from src.stage2_audit_checks import all_finite, optimizer_checks as positive_update_optimizer_checks, rebuild_stream, validation_checks


def optimizer_checks(checkpoint, trainer):
    """Adam creates moment/step state lazily on its first optimizer update."""
    checks = positive_update_optimizer_checks(checkpoint, trainer)
    if checkpoint["update"] == 0:
        optimizer = checkpoint["optimizer_state_dict"]
        checks["optimizer_steps_match_checkpoint"] = (
            optimizer["state"] == {}
            and bool(optimizer["param_groups"])
            and all(group["params"] for group in optimizer["param_groups"])
        )
    return checks


def load_events(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def atomic_write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        temporary = Path(temporary_name)
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue-id", default="stage2_erm_formal_v1")
    parser.add_argument("--skip-stream-rebuild", action="store_true")
    parser.add_argument("--smoke-run", help="Audit just this run; never marks the stage passed")
    parser.add_argument("--host-runtime-manifest", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=EXPERIMENTS_ROOT / "logs" / "stage2" / "stage2_formal_audit.json",
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=EXPERIMENTS_ROOT / "tables" / "stage2_erm_validation_summary.csv",
    )
    args = parser.parse_args()
    if (args.smoke_run or args.skip_stream_rebuild) and args.output.name == "stage2_formal_audit.json":
        parser.error("partial audits must use a separate --output")
    command = shlex.join([sys.executable, str(Path(__file__).absolute()), *sys.argv[1:]])
    job_root = args.output.parent / (args.output.stem + "_job")
    with JobLease(job_root, run_id=args.output.stem, exact_command=command, resume_command=command,
                  monitor_command=f"tail -f {job_root / 'heartbeat.json'}", expected_updates=30000,
                  physical_gpu_index=None, host_runtime_manifest=str(args.host_runtime_manifest)) as lease:
        code = run_audit(args, lease)
        lease.finish(update=2000 if args.smoke_run else 30000, status="complete" if code == 0 else "failed")
        return code


def run_audit(args, lease) -> int:
    protocol = load_protocol()
    trainer_config = load_trainer_config()
    protocol_hash = protocol_sha256(protocol)
    trainer_hash = canonical_sha256(trainer_config)
    updates = int(protocol["training"]["policy_updates"])
    batch_size = int(protocol["training"]["batch_size"])
    group_count = len(protocol["data"]["distributions"])
    validation_count = int(protocol["data"]["sizes_per_degree_distribution"]["validation"])
    validation_interval = int(trainer_config["validation_interval_updates"])
    validation_evaluations = 1 + updates // validation_interval
    expected_ledger = {
        "actor_forward_nets": updates * batch_size + validation_evaluations * group_count * validation_count + 1,
        "surrogate_evaluator_nets": 0,
        "exact_rest_evaluator_nets": updates * batch_size + validation_evaluations * group_count * validation_count + 2,
        "geosteiner_nets": updates * batch_size + 2,
    }
    queue_root = EXPERIMENTS_ROOT / "logs" / "stage2_queue" / args.queue_id
    queue = json.loads((queue_root / "queue.json").read_text(encoding="utf-8"))
    queue_job = json.loads((queue_root / "job.json").read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    run_checks: dict[str, dict[str, bool]] = {}
    stream_artifacts = {}
    provenance = json.loads((PROJECT_ROOT / "manifests/stage2_formal_batch_20260904T1801Z.json").read_text())
    code_checks = {name: file_sha256(PROJECT_ROOT / name) == expected
                   for name, expected in provenance["training_code"]["files"].items()}
    original_preflight = json.loads(Path(provenance["initial_project_preflight"]["path"]).read_text())
    legacy_checks = {name: file_sha256(WORKSPACE_ROOT / name) == expected for name, expected in
                     (original_preflight["code"]["legacy_assets"] | original_preflight["checkpoints"]).items()}
    if not all(code_checks.values()) or not all(legacy_checks.values()):
        raise RuntimeError(f"frozen training sources changed: {code_checks}, legacy: {legacy_checks}")
    for degree in protocol["scope"]["degrees"]:
        for seed in protocol["data"]["seeds"]["training"]:
            run_id = f"formal_erm_d{degree}_s{seed}_u{updates}"
            if args.smoke_run and args.smoke_run != run_id:
                continue
            checkpoint_root = EXPERIMENTS_ROOT / "checkpoints" / "stage2" / run_id
            log_root = EXPERIMENTS_ROOT / "logs" / "stage2" / run_id
            job_root = EXPERIMENTS_ROOT / "logs" / "jobs" / run_id
            result = json.loads((log_root / "result.json").read_text(encoding="utf-8"))
            latest = load_checkpoint(checkpoint_root / "latest.pt", map_location="cpu")
            best = load_checkpoint(checkpoint_root / "best.pt", map_location="cpu")
            events = load_events(log_root / "events.jsonl")
            validation_events = [event for event in events if event.get("event") == "validation"]
            initial = next(event for event in validation_events if event["update"] == 0)["result"]
            final = next(event for event in reversed(validation_events) if event["update"] == updates)["result"]
            stream_reproduced = False
            separation_passed = False
            if not args.skip_stream_rebuild:
                stream = rebuild_stream(
                    protocol=protocol, degree=degree, seed=seed, result=result,
                    root=EXPERIMENTS_ROOT / "logs/stage2/formal_stream_audit_v1",
                    dataset_root=EXPERIMENTS_ROOT / "cache/datasets/protocol_v1",
                    heartbeat=lambda update, path: lease.update(
                        update=len(records) * updates + update, status="running",
                        detail={"run_id": run_id, "stream_update": update, "cursor": path}),
                )
                stream_reproduced = (stream["status"] == "pass" and stream["completed_updates"] == updates
                                     and stream["stream_hash"] == latest["training_stream_hash"])
                separation_passed = stream["id_mismatches"] == stream["heldout_content_intersections"] == 0
                stream_artifacts[run_id] = {key: stream[key] for key in ("logical_path", "resolved_path")}
            job = json.loads((job_root / "job.json").read_text(encoding="utf-8"))
            task = json.loads((job_root / "task.json").read_text(encoding="utf-8"))
            partials = [
                str(path)
                for root in (checkpoint_root, log_root, job_root)
                for path in root.rglob("*.partial")
            ]
            checks = {
                "complete_2000_updates": result["status"] == "complete" and result["completed_updates"] == updates,
                "latest_checkpoint_complete": latest["update"] == updates,
                "best_checkpoint_matches_result": best["update"] == result["best"]["update"],
                "frozen_identity_matches": (
                    latest["run_identity"]["protocol_canonical_sha256"] == protocol_hash
                    and latest["run_identity"]["trainer_config_sha256"] == trainer_hash
                ),
                "call_ledger_exact": result["call_ledger"] == expected_ledger == latest["call_ledger"],
                "training_stream_reproduced": stream_reproduced,
                "training_ids_and_content_disjoint_from_validation_test": separation_passed,
                "all_train_and_validation_events_finite": all_finite(events) and result["nan_detected"] is False,
                "exact_budget_and_checkpoint_identities": (
                    latest["run_identity"] == best["run_identity"] and latest["best"] == best["best"] == result["best"]
                    and latest["run_identity"]["method"] == "erm"
                    and latest["run_identity"]["degree"] == result["degree"] == degree
                    and latest["run_identity"]["training_seed"] == result["training_seed"] == seed
                    and latest["run_identity"]["expected_updates"] == updates
                    and latest["run_identity"]["batch_size"] == result["batch_size"] == batch_size
                    and result["policy_update_nets"] == updates * batch_size),
                "validation_zero_and_final_present": initial["split"] == "validation" and final["split"] == "validation",
                "no_test_or_ispd_selection": (
                    result["test_or_ispd_used"] is False
                    and all(not event["result"]["selection"]["test_or_ispd_used"] for event in validation_events)
                ),
                "inverse_audit_passed": result["inverse_length_audit"]["status"] == "pass",
                "finite_metrics": all(
                    math.isfinite(float(value))
                    for value in (
                        initial["selection"]["nominal_mean"],
                        initial["selection"]["worst_group_mean"],
                        final["selection"]["nominal_mean"],
                        final["selection"]["worst_group_mean"],
                        result["best"]["nominal_mean"],
                        result["best"]["selection_objective"],
                    )
                ),
                "best_objective_not_worse_than_initial": (
                    result["best"]["selection_objective"] <= initial["selection"]["worst_group_mean"] + 1e-15
                ),
                "best_nominal_within_reasonable_gate": (
                    result["best"]["nominal_mean"]
                    <= initial["selection"]["nominal_mean"]
                    + protocol["training"]["selection"]["nominal_mean_regret_degradation_max_absolute"]
                ),
                "job_and_task_complete": job["status"] == "complete" and task["status"] == "complete",
                "lock_released": not (job_root / "run.lock").exists(),
                "only_latest_and_best": sorted(path.name for path in checkpoint_root.glob("*.pt"))
                == ["best.pt", "latest.pt"],
                "no_partial_artifacts": not partials,
                "checkpoint_hashes_match": (
                    file_sha256(checkpoint_root / "latest.pt") == result["latest_checkpoint_sha256"]
                    and file_sha256(checkpoint_root / "best.pt") == result["best_checkpoint_sha256"]
                ),
            }
            checks.update(validation_checks(validation_events, best, protocol, trainer_config, degree))
            checks.update({f"latest_{name}": passed for name, passed in optimizer_checks(latest, trainer_config).items()})
            checks.update({f"best_{name}": passed for name, passed in optimizer_checks(best, trainer_config).items()})
            run_checks[run_id] = checks
            print(json.dumps({"run_id": run_id, "passed": all(checks.values()),
                              "failed_checks": [key for key, passed in checks.items() if not passed]}), flush=True)
            segment_wall_seconds = sum(
                json.loads(path.read_text(encoding="utf-8"))["segment_wall_seconds"]
                for path in log_root.glob("segment_*.json")
            )
            records.append(
                {
                    "run_id": run_id,
                    "degree": degree,
                    "training_seed": seed,
                    "initial_nominal_mean": initial["selection"]["nominal_mean"],
                    "initial_ood_worst_group_mean": initial["selection"]["worst_group_mean"],
                    "best_update": result["best"]["update"],
                    "best_nominal_mean": result["best"]["nominal_mean"],
                    "best_ood_worst_group_mean": result["best"]["selection_objective"],
                    "final_nominal_mean": final["selection"]["nominal_mean"],
                    "final_ood_worst_group_mean": final["selection"]["worst_group_mean"],
                    "peak_memory_reserved_mib": round(result["peak_memory_reserved_bytes"] / 1024**2, 3),
                    "completed_segment_wall_seconds": round(segment_wall_seconds, 6),
                    "attempts": next(task["attempts"] for task in queue["tasks"] if task["run_id"] == run_id),
                    "final_physical_gpu_index": result["physical_gpu_index"],
                    "latest_checkpoint_sha256": result["latest_checkpoint_sha256"],
                    "best_checkpoint_sha256": result["best_checkpoint_sha256"],
                    "run_audit_passed": all(checks.values()),
                }
            )

    matrix = {(record["degree"], record["training_seed"]) for record in records}
    expected_matrix = {
        (degree, seed)
        for degree in protocol["scope"]["degrees"]
        for seed in protocol["data"]["seeds"]["training"]
    }
    top_checks = {
        "queue_complete": queue.get("status") == "complete" and queue_job.get("status") == "complete",
        "expected_run_matrix": (len(records) == 1 and records[0]["run_id"] == args.smoke_run)
        if args.smoke_run else len(records) == 15 and matrix == expected_matrix,
        "all_run_audits_passed": all(all(checks.values()) for checks in run_checks.values()),
        "three_seeds_per_degree": bool(args.smoke_run) or all(
            sum(record["degree"] == degree for record in records) == 3
            for degree in protocol["scope"]["degrees"]
        ),
        "no_nan_any_run": all(
            checks["all_train_and_validation_events_finite"]
            and checks["finite_metrics"]
            and checks["latest_model_optimizer_rng_finite"]
            and checks["best_model_optimizer_rng_finite"]
            for checks in run_checks.values()
        ),
        "training_sources_and_frozen_configs_unchanged": all(code_checks.values()),
        "legacy_assets_and_dac21_checkpoints_unchanged": all(legacy_checks.values()),
        "full_stream_rebuild_not_skipped": not args.skip_stream_rebuild,
    }
    audit = {
        "schema_version": 1,
        "kind": "stage2_formal_erm_audit",
        "captured_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "queue_id": args.queue_id,
        "protocol_canonical_sha256": protocol_hash,
        "trainer_config_canonical_sha256": trainer_hash,
        "expected_call_ledger_per_run": expected_ledger,
        "checks": top_checks,
        "run_checks": run_checks,
        "training_source_checks": code_checks,
        "legacy_asset_checks": legacy_checks,
        "stream_artifacts": stream_artifacts,
        "audit_scope": "one_run_smoke" if args.smoke_run else "15_formal_runs",
        "host_runtime_manifest": str(args.host_runtime_manifest),
        "artifact_root": {"logical_path": str(EXPERIMENTS_ROOT), "resolved_path": str(EXPERIMENTS_ROOT.resolve())},
        "accounting_scope": "effective checkpoint trajectory; failed/replayed work is not included in call ledger or completed_segment_wall_seconds",
        "audit_code_sha256": file_sha256(Path(__file__)),
        "records": records,
        "table_path": str(args.table),
        "status": ("smoke_pass" if args.smoke_run else "pass") if all(top_checks.values()) else "fail",
    }
    atomic_write_csv(args.table, records)
    atomic_write_json(args.output, audit, mode=0o444)
    print(
        json.dumps(
            {
                "status": audit["status"],
                "output": str(args.output),
                "table": str(args.table),
                "checks": top_checks,
            },
            sort_keys=True,
        )
    )
    return 0 if audit["status"] in {"pass", "smoke_pass"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
