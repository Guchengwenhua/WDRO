#!/usr/bin/env python3
"""Freeze the exact training-code/config/runtime provenance of the formal batch."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
EXPERIMENTS_ROOT = Path(__file__).resolve().parents[2] / "experiments"
sys.path.insert(0, str(PROJECT_ROOT))

from src.io_utils import atomic_write_json
from src.rest import file_sha256


TRAINING_FILES = [
    "configs/protocol_v1.json",
    "configs/data_contract_v1.json",
    "configs/stage2_trainer_v1.json",
    "src/accounting.py",
    "src/checkpoints.py",
    "src/datasets.py",
    "src/geometry.py",
    "src/geosteiner.py",
    "src/io_utils.py",
    "src/job_control.py",
    "src/metrics.py",
    "src/protocol.py",
    "src/rest.py",
    "src/trainer.py",
    "src/training_data.py",
    "src/validation.py",
    "scripts/capture_host_runtime.py",
    "scripts/train_erm.py",
]


def combined_hash(values: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue-id", default="stage2_erm_formal_v1")
    parser.add_argument("--initial-preflight", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    queue_root = EXPERIMENTS_ROOT / "logs" / "stage2_queue" / args.queue_id
    queue = json.loads((queue_root / "queue.json").read_text(encoding="utf-8"))
    job = json.loads((queue_root / "job.json").read_text(encoding="utf-8"))
    events = [
        json.loads(line)
        for line in (queue_root / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    launch_events = [event for event in events if event.get("event") == "task_launch"]
    hashes = {name: file_sha256(PROJECT_ROOT / name) for name in TRAINING_FILES}
    task_records = []
    for task in queue["tasks"]:
        launches = [event for event in launch_events if event["run_id"] == task["run_id"]]
        task_records.append(
            {
                "run_id": task["run_id"],
                "degree": task["degree"],
                "seed": task["seed"],
                "attempts_at_snapshot": task["attempts"],
                "current_status": task["status"],
                "current_physical_gpu_index": task["physical_gpu_index"],
                "current_pid": task["pid"],
                "current_exact_command": task["exact_command"],
                "current_resume_command": task["resume_command"],
                "stdout_log": task["stdout_log"],
                "launch_history": launches,
            }
        )
    initial_preflight = json.loads(args.initial_preflight.read_text(encoding="utf-8"))
    manifest = {
        "schema_version": 1,
        "kind": "stage2_formal_batch_provenance_snapshot",
        "captured_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "queue_id": args.queue_id,
        "research_protocol_override": False,
        "training_code": {"combined_sha256": combined_hash(hashes), "files": hashes},
        "initial_project_preflight": {
            "path": str(args.initial_preflight),
            "sha256": file_sha256(args.initial_preflight),
            "batch_ready": initial_preflight["batch_ready"],
        },
        "queue_job_at_snapshot": job,
        "memory_profile": {
            "path": "Codex/wdro_rest_min/manifests/stage2_memory_profile_v1.json",
            "sha256": file_sha256(PROJECT_ROOT / "manifests" / "stage2_memory_profile_v1.json"),
        },
        "artifact_root": {
            "logical_path": str(EXPERIMENTS_ROOT),
            "resolved_path": str(EXPERIMENTS_ROOT.resolve()),
        },
        "temporary_and_cache_environment": {
            "TMPDIR": str(EXPERIMENTS_ROOT / "tmp"),
            "TMP": str(EXPERIMENTS_ROOT / "tmp"),
            "TEMP": str(EXPERIMENTS_ROOT / "tmp"),
            "XDG_CACHE_HOME": str(EXPERIMENTS_ROOT / "cache" / "runtime"),
            "TORCH_HOME": str(EXPERIMENTS_ROOT / "cache" / "runtime" / "torch"),
        },
        "task_count": len(task_records),
        "tasks": task_records,
    }
    atomic_write_json(args.output, manifest, mode=0o444)
    sidecar = args.output.with_suffix(args.output.suffix + ".sha256")
    sidecar.write_text(f"{file_sha256(args.output)}  {args.output.name}\n", encoding="ascii")
    sidecar.chmod(0o444)
    print(
        json.dumps(
            {
                "manifest": str(args.output),
                "sha256": file_sha256(args.output),
                "training_code_combined_sha256": manifest["training_code"]["combined_sha256"],
                "task_count": len(task_records),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
