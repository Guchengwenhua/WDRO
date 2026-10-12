"""Verify finished recovery receipts and unchanged task identities; no scoring."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from delivery_audit import quiet_gate, lock_released, process

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
OUT = EXP / "diagnostics/recovery_check_20260912"


def read(path):
    return Path(path).read_text()


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def bind(path):
    return {"path": str(path), "resolved_path": str(Path(path).resolve()), "sha256": digest(path)}


def main():
    baseline_path = next(OUT.glob("runtime_*.json"))
    baseline = json.loads(read(baseline_path))
    verified = []
    for row in baseline["processes"]:
        if row["name"].split("/")[0] not in ("fixed", "joint"):
            continue
        for item in row["bindings"]:
            if item.get("exists") and item.get("sha256"):
                path = Path(item["logical_path"])
                assert digest(path) == item["sha256"], f"task record changed: {path}"
                verified.append(bind(path))
    assert len(verified) == 192
    queue_events = []
    for item in baseline["control_and_log_bindings"]:
        if item["logical_path"].endswith(("queue/fixed/events.jsonl", "queue/joint/events.jsonl")):
            path = Path(item["logical_path"])
            assert digest(path) == item["sha256"], f"queue task event stream changed: {path}"
            queue_events.append(bind(path))
    assert len(queue_events) == 2
    af_folder = EXP / "autofinish"
    af_launch = json.loads(read(af_folder / "launch.json"))
    af_process = process(af_launch)
    assert not af_process["active"], "autofinish still active"
    assert json.loads(read(af_folder / "exit.json"))["automatic_delivery_complete"] is True
    released_af_lock = lock_released(af_folder / "autofinish.lock")
    quiet = quiet_gate()
    receipts = []
    for relative in ("manifests/all_completion.json", "experiments/pipeline/complete.json",
                     "experiments/interpretation/all/extraction.json",
                     "experiments/diagnostics/delivery_audit.json", "experiments/autofinish/complete.json"):
        path = ROOT / relative
        document = json.loads(read(path))
        assert document["status"] == "pass", relative
        receipts.append(bind(path))
    delivery = json.loads(read(EXP / "diagnostics/delivery_audit.json"))
    completion = json.loads(read(ROOT / "manifests/all_completion.json"))
    assert completion["phases"] == {"fixed": {"tasks": 30, "cells": 1080}, "joint": {"tasks": 18, "cells": 216}}
    assert delivery["production_completion"]["sha256"] == digest(ROOT / "manifests/all_completion.json")
    assert delivery["full_audit_performed"] and delivery["oracle_calls"] == 0
    checked_outputs = []
    for relative in ("autofinish/complete.json", "interpretation/all/extraction.json"):
        document = json.loads(read(EXP / relative))
        for item in document.get("source_bindings", []) + document["artifacts"]:
            assert digest(item["path"]) == item["sha256"], f"automatic output binding changed: {item['path']}"
            checked_outputs.append(bind(Path(item["path"])))
    af_exit = json.loads(read(af_folder / "exit.json"))
    assert af_exit["pid"] == af_process["pid"]
    assert af_exit["at"] >= af_launch["at"], "autofinish exit predates current launch"
    result = {
        "status": "pass", "at_unix": time.time(), "phase_coverage": completion["phases"],
        "task_records_preserved": 192, "scientific_tasks_preserved": 48,
        "queue_event_streams_unchanged": 2, "new_scientific_worker_launches": 0,
        "new_scientific_oracle_calls": 0, "source": bind(Path(__file__)),
        "baseline_runtime": bind(baseline_path), "production_receipts": receipts,
        "automatic_output_bindings_verified": checked_outputs,
        "independent_delivery_recursive_bindings": delivery["recursive_bindings_verified"],
        "quiet_gate": quiet, "autofinish_process": af_process,
        "autofinish_lock_released": released_af_lock,
        "preserved_task_bindings": verified, "preserved_queue_events": queue_events,
        "gpu_snapshot": subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,memory.free,utilization.gpu", "--format=csv"], text=True),
        "gpu_processes": subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv"], text=True),
        "scope": "Checks immutable automatic receipts and recovery preservation; full recursive scientific-file audit is bound from delivery_audit.json. Human interpretation has a separate completion receipt."
    }
    target = OUT / "recovery_delivery_verification.json"
    partial = target.with_name(target.name + f".partial.{os.getpid()}")
    partial.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    partial.replace(target)
    print(json.dumps({"status": "pass", "output": str(target), "task_records_preserved": 192,
                      "new_scientific_worker_launches": 0, "new_scientific_oracle_calls": 0,
                      "recursive_bindings": delivery["recursive_bindings_verified"]}))


if __name__ == "__main__":
    main()
