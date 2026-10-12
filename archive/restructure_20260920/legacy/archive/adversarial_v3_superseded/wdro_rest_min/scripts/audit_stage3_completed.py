#!/usr/bin/env python3
"""Audit completed stage-3 runs early, at low CPU priority, using the final auditor.

The pipeline waits for this job before its final audit, which independently
rechecks every result and reuses the hash-bound input reconstruction cursors.
"""
import argparse
import json
import os
import shlex
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODEX = ROOT.parent
EXP = CODEX / "experiments"
sys.path.insert(0, str(ROOT))
from src.io_utils import atomic_write_json, append_jsonl
from src.job_control import JobLease, utc_now, process_is_alive
from src.rest import file_sha256
from audit_stage3 import audit_run
from run_stage3_baseline_queue import build_tasks, task_completed, result_path
from stage3_pipeline import snapshot_cpu, verify_sources

JOB = EXP / "logs/stage3_completed_audit"
QUEUE = EXP / "logs/stage3_queue/stage3_baselines_formal_v1/queue.json"


def stop_training(reason):
    """On an audit failure, stop admission and safely checkpoint only our runs."""
    queue = json.loads(QUEUE.read_text())
    targets = []
    controller = EXP / "logs/stage3_pipeline/pid.json"
    scheduler = EXP / "logs/stage3_queue/stage3_baselines_formal_v1/job.json"
    for path, script in ((scheduler, "run_stage3_baseline_queue.py"),
                         (controller, "stage3_pipeline.py")):
        if path.exists():
            targets.append((json.loads(path.read_text()).get("pid"), script))
    targets.extend((t.get("pid"), "train_baseline.py") for t in queue["tasks"]
                   if t["status"] == "running")
    stopped = []
    for pid, script in targets:
        cmdline = Path(f"/proc/{pid}/cmdline")
        if pid and cmdline.exists() and str(ROOT / "scripts" / script).encode() in cmdline.read_bytes():
            try:
                os.kill(int(pid), signal.SIGTERM)
                stopped.append({"pid": pid, "script": script})
            except ProcessLookupError:
                pass
    atomic_write_json(JOB / "failure.json", {"at_utc": utc_now(), "reason": reason,
        "stop_requested": stopped, "recovery": "Resolve the audit failure before launch_stage3.py; training resumes from latest."})


def update_progress(checked):
    path = CODEX / "01_WDRO-REST_项目进度.md"
    marker = "- 完成项提前审计："
    line = (f"{marker}{len(checked)}/45 正式 run 已通过原完整审计器（{utc_now()}）；"
            "证据 `experiments/logs/stage3_completed_audit/result.json` 与 `experiments/logs/stage3_audit/runs/`。"
            "每个已完成项均全量重建训练/候选输入，最终仍须 45-run 总审计和导出。")
    lines = path.read_text().splitlines()
    for i, old in enumerate(lines):
        if old.startswith(marker):
            lines[i] = line
            break
    else:
        index = next(i for i, old in enumerate(lines) if old.startswith("- 3.3 自动验收步骤"))
        lines.insert(index + 1, line)
    temporary = path.with_name(path.name + ".completed_audit.partial")
    temporary.write_text("\n".join(lines) + "\n")
    os.replace(temporary, path)


def run_audit_task(task, lease, checked_count, *, poll_seconds=15, handoff_when_training_complete=True):
    """Yield an active low-priority audit at its durable cursor when training ends.

    Only this executor's CPU worker is terminated. The frozen auditor writes a
    hash-bound cursor atomically every 25 updates, so the final auditor resumes
    without accepting an incomplete result or changing any check.
    """
    import concurrent.futures
    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as pool:
        future = pool.submit(audit_run, task)
        while not future.done():
            if handoff_when_training_complete and json.loads(QUEUE.read_text()).get("status") == "complete":
                # Check completion again before interpreting a worker exception
                # as an intentional handoff.
                if future.done():
                    return future.result()
                workers = list(pool._processes.values())
                for worker in workers:
                    worker.terminate()
                for worker in workers:
                    worker.join(timeout=5)
                    if worker.is_alive():
                        raise RuntimeError("CPU audit worker did not stop for final handoff")
                cursor = EXP / "logs/stage3_audit/cursors" / f"{task['run_id']}.json"
                saved = json.loads(cursor.read_text()) if cursor.exists() else None
                record = {"at_utc": utc_now(), "status": "paused_for_final_audit", "run_id": task["run_id"],
                    "workers": [{"pid": w.pid, "exitcode": w.exitcode} for w in workers],
                    "cursor_logical": str(cursor), "cursor_resolved": str(cursor.resolve()),
                    "cursor_completed_updates": saved["completed_updates"] if saved else 0,
                    "full_audit_still_required": True, "training_processes_signaled": False}
                atomic_write_json(JOB / "handoff.json", record)
                append_jsonl(JOB / "events.jsonl", {"event": "final_audit_handoff", **record})
                return None
            lease.update(update=checked_count, status="running", detail={"auditing": task["run_id"],
                "cursor": str(EXP / "logs/stage3_audit/cursors" / f"{task['run_id']}.json")})
            concurrent.futures.wait([future], timeout=poll_seconds)
        return future.result()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--max-runs", type=int)
    args = parser.parse_args()
    if (JOB / "failure.json").exists():
        raise RuntimeError("Unresolved completed-run audit failure; inspect failure.json before recovery")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.nice(10)
    verify_sources()
    command = shlex.join(["bash", str(ROOT / "scripts/stage3_env.sh"), sys.executable,
                         str(Path(__file__)), *sys.argv[1:]])
    tasks = build_tasks()
    with JobLease(JOB, run_id="stage3_completed_audit", exact_command=command,
                  resume_command=command, monitor_command=f"cat {JOB / 'heartbeat.json'}",
                  expected_updates=len(tasks), physical_gpu_index=None,
                  host_runtime_manifest=os.environ.get("WDRO_HOST_RUNTIME_MANIFEST")) as lease:
        checked = set()
        for task in tasks:
            path = EXP / "logs/stage3_audit/runs" / f"{task['run_id']}.json"
            if path.exists() and task_completed(task):
                old = json.loads(path.read_text())
                if old["status"] == "pass" and old["result_sha256"] == file_sha256(result_path(task)):
                    checked.add(task["run_id"])
        newly_checked = 0
        while True:
            # Once training is complete, hand the remaining runs to the final
            # eight-worker auditor instead of retaining this one-worker queue.
            if not args.once and json.loads(QUEUE.read_text()).get("status") == "complete":
                break
            candidates = [t for t in tasks if t["run_id"] not in checked and task_completed(t)
                          and not (EXP / "logs/jobs" / t["run_id"] / "run.lock").exists()]
            if candidates:
                task = candidates[0]
                runtime = snapshot_cpu("stage3_completed_run_audit_" + task["run_id"])
                atomic_write_json(JOB / "active_input.json", {"run_id": task["run_id"],
                    "input": str(result_path(task)), "input_sha256": file_sha256(result_path(task)),
                    "runtime_manifest": str(runtime), "at_utc": utc_now(), "device": "cpu", "nice": os.nice(0)})
                append_jsonl(JOB / "events.jsonl", {"event": "audit_start", "run_id": task["run_id"],
                    "at_utc": utc_now(), "runtime_manifest": str(runtime)})
                try:
                    output = run_audit_task(task, lease, len(checked), handoff_when_training_complete=not args.once)
                except Exception as error:
                    stop_training(f"{task['run_id']}: {type(error).__name__}: {error}")
                    raise
                if output is None:
                    break
                checked.add(task["run_id"])
                newly_checked += 1
                append_jsonl(JOB / "events.jsonl", {"event": "audit_pass", **output})
                report = {"at_utc": utc_now(), "status": "partial_pass", "runs_passed": sorted(checked),
                    "total": len(tasks), "stage3_complete": False, "final_audit_required": True}
                atomic_write_json(JOB / "result.json", report)
                update_progress(checked)
                print(json.dumps({"run_id": task["run_id"], "status": "pass", "audited": len(checked),
                                  "checks": len(output["checks"])}), flush=True)
                if args.max_runs and newly_checked >= args.max_runs:
                    break
                continue
            if args.once or len(checked) == len(tasks):
                break
            queue = json.loads(QUEUE.read_text())
            if queue.get("status") == "failed":
                raise RuntimeError("Training queue failed; inspect queue before continuing")
            lease.update(update=len(checked), status="running", detail={"waiting_for_completed_training": True})
            time.sleep(15)
        lease.finish(update=len(checked), status="complete", detail={"result": str(JOB / "result.json"),
                     "final_audit_required": True})


if __name__ == "__main__":
    main()
