#!/usr/bin/env python3
"""Persistent dynamic-preflight queue for the 15 formal stage-2 ERM runs."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
CODEX_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_ROOT = CODEX_ROOT / "experiments"
CAPTURE_SCRIPT = PROJECT_ROOT / "scripts" / "capture_host_runtime.py"
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "train_erm.py"
MEMORY_PROFILE_PATH = PROJECT_ROOT / "manifests" / "stage2_memory_profile_v1.json"
PROTOCOL_PATH = PROJECT_ROOT / "configs" / "protocol_v1.json"

sys.path.insert(0, str(PROJECT_ROOT))

from src.io_utils import append_jsonl, atomic_write_json
from src.job_control import process_is_alive, utc_now


def gpu_snapshot() -> dict[str, Any]:
    gpu_command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu,utilization.memory",
        "--format=csv,noheader,nounits",
    ]
    process_command = [
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    gpu_result = subprocess.run(gpu_command, check=True, capture_output=True, text=True)
    process_result = subprocess.run(process_command, check=True, capture_output=True, text=True)
    gpus = []
    for line in gpu_result.stdout.splitlines():
        index, uuid, used, free, util_gpu, util_memory = [part.strip() for part in line.split(",")]
        gpus.append(
            {
                "index": int(index),
                "uuid": uuid,
                "memory_used_mib": int(used),
                "memory_free_mib": int(free),
                "utilization_gpu_percent": int(util_gpu),
                "utilization_memory_percent": int(util_memory),
            }
        )
    processes = []
    for line in process_result.stdout.splitlines():
        if not line.strip():
            continue
        uuid, pid, name, memory = [part.strip() for part in line.split(",")]
        processes.append(
            {"gpu_uuid": uuid, "pid": int(pid), "process_name": name, "used_memory_mib": int(memory)}
        )
    return {"captured_at_utc": utc_now(), "gpus": gpus, "compute_processes": processes}


def build_tasks() -> list[dict[str, Any]]:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    tasks = []
    for degree in sorted(protocol["scope"]["degrees"], reverse=True):
        for seed in protocol["data"]["seeds"]["training"]:
            run_id = f"formal_erm_d{degree}_s{seed}_u{protocol['training']['policy_updates']}"
            tasks.append(
                {
                    "run_id": run_id,
                    "degree": degree,
                    "seed": seed,
                    "status": "pending",
                    "attempts": 0,
                    "physical_gpu_index": None,
                    "pid": None,
                    "host_runtime_manifest": None,
                    "stdout_log": str(
                        EXPERIMENTS_ROOT / "logs" / "stage2_queue" / "TASK_STDOUT_PLACEHOLDER" / f"{run_id}.log"
                    ),
                    "exact_command": None,
                    "resume_command": None,
                    "last_error": None,
                }
            )
    return tasks


def result_path(task: dict[str, Any]) -> Path:
    return EXPERIMENTS_ROOT / "logs" / "stage2" / task["run_id"] / "result.json"


def latest_checkpoint_path(task: dict[str, Any]) -> Path:
    return EXPERIMENTS_ROOT / "checkpoints" / "stage2" / task["run_id"] / "latest.pt"


def task_completed(task: dict[str, Any]) -> bool:
    path = result_path(task)
    if not path.is_file():
        return False
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return result.get("status") == "complete" and result.get("completed_updates") == 2000


def acquire_lock(path: Path, queue_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as error:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            pid = int(value.get("pid", -1))
        except Exception:
            pid = -1
        if process_is_alive(pid):
            raise RuntimeError(f"queue is already running under PID {pid}") from error
        stale = path.with_name(f"queue.lock.stale.{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
        os.replace(path, stale)
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump({"queue_id": queue_id, "pid": os.getpid(), "created_at_utc": utc_now()}, handle)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def run_preflight(*, queue_id: str, task: dict[str, Any], peak_mib: int, safety_mib: int) -> dict[str, Any] | None:
    purpose = f"{queue_id}_launch_{task['run_id']}_attempt_{task['attempts'] + 1}"
    command = [
        sys.executable,
        str(CAPTURE_SCRIPT),
        "--purpose",
        purpose,
        "--samples",
        "2",
        "--sample-interval-seconds",
        "1",
        "--measured-peak-mib",
        str(peak_mib),
        "--safety-margin-mib",
        str(safety_mib),
    ]
    completed = subprocess.run(command, cwd=WORKSPACE_ROOT, capture_output=True, text=True)
    if not completed.stdout.strip():
        return None
    summary = json.loads(completed.stdout.strip().splitlines()[-1])
    manifest_path = Path(summary["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {"returncode": completed.returncode, "summary": summary, "manifest": manifest, "path": manifest_path}


def choose_gpu(
    preflight: dict[str, Any], active_counts: dict[int, int], excluded_gpu_indices: set[int]
) -> int | None:
    if not preflight["manifest"].get("preflight_passed"):
        return None
    decisions = {row["index"]: row for row in preflight["manifest"]["gpu_decisions"]}
    eligible = [
        index
        for index in preflight["manifest"]["eligible_gpu_indices"]
        if index not in excluded_gpu_indices
    ]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda index: (
            active_counts.get(index, 0),
            -decisions[index]["minimum_free_memory_mib"],
            decisions[index]["average_gpu_utilization_percent"],
            index,
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue-id", default="stage2_erm_formal_v1")
    parser.add_argument("--max-simultaneous", type=int, default=15)
    parser.add_argument("--geosteiner-workers", type=int, default=3)
    parser.add_argument("--launch-stagger-seconds", type=float, default=3.0)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--exclude-gpu-indices", type=int, nargs="*", default=[])
    args = parser.parse_args()
    if args.max_simultaneous <= 0 or args.geosteiner_workers <= 0:
        parser.error("concurrency and worker counts must be positive")
    excluded_gpu_indices = set(args.exclude_gpu_indices)
    queue_root = EXPERIMENTS_ROOT / "logs" / "stage2_queue" / args.queue_id
    queue_root.mkdir(parents=True, exist_ok=True)
    queue_path = queue_root / "queue.json"
    heartbeat_path = queue_root / "heartbeat.json"
    events_path = queue_root / "events.jsonl"
    job_path = queue_root / "job.json"
    lock_path = queue_root / "queue.lock"
    acquire_lock(lock_path, args.queue_id)
    memory_profile = json.loads(MEMORY_PROFILE_PATH.read_text(encoding="utf-8"))
    safety_mib = int(memory_profile["per_gpu_safety_margin_mib"])
    exact_queue_command = shlex.join([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]])
    job = {
        "schema_version": 1,
        "queue_id": args.queue_id,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "tmux_session": os.environ.get("WDRO_TMUX_SESSION"),
        "started_at_utc": utc_now(),
        "exact_command": exact_queue_command,
        "monitor_command": f"tail -f {shlex.quote(str(events_path))}",
        "recovery_command": exact_queue_command,
        "queue_manifest": str(queue_path),
        "heartbeat": str(heartbeat_path),
        "events": str(events_path),
        "status": "running",
        "runtime_excluded_gpu_indices": sorted(excluded_gpu_indices),
    }
    atomic_write_json(job_path, job)

    if queue_path.exists():
        queue = json.loads(queue_path.read_text(encoding="utf-8"))
        tasks = queue["tasks"]
    else:
        tasks = build_tasks()
        for task in tasks:
            task["stdout_log"] = str(queue_root / f"{task['run_id']}.log")
        queue = {
            "schema_version": 1,
            "queue_id": args.queue_id,
            "created_at_utc": utc_now(),
            "protocol_path": str(PROTOCOL_PATH.relative_to(WORKSPACE_ROOT)),
            "memory_profile_path": str(MEMORY_PROFILE_PATH.relative_to(WORKSPACE_ROOT)),
            "artifact_root_logical": str(EXPERIMENTS_ROOT),
            "artifact_root_resolved": str(EXPERIMENTS_ROOT.resolve()),
            "max_simultaneous": args.max_simultaneous,
            "geosteiner_workers_per_run": args.geosteiner_workers,
            "launch_stagger_seconds": args.launch_stagger_seconds,
            "tasks": tasks,
            "status": "running",
        }
        atomic_write_json(queue_path, queue)

    children: dict[str, subprocess.Popen[Any]] = {}
    try:
        while True:
            active_counts: dict[int, int] = {}
            for task in tasks:
                if task_completed(task):
                    task["status"] = "complete"
                    task["pid"] = None
                    continue
                if task["status"] == "running":
                    child = children.get(task["run_id"])
                    alive = child.poll() is None if child is not None else process_is_alive(int(task.get("pid") or -1))
                    if alive:
                        gpu = int(task["physical_gpu_index"])
                        active_counts[gpu] = active_counts.get(gpu, 0) + 1
                        continue
                    returncode = child.returncode if child is not None else None
                    task["last_error"] = f"training process exited before completion (returncode={returncode})"
                    task["status"] = "pending" if task["attempts"] < args.max_attempts else "failed"
                    append_jsonl(
                        events_path,
                        {"event": "task_exit_incomplete", "at_utc": utc_now(), "run_id": task["run_id"], "returncode": returncode},
                    )

            complete_count = sum(task["status"] == "complete" for task in tasks)
            failed_count = sum(task["status"] == "failed" for task in tasks)
            running_count = sum(task["status"] == "running" for task in tasks)
            pending = [task for task in tasks if task["status"] == "pending"]
            if complete_count == len(tasks):
                queue["status"] = "complete"
                queue["completed_at_utc"] = utc_now()
                atomic_write_json(queue_path, queue)
                job.update({"status": "complete", "completed_at_utc": utc_now()})
                atomic_write_json(job_path, job)
                atomic_write_json(
                    heartbeat_path,
                    {"queue_id": args.queue_id, "status": "complete", "completed": complete_count, "total": len(tasks), "at_utc": utc_now()},
                )
                break
            if failed_count and not pending and running_count == 0:
                queue["status"] = "failed"
                atomic_write_json(queue_path, queue)
                job.update({"status": "failed", "finished_at_utc": utc_now()})
                atomic_write_json(job_path, job)
                return 2

            launched = False
            while pending and running_count < args.max_simultaneous:
                task = pending.pop(0)
                profile = memory_profile["profiles"][str(task["degree"])]
                preflight = run_preflight(
                    queue_id=args.queue_id,
                    task=task,
                    peak_mib=int(profile["capacity_peak_mib"]),
                    safety_mib=safety_mib,
                )
                if preflight is None:
                    task["last_error"] = "host-runtime preflight produced no manifest"
                    break
                gpu = choose_gpu(preflight, active_counts, excluded_gpu_indices)
                if gpu is None:
                    task["last_error"] = (
                        "no GPU passed the fresh capacity/utilization preflight after current-batch exclusions"
                    )
                    append_jsonl(
                        events_path,
                        {
                            "event": "launch_deferred",
                            "at_utc": utc_now(),
                            "run_id": task["run_id"],
                            "host_runtime_manifest": str(preflight["path"]),
                            "reason": task["last_error"],
                        },
                    )
                    break
                command = [
                    sys.executable,
                    str(TRAIN_SCRIPT),
                    "--degree",
                    str(task["degree"]),
                    "--seed",
                    str(task["seed"]),
                    "--scope",
                    "formal",
                    "--device",
                    "cuda",
                    "--physical-gpu-index",
                    str(gpu),
                    "--host-runtime-manifest",
                    str(preflight["path"]),
                    "--run-id",
                    task["run_id"],
                    "--geosteiner-workers",
                    str(args.geosteiner_workers),
                ]
                if latest_checkpoint_path(task).is_file():
                    command.append("--resume")
                stdout_path = Path(task["stdout_log"])
                stdout_path.parent.mkdir(parents=True, exist_ok=True)
                output_handle = stdout_path.open("a", encoding="utf-8")
                environment = dict(os.environ)
                environment["WDRO_TMUX_SESSION"] = os.environ.get("WDRO_TMUX_SESSION", "")
                child = subprocess.Popen(
                    command,
                    cwd=WORKSPACE_ROOT,
                    env=environment,
                    stdout=output_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                )
                output_handle.close()
                task.update(
                    {
                        "status": "running",
                        "attempts": int(task["attempts"]) + 1,
                        "physical_gpu_index": gpu,
                        "pid": child.pid,
                        "host_runtime_manifest": str(preflight["path"]),
                        "exact_command": shlex.join(command),
                        "resume_command": shlex.join([*command, "--resume"] if "--resume" not in command else command),
                        "launched_at_utc": utc_now(),
                        "last_error": None,
                    }
                )
                children[task["run_id"]] = child
                active_counts[gpu] = active_counts.get(gpu, 0) + 1
                running_count += 1
                launched = True
                append_jsonl(
                    events_path,
                    {
                        "event": "task_launch",
                        "at_utc": utc_now(),
                        "run_id": task["run_id"],
                        "degree": task["degree"],
                        "seed": task["seed"],
                        "physical_gpu_index": gpu,
                        "pid": child.pid,
                        "host_runtime_manifest": str(preflight["path"]),
                        "exact_command": task["exact_command"],
                    },
                )
                queue["last_updated_at_utc"] = utc_now()
                atomic_write_json(queue_path, queue)
                time.sleep(args.launch_stagger_seconds)
                append_jsonl(
                    events_path,
                    {
                        "event": "post_launch_gpu_sample",
                        "at_utc": utc_now(),
                        "run_id": task["run_id"],
                        "physical_gpu_index": gpu,
                        "pid": child.pid,
                        "process_returncode": child.poll(),
                        "snapshot": gpu_snapshot(),
                    },
                )

            complete_count = sum(task["status"] == "complete" for task in tasks)
            failed_count = sum(task["status"] == "failed" for task in tasks)
            running_count = sum(task["status"] == "running" for task in tasks)
            pending_count = sum(task["status"] == "pending" for task in tasks)
            queue["last_updated_at_utc"] = utc_now()
            queue["counts"] = {
                "complete": complete_count,
                "running": running_count,
                "pending": pending_count,
                "failed": failed_count,
                "total": len(tasks),
            }
            atomic_write_json(queue_path, queue)
            atomic_write_json(
                heartbeat_path,
                {
                    "queue_id": args.queue_id,
                    "pid": os.getpid(),
                    "status": "running",
                    "counts": queue["counts"],
                    "last_update_utc": utc_now(),
                    "active_gpu_counts": active_counts,
                },
            )
            time.sleep(args.poll_seconds if not launched else min(args.poll_seconds, 5.0))
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
