"""Safe, persistent GPU queue with restartable worker supervision (stdlib only)."""
from __future__ import annotations

import argparse
from collections import deque
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
OPS = ROOT / "artifacts/operations"
STOP = False


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.partial")
    with tmp.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def local_path(value):
    path = Path(value)
    path = path if path.is_absolute() else ROOT / path
    path.resolve().relative_to(ROOT.resolve())
    return path


def identity(pid):
    """Parse /proc robustly even when the process name contains whitespace."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        fields = stat[stat.rfind(")") + 2:].split()
        if fields[0] == "Z":
            return None
        return {"pid": int(pid), "start_ticks": fields[19],
                "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}
    except (FileNotFoundError, ProcessLookupError):
        return None


def is_alive(record):
    current = identity(record["pid"])
    return current is not None and all(current.get(k) == record.get(k)
                                       for k in ("pid", "start_ticks", "boot_id"))


def resource_snapshot():
    def query(args):
        proc = subprocess.run(["nvidia-smi", *args], capture_output=True, text=True, timeout=15)
        if proc.returncode:
            raise RuntimeError(proc.stderr.strip() or "nvidia-smi failed")
        return proc.stdout

    raw = query(["--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu",
                 "--format=csv,noheader,nounits"])
    gpus = []
    for line in raw.splitlines():
        index, uuid, name, total, used, free, util = [s.strip() for s in line.split(",")]
        gpus.append({"index": int(index), "uuid": uuid, "name": name,
                     "total_mib": int(total), "used_mib": int(used),
                     "free_mib": int(free), "utilization": int(util)})
    processes = query(["--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
                       "--format=csv,noheader,nounits"])
    return {"at": time.time(), "host": socket.gethostname(), "gpus": gpus,
            "gpu_processes_csv": processes, "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "process_visibility": "reported_entries_not_guaranteed_complete" if processes.strip() else "empty_or_unavailable_not_evidence_of_idle_gpu",
            "load_average": list(os.getloadavg()), "disk_free_bytes": shutil.disk_usage(ROOT).free,
            "branch_logical": str(ROOT), "branch_resolved": str(ROOT.resolve())}


def completion(task):
    path = local_path(task["complete"])
    try:
        value = read_json(path)
        if not isinstance(value, dict) or not value:
            return None
        if value.get("status", "complete") not in ("complete", "completed", "done", "PASS"):
            return None
        if value.get("complete") is False:
            return None
        for key, expected in task.get("completion_checks", {}).items():
            if value.get(key) != expected:
                return None
        for key, expected in task.get("binding_checks", {}).items():
            if value.get("binding", {}).get(key) != expected:
                return None
        if task.get("verify_artifacts"):
            files = value.get("artifacts", value.get("artifacts_sha256", {}))
            if not isinstance(files, dict) or not set(task.get("required_artifacts", ["latest.pt", "best.pt", "results.npz", "run.json"])) <= files.keys():
                return None
            for relative, digest in files.items():
                artifact = (path.parent / relative).resolve()
                artifact.relative_to(path.parent.resolve())
                if sha256(artifact) != digest:
                    return None
            if value.get("binding", {}).get("protocol_sha256") != sha256(ROOT / "protocol.json"):
                return None
            for field in ("code_sha256", "dependencies"):
                code_hashes = value.get("binding", {}).get(field, {})
                if not isinstance(code_hashes, dict) or not code_hashes:
                    return None
                for source, digest in code_hashes.items():
                    if sha256(source) != digest:
                        return None
            if not str(value.get("identity", {}).get("device", "")).startswith("cuda"):
                return None
        return {"path": str(path), "sha256": sha256(path), "record": value}
    except (ValueError, OSError, TypeError, AttributeError):
        return None


def validate_manifest(value):
    tasks = value["tasks"]
    ids = {task["id"] for task in tasks}
    if len(ids) != len(tasks):
        raise ValueError("duplicate task ids")
    for task in tasks:
        if not task["id"] or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in task["id"]):
            raise ValueError("unsafe task id")
        for key in ("complete", "checkpoint"):
            if task.get(key):
                local_path(task[key])
        if not isinstance(task["args"], list) or not all(isinstance(s, str) for s in task["args"]):
            raise ValueError("task args must be a string array")
        if not set(task.get("depends_on", [])) <= ids or task["id"] in task.get("depends_on", []):
            raise ValueError("unknown or self dependency")
    remaining, visited = {t["id"]: set(t.get("depends_on", [])) for t in tasks}, set()
    while remaining:
        ready = [k for k, deps in remaining.items() if deps <= visited]
        if not ready:
            raise ValueError("dependency cycle")
        for key in ready:
            visited.add(key)
            del remaining[key]
    return tasks


def source_hashes(manifest_path, manifest):
    paths = {manifest_path.resolve(), *ROOT.glob("*.py"), *ROOT.glob("*.sh")}
    paths.update(local_path(p) for p in manifest.get("source_paths", ["protocol.json"]))
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(paths) if p.is_file()}


def preflight_gate(manifest):
    gate = manifest.get("cpu_diagnostic_gate")
    if not gate:
        return None
    try:
        report = read_json(local_path(gate["report"]))
        if not report or report.get("status") != "PASS":
            raise ValueError("CPU/GPU preflight has not passed")
        for field in ("unit_tests_passed", "cpu_resume_exact", "gpu_resume_exact", "gpu_batch64_pilot_passed", "all_oracle_calls_zero"):
            if report.get(field) is not True:
                raise ValueError("Preflight missing " + field)
        if report["protocol_sha256"] != sha256(ROOT / "protocol.json"):
            raise ValueError("Preflight protocol stale")
        for path, digest in report["source_sha256"].items():
            if sha256(local_path(path)) != digest:
                raise ValueError("Preflight source stale: " + path)
        for path, digest in report["evidence_sha256"].items():
            if sha256(local_path(path)) != digest:
                raise ValueError("Preflight evidence changed: " + path)
        return None
    except (ValueError, OSError, KeyError, TypeError, AttributeError) as exc:
        return str(exc)


def admissible_gpu(history, task, args):
    return (len(history) >= args.admission_samples and
            all(g["utilization"] <= args.max_gpu_util for g in history) and
            min(g["free_mib"] for g in history) >=
            max(args.min_free_mib, task.get("required_free_mib", 6144) + args.memory_margin_mib))


def stop_handler(_signum, _frame):
    global STOP
    STOP = True


def worker(spec_path):
    """A detached wrapper owns its GPU lease even if the scheduler exits."""
    spec = read_json(spec_path)
    spec["worker_identity"] = identity(os.getpid())
    atomic(spec_path, spec)
    result_path = local_path(spec["result_path"])
    lease = local_path(f"artifacts/operations/gpu_{spec['gpu']}.lock").open("a+")
    try:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        atomic(result_path, {"status": "deferred", "reason": "gpu_lease_busy", "at": time.time()})
        return 75
    meta = {"task_id": spec["task_id"], "gpu": spec["gpu"], **identity(os.getpid())}
    lease.seek(0)
    lease.truncate()
    json.dump(meta, lease)
    lease.flush()
    start = time.time()
    try:
        snapshot = resource_snapshot()
        gpu = next(g for g in snapshot["gpus"] if g["index"] == spec["gpu"])
        if gpu["free_mib"] < spec["admission_free_mib"] or gpu["utilization"] > spec["max_gpu_util"]:
            atomic(result_path, {"status": "deferred", "reason": "immediate_admission_changed",
                                 "at": time.time(), "snapshot": snapshot})
            return 75
        if spec.get("cpu_affinity"):
            os.sched_setaffinity(0, spec["cpu_affinity"])
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(spec["gpu"]),
                   RL_ONLY_CPU_THREADS=str(len(spec["cpu_affinity"])))
        child = subprocess.Popen(spec["command"], cwd=ROOT, env=env)
        meta.update({"child_identity": identity(child.pid), "started_at": start,
                     "command": spec["command"], "cpu_affinity": spec["cpu_affinity"]})
        atomic(local_path(spec["heartbeat_path"]), {**meta, "at": time.time()})
        while child.poll() is None:
            atomic(local_path(spec["heartbeat_path"]), {**meta, "at": time.time()})
            time.sleep(10)
        record = {"status": "exited", "returncode": child.returncode, "started_at": start,
                  "finished_at": time.time(), "elapsed_seconds": time.time() - start, **meta}
        atomic(result_path, record)
        return child.returncode
    except BaseException:
        atomic(result_path, {"status": "error", "returncode": 1, "at": time.time(),
                             "error": traceback.format_exc(), **meta})
        raise
    finally:
        fcntl.flock(lease, fcntl.LOCK_UN)
        lease.close()


def estimate(tasks, finished_records, pending, running):
    measured = []
    task_map = {t["id"]: t for t in tasks}
    for key, record in finished_records.items():
        if task_map[key].get("kind") == "training" and record.get("elapsed_seconds", 0) > 0:
            measured.append(record["elapsed_seconds"])
    active = []
    for key in running:
        task = task_map[key]
        heartbeat = read_json(local_path(task["complete"]).parent / "heartbeat.json", {})
        if heartbeat.get("seconds_per_update") and heartbeat.get("updates_completed", 0) >= 2:
            active.append({"task_id": key, "kind": task.get("kind"), "heartbeat_at": heartbeat.get("at", heartbeat.get("updated")),
                           "updates_completed": heartbeat["updates_completed"],
                           "updates_target": heartbeat.get("updates_target"),
                           "seconds_per_update": heartbeat["seconds_per_update"],
                           "estimated_remaining_seconds": heartbeat.get("estimated_remaining_seconds")})
    return {"capacity_wait_seconds": None, "capacity_wait_explanation": "unknown; depends on other users releasing GPU compute",
            "measured_completed_training_seconds": measured,
            "running_measured_progress": active,
            "single_gpu_active_seconds_remaining": (sum(measured) / len(measured) *
                 sum(task_map[k].get("kind") == "training" for k in set(pending) | set(running))) if measured else None,
            "note": "Per-run estimates use measured CUDA update time; queue estimate awaits completed training. Evaluation, queue waiting and sharing changes add time."}


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--manifest", default="queue.json")
    parser.add_argument("--poll-seconds", type=float, default=20)
    parser.add_argument("--admission-samples", type=int, default=3)
    parser.add_argument("--max-gpu-util", type=int, default=50)
    parser.add_argument("--min-free-mib", type=int, default=8192)
    parser.add_argument("--memory-margin-mib", type=int, default=2048)
    parser.add_argument("--min-disk-gib", type=float, default=20)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker-spec")
    args = parser.parse_args(argv)
    if args.worker_spec:
        return worker(args.worker_spec)
    if not (1 <= args.admission_samples and 0 <= args.max_gpu_util <= 80 and
            args.memory_margin_mib >= 1024 and args.poll_seconds >= 1 and args.max_workers >= 1 and args.max_retries >= 0):
        parser.error("invalid admission parameters; utilization cap <=80 and memory margin >=1024 MiB are required")
    OPS.mkdir(parents=True, exist_ok=True)
    lock = (OPS / "queue.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error("another scheduler owns queue.lock; attach to that scheduler")
    def finish(code):
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
        return code
    manifest_path = local_path(args.manifest)
    manifest = read_json(manifest_path)
    tasks = validate_manifest(manifest)
    task_map = {task["id"]: task for task in tasks}
    gate_error = preflight_gate(manifest)
    if gate_error and not args.dry_run:
        finish(1)
        raise RuntimeError(f"CPU preflight gate not satisfied: {gate_error}")
    hashes = source_hashes(manifest_path, manifest)
    frozen = read_json(OPS / "source_hashes.json")
    if frozen and frozen["hashes"] != hashes and not args.dry_run:
        raise RuntimeError("source/config hashes changed since launch; inspect frozen protocol before resuming")
    launch_id = f"{time.time_ns()}_{os.getpid()}"
    scheduler_identity = identity(os.getpid())
    launch = {"at": time.time(), "launch_id": launch_id, "argv": sys.argv,
              "identity": scheduler_identity, "hashes": hashes, "options": vars(args)}
    if not args.dry_run:
        atomic(OPS / "source_hashes.json", {"hashes": hashes, "frozen_at": (frozen or {}).get("frozen_at", time.time())})
        atomic(OPS / f"launch_{launch_id}.json", launch)
        atomic(OPS / "launch.json", launch)
    prior = read_json(OPS / "queue.json", {})
    attempts = prior.get("attempts", {})
    running = {row["task_id"]: row for row in prior.get("running", []) if row["task_id"] in task_map}
    # Recover the tiny launch-to-state-publication window via durable attempt records.
    for path in sorted(OPS.glob("attempt_*.json")):
        row = read_json(path, {})
        key = row.get("task_id")
        if key in task_map and row.get("worker_identity"):
            if is_alive(row["worker_identity"]) or row["attempt"] > attempts.get(key, 0):
                if row.get("started_at", 0) >= running.get(key, {}).get("started_at", 0):
                    running[key] = row
                    attempts[key] = max(attempts.get(key, 0), row["attempt"])
    finished_records = prior.get("finished_records", {})
    completed = {task["id"]: record for task in tasks if (record := completion(task))}
    failures = prior.get("failures", {})
    for key in completed:
        failures.pop(key, None)
    histories = {}
    cpu_pool = sorted(os.sched_getaffinity(0))
    signal.signal(signal.SIGTERM, stop_handler)
    signal.signal(signal.SIGINT, stop_handler)
    while True:
        # Reap exited wrappers launched by this scheduler, without touching other processes.
        while True:
            try:
                reaped, _ = os.waitpid(-1, os.WNOHANG)
                if not reaped:
                    break
            except ChildProcessError:
                break
        for key, row in list(running.items()):
            result = read_json(local_path(row["result_path"]))
            alive = is_alive(row["worker_identity"])
            heartbeat = read_json(local_path(row["heartbeat_path"]), {})
            child_identity = heartbeat.get("child_identity")
            if not alive and child_identity and is_alive(child_identity):
                row["recovery_state"] = "wrapper_gone_child_still_running"
                # Preserve its GPU reservation and never launch a duplicate child.
                continue
            if result is None and alive:
                continue
            if alive:  # Wrapper may be flushing a result; retain GPU lease until it exits.
                continue
            result = result or {"status": "lost_worker", "returncode": None}
            done = completion(task_map[key])
            if done and (result.get("returncode") == 0 or result.get("status") == "lost_worker"):
                if result.get("status") == "lost_worker":
                    result = {**result, "status": "completion_verified_exit_unobserved",
                              "note": "Both recorded process identities exited; trainer completion artifact validated."}
                completed[key] = done
                finished_records[key] = result
                failures.pop(key, None)
            elif result.get("status") == "deferred" or result.get("returncode") == 75:
                attempts[key] = max(0, attempts.get(key, 1) - 1)
                histories.pop(row["gpu"], None)
            else:
                failures[key] = {"attempt": attempts.get(key, 1), "at": time.time(), "result": result,
                                 "retry_exhausted": attempts.get(key, 1) > args.max_retries}
            del running[key]
        pending = [task["id"] for task in tasks if task["id"] not in completed and task["id"] not in running
                   and not failures.get(task["id"], {}).get("retry_exhausted")]
        exhausted = {key for key, item in failures.items() if item.get("retry_exhausted")}
        blocked = {key for key in pending if set(task_map[key].get("depends_on", [])) & exhausted}
        snapshot_error = None
        source_error = None
        gate_error = preflight_gate(manifest)
        if not args.dry_run and source_hashes(manifest_path, manifest) != hashes:
            source_error = "source/config changed after launch; admission paused"
        try:
            snapshot = resource_snapshot()
            atomic(OPS / "resource_latest.json", snapshot)
            with (OPS / "resource_history.jsonl").open("a") as stream:
                stream.write(json.dumps(snapshot, sort_keys=True) + "\n")
            for gpu in snapshot["gpus"]:
                histories.setdefault(gpu["index"], deque(maxlen=args.admission_samples)).append(gpu)
        except Exception as exc:
            snapshot, snapshot_error = None, repr(exc)
            histories.clear()
        disk_ok = snapshot is not None and snapshot["disk_free_bytes"] >= args.min_disk_gib * 2**30
        eligible_resources = []
        if snapshot and disk_ok and not STOP and not source_error and not gate_error:
            used_gpus = {row["gpu"] for row in running.values()}
            used_cpus = {cpu for row in running.values() for cpu in row.get("cpu_affinity", [])}
            for gpu in sorted(snapshot["gpus"], key=lambda g: (-g["free_mib"], g["index"])):
                if len(running) >= args.max_workers or gpu["index"] in used_gpus:
                    continue
                history = histories[gpu["index"]]
                if len(history) < args.admission_samples or any(g["utilization"] > args.max_gpu_util for g in history):
                    continue
                choices = [task_map[k] for k in pending if k not in blocked and
                           set(task_map[k].get("depends_on", [])) <= completed.keys()]
                task = next((t for t in choices if admissible_gpu(history, t, args)), None)
                if task is None:
                    continue
                cpus = [c for c in cpu_pool if c not in used_cpus][:task.get("cpu_threads", 3)]
                if len(cpus) < task.get("cpu_threads", 3):
                    continue
                eligible_resources.append({"gpu": gpu["index"], "task_id": task["id"], "cpu_affinity": cpus})
                if args.dry_run:
                    continue
                key = task["id"]
                attempt = attempts.get(key, 0) + 1
                token = f"{key}_a{attempt}_{time.time_ns()}"
                command = ["bash", str(ROOT / "run_env.sh"), *task["args"], "--physical-gpu-index", str(gpu["index"])]
                command.extend(task.get("resume_args", []))
                row = {"task_id": key, "gpu": gpu["index"], "gpu_uuid": gpu["uuid"], "attempt": attempt,
                       "cpu_affinity": cpus, "command": command, "started_at": time.time(),
                       "result_path": f"artifacts/operations/result_{token}.json",
                       "heartbeat_path": f"artifacts/operations/heartbeat_{token}.json",
                       "admission_free_mib": max(args.min_free_mib, task.get("required_free_mib", 6144) + args.memory_margin_mib),
                       "max_gpu_util": args.max_gpu_util, "source_hashes": hashes,
                       "checkpoint_present": bool(task.get("checkpoint") and local_path(task["checkpoint"]).exists()),
                       "admission_history": list(history)}
                spec_path = OPS / f"attempt_{token}.json"
                atomic(spec_path, row)
                atomic(OPS / f"preflight_{token}.json", snapshot)
                with (OPS / f"{key}.log").open("a") as log:
                    process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker-spec", str(spec_path)],
                                               cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                row["worker_identity"] = identity(process.pid)
                if row["worker_identity"] is None:
                    row["worker_identity"] = {"pid": process.pid, "start_ticks": "gone", "boot_id": "gone"}
                atomic(spec_path, row)
                attempts[key] = attempt
                running[key] = row
                pending.remove(key)
                used_gpus.add(gpu["index"])
                used_cpus.update(cpus)
        status = "running" if running else "waiting_for_resources"
        if snapshot_error:
            status = "waiting_for_gpu_telemetry"
        elif not disk_ok:
            status = "waiting_for_disk_capacity"
        elif not pending and not running:
            status = "failed" if exhausted else "tasks_complete"
        elif pending and not running and all(key in blocked for key in pending):
            status = "failed_dependencies"
        if STOP:
            status = "scheduler_stopped_workers_preserved"
        elif source_error:
            status = "blocked_source_changes"
        elif gate_error:
            status = "blocked_cpu_preflight"
        state = {"at": time.time(), "status": status, "launch_id": launch_id,
                 "scheduler_identity": scheduler_identity, "pending": pending,
                 "running": list(running.values()), "completed": completed,
                 "failures": failures, "attempts": attempts, "blocked": sorted(blocked),
                 "finished_records": finished_records, "snapshot_error": snapshot_error,
                 "source_error": source_error,
                 "preflight_gate_error": gate_error,
                 "eligible_resources": eligible_resources,
                 "eta": estimate(tasks, finished_records, pending, running)}
        atomic(OPS / ("dry_run.json" if args.dry_run else "queue.json"), state)
        if args.dry_run or args.once or STOP:
            return finish(0)
        if status == "tasks_complete":
            post = manifest.get("postprocess")
            if post:
                with (OPS / "postprocess.log").open("a") as log:
                    result = subprocess.run(["bash", str(ROOT / "run_env.sh"), *post["args"]], cwd=ROOT,
                                            stdout=log, stderr=subprocess.STDOUT)
                if result.returncode or not completion(post):
                    atomic(OPS / "exit.json", {"at": time.time(), "launch_id": launch_id,
                                             "status": "postprocess_failed", "returncode": result.returncode})
                    return finish(1)
            atomic(OPS / "exit.json", {"at": time.time(), "launch_id": launch_id, "status": "complete",
                                      "completed": sorted(completed), "running": [], "pending": []})
            return finish(0)
        if status in ("failed", "failed_dependencies"):
            atomic(OPS / "exit.json", {"at": time.time(), "launch_id": launch_id, "status": status,
                                      "failures": failures, "blocked": sorted(blocked)})
            return finish(1)
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
