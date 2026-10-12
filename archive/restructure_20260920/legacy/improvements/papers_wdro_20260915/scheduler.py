"""Persistent queue for the basic-WDRO paper experiments; Python standard library.

Commands are argv arrays (never shell strings). Each task owns task/GPU flock
leases inherited by its child. Restart checks Linux process identity and kernel
locks before replay. Only this queue's own child process group may be signalled.
"""
from __future__ import annotations

import argparse
from collections import deque
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
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
    temp = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.partial")
    with temp.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def local(value):
    path = Path(value)
    path = path if path.is_absolute() else ROOT / path
    path.resolve().relative_to(ROOT.resolve())
    return path


def identity(pid):
    try:
        if int(pid) <= 0:
            return None
        fields = Path(f"/proc/{int(pid)}/stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] in ("Z", "X", "x"):
            return None
        return dict(pid=int(pid), start_ticks=fields[19],
                    boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip())
    except (FileNotFoundError, ProcessLookupError):
        return None


def alive(record):
    return bool(record and identity(record["pid"]) == record)


def acquire(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a+")
    try:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        stream.close()
        return None
    return stream


def held(path):
    stream = acquire(path)
    if stream is None:
        return True
    stream.close()
    return False


def effective_cpu_count():
    count = len(os.sched_getaffinity(0))
    # Resolve both unified and legacy quota limits. Affinity alone can be 192
    # while the container actually owns only about 31 CPU cores.
    for path in (Path("/sys/fs/cgroup/cpu.max"),):
        try:
            quota, period = path.read_text().split()
            if quota != "max":
                count = min(count, max(1, int(int(quota) / int(period))))
        except (OSError, ValueError):
            pass
    try:
        quota = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
        period = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
        if quota > 0:
            count = min(count, max(1, quota // period))
    except (OSError, ValueError):
        pass
    return count


def resource_snapshot():
    query = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.free,utilization.gpu",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15, check=True)
    gpus = []
    for line in query.stdout.splitlines():
        fields = [v.strip() for v in line.split(",")]
        try:
            index, uuid, name, total, free, util = fields
            gpus.append(dict(index=int(index), uuid=uuid, name=name, total_mib=int(total),
                             free_mib=int(free), utilization=int(util)))
        except ValueError:
            continue  # Unknown telemetry cannot authorize admission.
    try:
        processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
                                             "--format=csv,noheader,nounits"], text=True, timeout=15)
    except (OSError, subprocess.SubprocessError) as error:
        processes = f"unavailable: {error}"
    return dict(at=time.time(), host=socket.gethostname(), gpus=gpus, gpu_processes_csv=processes,
                process_visibility="may_be_incomplete_not_used_to_infer_idle", load=list(os.getloadavg()),
                cpu_affinity=sorted(os.sched_getaffinity(0)), effective_cpus=effective_cpu_count(),
                disk_free_bytes=shutil.disk_usage(ROOT).free, memory_info=Path("/proc/meminfo").read_text(),
                branch_logical=str(ROOT), branch_resolved=str(ROOT.resolve()))


def admissible(history, task, args):
    if len(history) < args.admission_samples:
        return False
    util = [g["utilization"] for g in history]
    return (sum(util) / len(util) <= args.max_mean_util and min(util) <= args.required_low_util
            and max(util) <= args.max_sample_util
            and min(g["free_mib"] for g in history) >= task.get("required_free_mib", 4096) + args.memory_margin_mib)


def effective_requirement(task, task_map, completed):
    """Bind formal admission to verified same-batch GPU smoke peak reservations.

Keep the frozen queue unchanged. Recheck metrics bytes against the already
accepted completion record before using its measured reservation as evidence.
"""
    result = dict(task)
    required = max(4096, task.get("required_free_mib", 4096))
    evidence = []
    for dependency in task.get("depends_on", []):
        if not dependency.startswith("gpu_smoke_"):
            continue
        accepted = completed.get(dependency)
        if not accepted:
            raise ValueError(f"GPU smoke dependency has not passed acceptance: {dependency}")
        complete_path = local(task_map[dependency]["complete"])
        metrics_path = complete_path.parent / "metrics.json"
        expected = accepted["record"].get("artifacts", {}).get("metrics.json")
        if not expected or digest(metrics_path) != expected:
            raise ValueError(f"GPU smoke metrics changed or are not hash-bound: {dependency}")
        metrics = read(metrics_path, {})
        peak = metrics.get("cuda_max_reserved_mib")
        if isinstance(peak, bool) or not isinstance(peak, (int, float)) or not math.isfinite(peak) or peak <= 0:
            raise ValueError(f"GPU smoke must report positive finite cuda_max_reserved_mib: {dependency}")
        measured_required = math.ceil(peak + 1024)
        required = max(required, measured_required)
        evidence.append(dict(task_id=dependency, metrics_path=str(metrics_path), metrics_sha256=expected,
                             cuda_max_reserved_mib=peak, measured_requirement_mib=measured_required))
    result["required_free_mib"] = required
    result["resource_requirement_evidence"] = evidence
    return result


def validate_manifest(manifest):
    tasks = manifest["tasks"]
    ids = {task["id"] for task in tasks}
    if len(ids) != len(tasks) or not tasks:
        raise ValueError("tasks must have nonempty unique ids")
    for task in tasks:
        if re.fullmatch(r"[A-Za-z0-9_-]+", task["id"]) is None:
            raise ValueError("unsafe task id")
        if not isinstance(task["command"], list) or not task["command"] or not all(isinstance(s, str) for s in task["command"]):
            raise ValueError("command must be a nonempty string array")
        for field in ("complete", "checkpoint", "progress"):
            if task.get(field):
                local(task[field])
        if "complete" not in task or task.get("cpu_threads", 2) < 1:
            raise ValueError("task requires completion path and positive cpu_threads")
        if not set(task.get("depends_on", [])) <= ids:
            raise ValueError("unknown task dependency")
    todo = {t["id"]: set(t.get("depends_on", [])) for t in tasks}
    seen = set()
    while todo:
        ready = [key for key, deps in todo.items() if deps <= seen]
        if not ready:
            raise ValueError("dependency cycle")
        for key in ready:
            seen.add(key)
            del todo[key]
    return tasks


def completion(task):
    try:
        path = local(task["complete"])
        value = read(path)
        if not isinstance(value, dict) or value.get("status") not in ("complete", "completed", "PASS"):
            return None
        for key, expected in task.get("completion_checks", {}).items():
            if value.get(key) != expected:
                return None
        for key, expected in task.get("binding_checks", {}).items():
            if value.get("binding", {}).get(key) != expected:
                return None
        artifacts = value.get("artifacts", value.get("artifacts_sha256", {}))
        if task.get("verify_artifacts", True):
            if not isinstance(artifacts, dict) or not artifacts:
                return None
            for relative, expected in artifacts.items():
                artifact = (path.parent / relative).resolve()
                artifact.relative_to(path.parent.resolve())
                if digest(artifact) != expected:
                    return None
        return dict(path=str(path), sha256=digest(path), record=value)
    except (OSError, ValueError, TypeError, KeyError):
        return None


def frozen_sources(manifest_path, manifest):
    paths = {Path(__file__).resolve(), manifest_path.resolve()}
    paths.update(local(p) for p in manifest.get("source_paths", []))
    return {str(p): digest(p) for p in sorted(paths)}


def make_command(task, gpu, cpus):
    command = list(task["command"])
    if task.get("checkpoint") and local(task["checkpoint"]).exists():
        command.extend(task.get("resume_args", []))
    replace = {"{gpu}": str(gpu), "{cpu_threads}": str(len(cpus)), "{root}": str(ROOT)}
    return [substitute(part, replace) for part in command]


def substitute(value, mapping):
    for key, replacement in mapping.items():
        value = value.replace(key, replacement)
    return value


def discover_wrapper(spec_path):
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if entry.stat().st_uid != os.getuid():
                continue
            argv = (entry / "cmdline").read_bytes().decode(errors="replace").split("\0")
            if str(spec_path) in argv and "--worker-spec" in argv:
                current = identity(int(entry.name))
                if current:
                    return current
        except (OSError, ProcessLookupError):
            continue
    return None


def recover_attempts(task_map):
    latest = {}
    for path in sorted(OPS.glob("attempt_*.json")):
        row = read(path, {})
        key = row.get("task_id")
        if key not in task_map:
            continue
        row["spec_path"] = str(path)
        if row.get("started_at", 0) > latest.get(key, {}).get("started_at", -1):
            latest[key] = row
    for row in latest.values():
        if not row.get("worker_identity"):
            row["worker_identity"] = discover_wrapper(Path(row["spec_path"]))
    return latest


def request_stop(_signum, _frame):
    global STOP
    STOP = True


def worker(spec_path):
    spec_path = local(spec_path)
    spec = read(spec_path)
    # The scheduler transfers these open flock descriptions through exec. Child
    # inherits them too, so a killed wrapper cannot release a live child's lease.
    fds = spec["lease_fds"]
    for fd in fds:
        os.fstat(fd)
    spec["worker_identity"] = identity(os.getpid())
    atomic(spec_path, spec)
    child = None
    child_ident = None
    started = time.time()
    stop_reason = None
    termination_at = None
    pressure_count = 0
    result = dict(status="error", returncode=1)
    try:
        fresh = resource_snapshot()
        gpu = next(g for g in fresh["gpus"] if g["index"] == spec["gpu"])
        if gpu["free_mib"] < spec["admission_free_mib"] or gpu["utilization"] > spec["max_sample_util"]:
            result = dict(status="capacity_deferred", returncode=75, reason="immediate_admission_changed", snapshot=fresh)
            return 75
        if any(digest(path) != sha for path, sha in spec["source_hashes"].items()):
            raise RuntimeError("source changed between admission and worker launch")
        os.sched_setaffinity(0, spec["cpu_affinity"])
        os.nice(10)
        threads = str(len(spec["cpu_affinity"]))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(spec["gpu"]), PAPERS_WDRO_CPU_THREADS=threads,
                   OMP_NUM_THREADS=threads, MKL_NUM_THREADS=threads, OPENBLAS_NUM_THREADS=threads,
                   NUMEXPR_NUM_THREADS=threads, PYTHONUNBUFFERED="1")
        child = subprocess.Popen(spec["command"], cwd=ROOT, env=env, pass_fds=tuple(fds), start_new_session=True)
        child_ident = identity(child.pid)
        spec["child_identity"] = child_ident
        atomic(spec_path, spec)
        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)
        while child.poll() is None:
            now = time.time()
            progress_path = local(spec["progress"])
            progress_age = now - (progress_path.stat().st_mtime if progress_path.exists() else started)
            try:
                current = resource_snapshot()
                free = next(g["free_mib"] for g in current["gpus"] if g["index"] == spec["gpu"])
                pressure_count = pressure_count + 1 if free < spec["pressure_free_mib"] else 0
            except (OSError, subprocess.SubprocessError, StopIteration):
                free = None
                pressure_count = 0
            if stop_reason is None:
                if STOP:
                    stop_reason = "worker_stop_requested"
                elif pressure_count >= spec["pressure_samples"]:
                    stop_reason = "gpu_memory_pressure"
                elif progress_age > spec["progress_timeout_seconds"]:
                    stop_reason = "progress_timeout"
                elif now - started > spec["task_timeout_seconds"]:
                    stop_reason = "task_timeout"
                if stop_reason and alive(child_ident):
                    os.killpg(child.pid, signal.SIGTERM)
                    termination_at = now
            elif termination_at and now - termination_at > spec["termination_grace_seconds"] and alive(child_ident):
                os.killpg(child.pid, signal.SIGKILL)
            atomic(local(spec["heartbeat_path"]), dict(at=now, worker_identity=spec["worker_identity"],
                   child_identity=child_ident, task_id=spec["task_id"], gpu=spec["gpu"],
                   progress_age_seconds=progress_age, gpu_free_mib=free, stop_reason=stop_reason))
            time.sleep(spec["worker_poll_seconds"])
        capacity = stop_reason in ("gpu_memory_pressure", "worker_stop_requested") or child.returncode == 75
        result = dict(status="capacity_deferred" if capacity else "exited", returncode=75 if capacity else child.returncode,
                      child_returncode=child.returncode, stop_reason=stop_reason)
        return result["returncode"]
    except BaseException:
        result = dict(status="error", returncode=1, error=traceback.format_exc())
        raise
    finally:
        atomic(local(spec["result_path"]), dict(**result, started_at=started, finished_at=time.time(),
               elapsed_seconds=time.time() - started, worker_identity=spec["worker_identity"], child_identity=child_ident))
        # If child survived an unexpected wrapper exception, its inherited FDs
        # deliberately retain the kernel leases; scheduler recovery waits for it.
        for fd in fds:
            os.close(fd)


def status_for(pending, running, completed, failures, blocked):
    if running:
        return "running"
    if not pending:
        return "failed" if failures else "tasks_complete"
    if set(pending) <= set(blocked):
        return "failed_dependencies"
    return "waiting_for_resources"


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--manifest", default="queue.json")
    parser.add_argument("--poll-seconds", type=float, default=20)
    parser.add_argument("--admission-samples", type=int, default=3)
    parser.add_argument("--max-mean-util", type=float, default=80)
    parser.add_argument("--required-low-util", type=int, default=70)
    parser.add_argument("--max-sample-util", type=int, default=95)
    parser.add_argument("--memory-margin-mib", type=int, default=2048)
    parser.add_argument("--min-disk-gib", type=float, default=10)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--cpu-budget", type=int, default=16)
    parser.add_argument("--max-business-failures", type=int, default=3)
    parser.add_argument("--retry-seconds", type=float, default=120)
    parser.add_argument("--progress-timeout-seconds", type=float, default=1800)
    parser.add_argument("--task-timeout-seconds", type=float, default=86400)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker-spec")
    args = parser.parse_args(argv)
    if args.worker_spec:
        return worker(args.worker_spec)
    if not (args.admission_samples >= 3 and 0 <= args.required_low_util <= args.max_mean_util <= 80
            and args.max_mean_util <= args.max_sample_util <= 95 and args.memory_margin_mib >= 2048
            and args.poll_seconds >= 1 and args.max_workers > 0 and args.cpu_budget > 0
            and args.max_business_failures > 0):
        parser.error("invalid admission or concurrency parameters")
    OPS.mkdir(parents=True, exist_ok=True)
    queue_lock = acquire(OPS / "scheduler.lock")
    if queue_lock is None:
        parser.error("another live scheduler owns scheduler.lock")
    try:
        return run_queue(args)
    finally:
        queue_lock.close()


def run_queue(args):
    manifest_path = local(args.manifest)
    manifest = read(manifest_path)
    tasks = validate_manifest(manifest)
    task_map = {t["id"]: t for t in tasks}
    hashes = frozen_sources(manifest_path, manifest)
    freeze = read(OPS / "source_freeze.json")
    if freeze and freeze["sha256"] != hashes and not args.dry_run:
        raise RuntimeError("source or queue changed since freeze; inspect before resuming")
    if not args.dry_run:
        atomic(OPS / "source_freeze.json", dict(sha256=hashes, at=(freeze or {}).get("at", time.time())))
    prior = read(OPS / "queue_state.json", {})
    latest = recover_attempts(task_map)
    attempts = prior.get("attempts", {})
    failures = prior.get("failures", {})
    retries = prior.get("retry_after", {})
    finished = prior.get("finished", {})
    running = {}
    completed = {}
    for key, row in latest.items():
        attempts[key] = max(attempts.get(key, 0), row["attempt"])
        running[key] = row
    histories = {}
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    while True:
        while True:
            try:
                pid, _ = os.waitpid(-1, os.WNOHANG)
                if not pid:
                    break
            except ChildProcessError:
                break
        for key, row in list(running.items()):
            disk_row = read(row["spec_path"], row)
            row.update(disk_row)
            if alive(row.get("worker_identity")) or alive(row.get("child_identity")) or held(OPS / f"task_{key}.lock"):
                continue
            result = read(local(row["result_path"]), dict(status="lost_worker", returncode=None))
            done = completion(task_map[key])
            previous_result = finished.get(key, {}).get("result_path") == row["result_path"]
            if done and result.get("returncode") == 0:
                completed[key] = done
                failures.pop(key, None)
            elif not previous_result:
                if result.get("returncode") == 75 or result.get("status") == "capacity_deferred":
                    retries[key] = time.time() + args.retry_seconds
                else:
                    failure = failures.setdefault(key, dict(count=0, events=[]))
                    failure["count"] += 1
                    failure["events"].append(dict(at=time.time(), attempt=row["attempt"], result=result))
                    retries[key] = time.time() + args.retry_seconds
            finished[key] = dict(result_path=row["result_path"], **result)
            del running[key]
        # A previous successful result is necessary even if complete.json exists.
        for task in tasks:
            key = task["id"]
            if key not in running and key not in completed and finished.get(key, {}).get("returncode") == 0:
                done = completion(task)
                if done:
                    completed[key] = done
        exhausted = {key for key, value in failures.items() if value["count"] >= args.max_business_failures}
        pending = [t["id"] for t in tasks if t["id"] not in completed and t["id"] not in running and t["id"] not in exhausted]
        blocked = set(exhausted)
        while True:
            new = {t["id"] for t in tasks if set(t.get("depends_on", [])) & blocked} - blocked
            if not new:
                break
            blocked |= new
        error = None
        snap = None
        try:
            snap = resource_snapshot()
            atomic(OPS / "resource_latest.json", snap)
            with (OPS / "resource_history.jsonl").open("a") as stream:
                stream.write(json.dumps(snap, sort_keys=True) + "\n")
            live_indices = {g["index"] for g in snap["gpus"]}
            histories = {key: val for key, val in histories.items() if key in live_indices}
            for gpu in snap["gpus"]:
                histories.setdefault(gpu["index"], deque(maxlen=args.admission_samples)).append(gpu)
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            error = repr(exc)
            histories.clear()
        source_changed = frozen_sources(manifest_path, manifest) != hashes
        disk_ok = snap and snap["disk_free_bytes"] >= args.min_disk_gib * 2**30
        eligible = []
        requirement_errors = {}
        if snap and disk_ok and not source_changed and not STOP:
            reserved_gpus = {row["gpu"] for row in running.values()}
            reserved_cpus = {cpu for row in running.values() for cpu in row["cpu_affinity"]}
            cpu_budget = min(args.cpu_budget, snap["effective_cpus"])
            for gpu in sorted(snap["gpus"], key=lambda g: (g["utilization"], -g["free_mib"], g["index"])):
                if len(running) >= args.max_workers or gpu["index"] in reserved_gpus:
                    continue
                choices = []
                for key in pending:
                    if key in blocked or not set(task_map[key].get("depends_on", [])) <= completed.keys() or retries.get(key, 0) > time.time():
                        continue
                    try:
                        choices.append(effective_requirement(task_map[key], task_map, completed))
                    except (OSError, ValueError, KeyError, TypeError) as exc:
                        requirement_errors[key] = str(exc)
                task = next((t for t in choices if admissible(histories[gpu["index"]], t, args)), None)
                if task is None or len(reserved_cpus) + task.get("cpu_threads", 2) > cpu_budget:
                    continue
                cpus = [c for c in snap["cpu_affinity"] if c not in reserved_cpus][:task.get("cpu_threads", 2)]
                eligible.append(dict(task_id=task["id"], gpu=gpu["index"], cpus=cpus,
                                     required_free_mib=task["required_free_mib"],
                                     admission_free_mib=task["required_free_mib"] + args.memory_margin_mib,
                                     resource_requirement_evidence=task["resource_requirement_evidence"]))
                if args.dry_run:
                    continue
                leases = []
                for path in (OPS / f"task_{task['id']}.lock", OPS / f"gpu_{gpu['index']}.lock"):
                    lease = acquire(path)
                    if lease is None:
                        break
                    leases.append(lease)
                if len(leases) != 2:
                    for lease in leases:
                        lease.close()
                    continue
                key = task["id"]
                attempt = attempts.get(key, 0) + 1
                token = f"{key}_a{attempt}_{time.time_ns()}"
                spec_path = OPS / f"attempt_{token}.json"
                row = dict(task_id=key, attempt=attempt, gpu=gpu["index"], gpu_uuid=gpu["uuid"], cpu_affinity=cpus,
                           command=make_command(task, gpu["index"], cpus), started_at=time.time(), source_hashes=hashes,
                           lease_fds=[lease.fileno() for lease in leases],
                           admission_free_mib=task.get("required_free_mib", 4096) + args.memory_margin_mib,
                           required_free_mib=task["required_free_mib"], resource_requirement_evidence=task["resource_requirement_evidence"],
                           admission_history=list(histories[gpu["index"]]), max_sample_util=args.max_sample_util,
                           pressure_free_mib=task.get("pressure_free_mib", 1024), pressure_samples=2,
                           worker_poll_seconds=max(1, task.get("worker_poll_seconds", 10)), termination_grace_seconds=120,
                           progress=task.get("progress", str(local(task["complete"]).parent / "heartbeat.json")),
                           progress_timeout_seconds=task.get("progress_timeout_seconds", args.progress_timeout_seconds),
                           task_timeout_seconds=task.get("task_timeout_seconds", args.task_timeout_seconds),
                           result_path=f"artifacts/operations/result_{token}.json",
                           heartbeat_path=f"artifacts/operations/heartbeat_{token}.json", spec_path=str(spec_path))
                try:
                    atomic(spec_path, row)
                    atomic(OPS / f"preflight_{token}.json", snap)
                    with (OPS / f"{key}.log").open("a") as log:
                        child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker-spec", str(spec_path)],
                                                 cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                                 pass_fds=tuple(row["lease_fds"]), start_new_session=True)
                    row["worker_identity"] = identity(child.pid)
                    # Separate receipt avoids overwriting the wrapper's newer spec.
                    atomic(OPS / f"launch_{token}.json", row)
                finally:
                    for lease in leases:
                        lease.close()
                attempts[key] = attempt
                running[key] = row
                pending.remove(key)
                reserved_gpus.add(gpu["index"])
                reserved_cpus.update(cpus)
        status = status_for(pending, running, completed, exhausted, blocked)
        if source_changed:
            status = "blocked_source_changes"
        elif STOP:
            status = "scheduler_stopped_workers_preserved"
        elif error and status not in ("tasks_complete", "failed", "failed_dependencies"):
            status = "waiting_for_gpu_telemetry"
        elif not disk_ok and status not in ("tasks_complete", "failed", "failed_dependencies"):
            status = "waiting_for_disk_capacity"
        elif requirement_errors and not running and all(key in requirement_errors or key in blocked for key in pending):
            status = "blocked_gpu_smoke_memory_evidence"
        progress = {}
        for key, row in running.items():
            progress[key] = read(local(row["progress"]), {})
        state = dict(at=time.time(), status=status, scheduler_identity=identity(os.getpid()), pending=pending,
                     running=running, completed=completed, failures=failures, exhausted=sorted(exhausted),
                     blocked=sorted(blocked), attempts=attempts, retry_after=retries, finished=finished,
                     telemetry_error=error, source_changed=source_changed, eligible=eligible, progress=progress,
                     resource_requirement_errors=requirement_errors,
                     eta=dict(capacity_wait_seconds=None, note="GPU queue wait depends on external load; runner heartbeat carries measured progress"))
        atomic(OPS / ("dry_run.json" if args.dry_run else "queue_state.json"), state)
        if args.dry_run or args.once or STOP:
            return 0
        if status == "tasks_complete":
            post = manifest.get("postprocess")
            if post and not completion(post):
                with (OPS / "postprocess.log").open("a") as log:
                    result = subprocess.run(post["command"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                            timeout=post.get("timeout_seconds", 3600))
                if result.returncode or not completion(post):
                    atomic(OPS / "exit.json", dict(at=time.time(), status="postprocess_failed", returncode=result.returncode))
                    return 1
            atomic(OPS / "exit.json", dict(at=time.time(), status="complete", completed=sorted(completed), running=[], pending=[]))
            return 0
        if status in ("failed", "failed_dependencies"):
            atomic(OPS / "exit.json", dict(at=time.time(), status=status, failures=failures, blocked=sorted(blocked)))
            return 1
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
