"""CPU-only final-acceptance watcher. It never starts or repairs training."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

SCRIPT = Path(__file__).resolve()
OUT = SCRIPT.parent
ROOT = OUT.parents[1]
STOP_REQUESTED = False


class GateError(RuntimeError):
    pass


def read_json(path, required=False):
    try:
        value = json.loads(Path(path).read_text())
    except FileNotFoundError:
        if required:
            raise GateError(f"required artifact missing: {path}")
        return None
    if not isinstance(value, dict):
        raise GateError(f"expected JSON object: {path}")
    return value


def atomic(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.partial")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def process_identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        return {"pid": int(pid), "start_ticks": fields[19],
                "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}
    except (FileNotFoundError, ProcessLookupError):
        return None


def is_alive(identity):
    return bool(identity) and process_identity(identity["pid"]) == identity


def lock_released(path):
    if not Path(path).is_file():
        raise GateError(f"expected scheduler lock file missing: {path}")
    with Path(path).open("rb") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        fcntl.flock(stream, fcntl.LOCK_UN)
        return True


def assess_main(root, launch_id, scheduler_identity):
    """Return waiting/ready; raise for failure, stale identity, or wrong reports."""
    root = Path(root)
    ops = root / "artifacts/operations"
    queue = read_json(ops / "queue.json", required=True)
    launch = read_json(ops / "launch.json", required=True)
    if launch.get("launch_id") != launch_id or queue.get("launch_id") != launch_id:
        raise GateError("main scheduler launch changed; this watcher does not follow restarts")
    if launch.get("identity") != scheduler_identity or queue.get("scheduler_identity") != scheduler_identity:
        raise GateError("main scheduler recorded process identity changed")
    status = queue.get("status")
    if status not in {"running", "waiting_for_resources", "waiting_for_gpu_telemetry",
                      "waiting_for_disk_capacity", "tasks_complete"}:
        raise GateError(f"main queue is not in a normal state: {status}")
    for field in ("source_error", "preflight_gate_error"):
        if queue.get(field):
            raise GateError(f"main queue {field}: {queue[field]}")
    if any(item.get("retry_exhausted") for item in queue.get("failures", {}).values()):
        raise GateError("main queue has exhausted task retries")
    exit_record = read_json(ops / "exit.json")
    report = read_json(root / "artifacts/report/complete.json")
    alive = is_alive(scheduler_identity)
    # A process may exit between the first marker read and this liveness check.
    if not alive:
        exit_record = read_json(ops / "exit.json")
        report = read_json(root / "artifacts/report/complete.json")
    if exit_record is not None:
        if exit_record.get("launch_id") != launch_id:
            raise GateError("main exit record belongs to a different launch")
        if exit_record.get("status") != "complete":
            raise GateError(f"main scheduler exit reports failure: {exit_record.get('status')}")
        if exit_record.get("running") or exit_record.get("pending"):
            raise GateError("main complete exit still lists unfinished tasks")
    if report is not None:
        if report.get("status") != "complete" or report.get("acceptance") != "accepted_gpu":
            raise GateError("formal report exists without complete accepted_gpu status")
        if report.get("protocol_sha256") != sha256(root / "protocol.json"):
            raise GateError("formal report protocol hash mismatch")
    detail = {"queue_status": status, "completed": len(queue.get("completed", {})),
              "running": len(queue.get("running", [])), "pending": len(queue.get("pending", [])),
              "failures": queue.get("failures", {}), "queue_updated_at": queue.get("at"),
              "scheduler_alive": alive, "exit_complete": exit_record is not None,
              "report_accepted_gpu": report is not None}
    if exit_record is not None and report is not None:
        task_ids = {task["id"] for task in read_json(root / "queue.json", required=True)["tasks"]}
        if set(queue.get("completed", {})) != task_ids or queue.get("running") or queue.get("pending"):
            raise GateError("completion markers conflict with main task queue")
        if alive or not lock_released(ops / "queue.lock"):
            return {"state": "waiting_for_scheduler_exit", **detail}
        return {"state": "ready", **detail}
    if not alive:
        raise GateError("main scheduler exited without both successful exit and accepted_gpu report; no automatic restart")
    if status != "tasks_complete" and time.time() - float(queue.get("at", 0)) > 180:
        raise GateError("main scheduler heartbeat older than 180 seconds")
    return {"state": "waiting", **detail}


def build_binding(root, out):
    root, out = Path(root), Path(out)
    frozen_path = root / "artifacts/operations/source_hashes.json"
    frozen = read_json(frozen_path, required=True)
    launch = read_json(root / "artifacts/operations/launch.json", required=True)
    paths = {root / name for name in frozen["hashes"]}
    paths.update({SCRIPT, out / "audit_acceptance.py", out / "statistics_review.py", frozen_path,
                  root / "protocol.json", root / "run_env.sh"})
    for path in paths:
        path.resolve().relative_to(root.resolve())
        if not path.is_file():
            raise GateError(f"binding input missing: {path}")
    for name, digest in frozen["hashes"].items():
        if sha256(root / name) != digest:
            raise GateError(f"main frozen source/config drift: {name}")
    return {"main_launch_id": launch["launch_id"], "main_scheduler_identity": launch["identity"],
            "files": {str(path.relative_to(root)): sha256(path) for path in sorted(paths)},
            "protocol_sha256": sha256(root / "protocol.json"),
            "audit_command": ["bash", str(root / "run_env.sh"), str(out / "audit_acceptance.py"), "--full"]}


def verify_binding(root, binding):
    for name, expected in binding["files"].items():
        path = Path(root) / name
        if not path.is_file() or sha256(path) != expected:
            raise GateError(f"watcher-bound source/config changed: {name}")


def require_final_pass(final):
    if final.get("status") != "PASS":
        raise GateError(f"acceptance_final.json does not indicate PASS: {final.get('status')}")


def stop_handler(_signum, _frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--poll-seconds", type=float, default=20)
    parser.add_argument("--check-only", action="store_true", help="readiness check; no lock, loop, subprocess or output files")
    args = parser.parse_args(argv)
    if not 1 <= args.poll_seconds <= 30:
        parser.error("poll-seconds must be between 1 and 30")
    if args.check_only:
        binding = build_binding(ROOT, OUT)
        print(json.dumps(assess_main(ROOT, binding["main_launch_id"], binding["main_scheduler_identity"]), indent=2))
        return 0
    lock = (OUT / "watcher.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        print("Another watcher or inherited acceptance-audit process holds watcher.lock", file=sys.stderr)
        return 2
    identity = process_identity(os.getpid())
    started = time.time()
    watcher_id = f"{time.time_ns()}_{os.getpid()}"
    base = {"watcher_id": watcher_id, "identity": identity, "started_at": started,
            "main_branch": str(ROOT), "cpu_only": True, "automatic_training_restart": False}
    lock.seek(0)
    lock.truncate()
    json.dump(base, lock)
    lock.flush()
    signal.signal(signal.SIGTERM, stop_handler)
    signal.signal(signal.SIGINT, stop_handler)
    try:
        prior = read_json(OUT / "watcher_status.json")
        if prior and is_alive(prior.get("audit_identity")):
            raise GateError("prior acceptance audit still live; duplicate CPU audit refused")
        binding = build_binding(ROOT, OUT)
        atomic(OUT / "watcher_binding.json", {**base, "binding": binding})
        atomic(OUT / "watcher_launch.json", {**base, "argv": sys.argv, "poll_seconds": args.poll_seconds})
        while True:
            verify_binding(ROOT, binding)
            state = assess_main(ROOT, binding["main_launch_id"], binding["main_scheduler_identity"])
            atomic(OUT / "watcher_status.json", {**base, "at": time.time(), "status": "waiting", "main": state})
            if STOP_REQUESTED:
                atomic(OUT / "watcher_exit.json", {**base, "at": time.time(), "status": "stopped_before_audit"})
                return 130
            if state["state"] == "ready":
                break
            time.sleep(args.poll_seconds)
        verify_binding(ROOT, binding)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
        with (OUT / "watcher_audit.log").open("a") as log:
            # The child inherits this lease, preventing duplicate audits even if
            # the watcher is forcibly killed. No GPU is visible to the audit.
            child = subprocess.Popen(binding["audit_command"], cwd=ROOT, env=env,
                                     stdout=log, stderr=subprocess.STDOUT, pass_fds=(lock.fileno(),))
            audit_identity = process_identity(child.pid)
            while child.poll() is None:
                atomic(OUT / "watcher_status.json", {**base, "at": time.time(), "status": "auditing",
                       "audit_identity": audit_identity, "main": state,
                       "stop_requested": STOP_REQUESTED})
                time.sleep(min(args.poll_seconds, 10))
        verify_binding(ROOT, binding)
        if child.returncode != 0:
            raise GateError(f"audit_acceptance.py --full exited {child.returncode}; see watcher_audit.log")
        final_path = OUT / "acceptance_final.json"
        final = read_json(final_path, required=True)
        require_final_pass(final)
        terminal = {**base, "at": time.time(), "status": "complete", "audit_returncode": child.returncode,
                    "acceptance_final": str(final_path), "acceptance_final_sha256": sha256(final_path),
                    "binding_sha256": sha256(OUT / "watcher_binding.json")}
        atomic(OUT / "watcher_status.json", terminal)
        atomic(OUT / "watcher_exit.json", terminal)
        return 0
    except Exception as exc:
        failure = {**base, "at": time.time(), "status": "failed", "error": str(exc),
                   "traceback": traceback.format_exc()}
        atomic(OUT / "watcher_status.json", failure)
        atomic(OUT / "watcher_exit.json", failure)
        print(f"Acceptance watcher failed: {exc}", file=sys.stderr)
        return 1
    finally:
        # Close without LOCK_UN: an audit child may still inherit this lease.
        lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
