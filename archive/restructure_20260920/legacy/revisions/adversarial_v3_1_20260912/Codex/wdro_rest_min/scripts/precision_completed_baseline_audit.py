"""Run the unchanged formal auditor on finished correction runs, then hand off.

This CPU-only driver shares the existing early-audit lease with stage3_pipeline.
It never changes training, selection, the auditor, or the final acceptance gate.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODEX = ROOT.parent
EXP = CODEX / "experiments"
ORIGINAL = Path("/remote-home/Guilin/DRO/Codex")
sys.path.insert(0, str(ROOT))

from src.io_utils import append_jsonl, atomic_write_json
from src.job_control import JobLease, utc_now
from src.rest import file_sha256
from run_precision_training_queue import source_gate, process_is_alive
from stage3_pipeline import verify_sources
from precision_replay_pipeline import cpu_preflight, progress

JOB = EXP / "logs/stage3_completed_audit"
QUEUE = EXP / "logs/stage3_queue/stage3_baselines_formal_v1/queue.json"
OPERATIONS = ROOT / "manifests/precision_completed_baseline_audit_operations_v1.json"
SESSION = "precision_completed_baseline_audit"
SOCKET = ORIGINAL / "experiments/tmp/precision_v2.sock"
STOP = False


def read(path):
    return json.loads(Path(path).read_text())


def artifact(path):
    path = Path(path)
    return dict(logical_path=str(path), resolved_path=str(path.resolve()), sha256=file_sha256(path))


def source_check():
    source_gate()
    verify_sources()
    binding = read(OPERATIONS)
    for item in binding["files"]:
        assert file_sha256(Path(item["logical_path"])) == item["sha256"], item["logical_path"]


def process_matches(pid, script):
    if not pid or not process_is_alive(int(pid)):
        return False
    try:
        return str(ROOT / "scripts" / script).encode() in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    except FileNotFoundError:
        return False


def request_signal(pid, script, sig):
    if process_matches(pid, script):
        try:
            os.kill(int(pid), sig)
            return dict(pid=int(pid), script=script, signal=int(sig), at_utc=utc_now())
        except ProcessLookupError:
            pass
    return None


def stop_dependents(reason):
    # Freeze admission before rereading the queue. Only this revision is scoped.
    scheduler_file = QUEUE.parent / "job.json"
    scheduler = read(scheduler_file).get("pid") if scheduler_file.exists() else None
    signaled = []
    item = request_signal(scheduler, "run_precision_training_queue.py", signal.SIGSTOP)
    if item:
        signaled.append(item)
    tasks = read(QUEUE)["tasks"]
    for task in tasks:
        if task["status"] == "running":
            item = request_signal(task.get("pid"), "train_baseline.py", signal.SIGTERM)
            if item:
                signaled.append(item)
    for sig in (signal.SIGTERM, signal.SIGCONT):
        item = request_signal(scheduler, "run_precision_training_queue.py", sig)
        if item:
            signaled.append(item)
    controller_file = EXP / "logs/stage3_pipeline/pid.json"
    if controller_file.exists():
        item = request_signal(read(controller_file)["pid"], "stage3_pipeline.py", signal.SIGTERM)
        if item:
            signaled.append(item)
    report = dict(at_utc=utc_now(), reason=reason, stop_requested=signaled,
                  recovery="Resolve failure.json, then use precision_replay launch.json recovery_command; train from latest.")
    atomic_write_json(JOB / "failure.json", report)
    progress("R3提前完整审计失败，依赖批跑已停止", str(JOB / "failure.json"))


def on_stop(signum, frame):
    global STOP
    STOP = True


def run_report(task):
    return EXP / "logs/stage3_audit/runs" / (task["run_id"] + ".json")


def is_audited(task):
    p = run_report(task)
    result = EXP / "logs/stage3" / task["run_id"] / "result.json"
    if not p.exists() or not result.exists():
        return False
    value = read(p)
    return value["status"] == "pass" and all(value["checks"].values()) and value["result_sha256"] == file_sha256(result)


def terminate_worker(child):
    if child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=5)


def run_one(task, lease, checked):
    source_check()
    preflight = cpu_preflight("precision_completed_audit_" + task["run_id"])
    cursor = EXP / "logs/stage3_audit/cursors" / (task["run_id"] + ".json")
    before = read(cursor)["completed_updates"] if cursor.exists() else 0
    command = [sys.executable, str(Path(__file__)), "--audit-run", task["run_id"]]
    log = JOB / (task["run_id"] + ".log")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", WDRO_HOST_RUNTIME_MANIFEST=str(preflight))
    with log.open("a") as out:
        child = subprocess.Popen(command, stdout=out, stderr=subprocess.STDOUT, env=env, cwd=CODEX)
        record = dict(at_utc=utc_now(), run_id=task["run_id"], pid=child.pid, command=command,
                      exact_command=shlex.join(command), log=str(log), log_resolved=str(log.resolve()),
                      preflight=str(preflight), cursor=str(cursor), cursor_at_start=before,
                      input=artifact(EXP / "logs/stage3" / task["run_id"] / "result.json"))
        atomic_write_json(JOB / "active_input.json", record)
        append_jsonl(JOB / "events.jsonl", dict(event="audit_start", **record))
        try:
            while child.poll() is None:
                handoff = read(QUEUE)["status"] == "complete"
                if STOP or handoff:
                    terminate_worker(child)
                    saved = read(cursor) if cursor.exists() else {}
                    pause = dict(at_utc=utc_now(), status="paused_for_final_audit" if handoff else "paused",
                                 run_id=task["run_id"], worker_pid=child.pid, worker_exit=child.returncode,
                                 cursor=str(cursor), cursor_resolved=str(cursor.resolve()),
                                 cursor_completed_updates=saved.get("completed_updates", 0),
                                 cursor_at_start=before, full_audit_still_required=not is_audited(task))
                    atomic_write_json(JOB / "handoff.json", pause)
                    append_jsonl(JOB / "events.jsonl", dict(event="audit_pause", **pause))
                    return False
                lease.update(update=len(checked), status="running", detail=dict(auditing=task["run_id"],
                             worker_pid=child.pid, cursor=str(cursor), log=str(log)))
                time.sleep(2)
            if child.returncode != 0 or not is_audited(task):
                raise RuntimeError(f"{task['run_id']} audit exit {child.returncode}: {log}")
        finally:
            terminate_worker(child)
    append_jsonl(JOB / "events.jsonl", dict(event="audit_pass", at_utc=utc_now(),
                 run_id=task["run_id"], cursor_at_start=before, report=artifact(run_report(task))))
    return True


def run(args):
    source_check()
    signal.signal(signal.SIGTERM, on_stop)
    signal.signal(signal.SIGINT, on_stop)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.nice(max(0, 19 - os.nice(0)))
    command = ["bash", str(ROOT / "scripts/stage3_env.sh"), sys.executable, str(Path(__file__))]
    recovery = shlex.join([*command, "--launch"])
    tasks = read(QUEUE)["tasks"]
    checked = {t["run_id"] for t in tasks if is_audited(t)}
    with JobLease(JOB, run_id=SESSION, exact_command=shlex.join([*command, *sys.argv[1:]]),
                  resume_command=recovery, monitor_command=f"cat {JOB / 'heartbeat.json'}",
                  expected_updates=45, physical_gpu_index=None,
                  host_runtime_manifest=os.environ.get("WDRO_HOST_RUNTIME_MANIFEST")) as lease:
        new_count = 0
        while not STOP:
            source_check()
            queue = read(QUEUE)
            if queue["status"] == "complete":
                break
            candidates = [t for t in queue["tasks"] if t["status"] == "complete"
                          and t["run_id"] not in checked
                          and not (EXP / "logs/jobs" / t["run_id"] / "run.lock").exists()]
            if not candidates:
                if args.once:
                    break
                lease.update(update=len(checked), status="waiting_completed_training")
                time.sleep(5)
                continue
            task = candidates[0]
            try:
                if not run_one(task, lease, checked):
                    break
            except Exception as error:
                stop_dependents(f"{type(error).__name__}: {error}")
                raise
            checked.add(task["run_id"])
            new_count += 1
            report = dict(at_utc=utc_now(), status="partial_pass", runs_passed=sorted(checked), total=45,
                          stage6_complete=False, final_audit_required=True, auditor_unchanged=True)
            atomic_write_json(JOB / "result.json", report)
            progress(f"R3已完成项完整审计{len(checked)}/45通过；整批仍待训练及总审计", str(JOB / "result.json"), True)
            print(json.dumps(dict(status="pass", run_id=task["run_id"], audited=len(checked))), flush=True)
            if args.max_new_runs and new_count >= args.max_new_runs:
                break
        lease.finish(update=len(checked), status="paused" if STOP else "complete",
                     detail=dict(final_audit_required=True, result=str(JOB / "result.json")))


def launch():
    source_check()
    assert (JOB / "foreground_acceptance.json").exists(), "foreground complete and interruption/resume checks required"
    assert read(JOB / "foreground_acceptance.json")["status"] == "pass"
    tmux = ["tmux", "-S", str(SOCKET)]
    if subprocess.run([*tmux, "has-session", "-t", SESSION], capture_output=True).returncode == 0:
        raise RuntimeError("Existing persistent auditor; inspect heartbeat")
    for path in (JOB / "run.lock", EXP / "logs/stage3_audit/stage3_completion_audit/run.lock"):
        if path.exists() and process_is_alive(read(path)["pid"]):
            raise RuntimeError(f"Existing auditor: {path}")
    preflight = cpu_preflight("precision_completed_audit_persistent_launch")
    command = ["env", f"WDRO_TMUX_SESSION={SESSION}", f"WDRO_HOST_RUNTIME_MANIFEST={preflight}",
               "CUDA_VISIBLE_DEVICES=", "bash", str(ROOT / "scripts/stage3_env.sh"), sys.executable, str(Path(__file__))]
    log = JOB / "controller.log"
    start = [*tmux, "new-session", "-d", "-s", SESSION, "-c", str(CODEX),
             f"exec {shlex.join(command)} >> {shlex.quote(str(log))} 2>&1"]
    record = dict(at_utc=utc_now(), session=SESSION, socket=str(SOCKET), socket_resolved=str(SOCKET.resolve()),
                  command=command, exact_command=shlex.join(command), tmux_command=start,
                  recovery_command=shlex.join(["bash", str(ROOT / "scripts/stage3_env.sh"), sys.executable, str(Path(__file__)), "--launch"]),
                  monitor_command=f"cat {JOB / 'heartbeat.json'}", preflight=str(preflight),
                  log=str(log), log_resolved=str(log.resolve()), operations=artifact(OPERATIONS))
    atomic_write_json(JOB / "launch.json", record)
    subprocess.run(start, check=True)
    print(json.dumps(dict(status="launched", manifest=str(JOB / "launch.json"))), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--max-new-runs", type=int)
    parser.add_argument("--audit-run")
    args = parser.parse_args()
    if args.audit_run:
        source_check()
        task = next(t for t in read(QUEUE)["tasks"] if t["run_id"] == args.audit_run)
        assert task["status"] == "complete"
        from audit_stage3 import audit_run
        print(json.dumps(audit_run(task)), flush=True)
    elif args.launch:
        launch()
    else:
        run(args)


if __name__ == "__main__":
    main()
