#!/usr/bin/env python3
"""Launch/recover the low-priority, completed-run auditor in server tmux."""
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODEX = ROOT.parent
EXP = CODEX / "experiments"
sys.path.insert(0, str(ROOT))
from src.io_utils import atomic_write_json
from src.job_control import process_is_alive, utc_now
from src.rest import file_sha256
from stage3_pipeline import verify_sources, snapshot_cpu
from capture_host_runtime import gpu_sample, compute_processes


def main():
    verify_sources()
    job = EXP / "logs/stage3_completed_audit"
    # A completed formal foreground audit validates this exact auditor/cursor
    # path before detaching its repeatable driver.
    evidence = json.loads((job / "result.json").read_text())
    if evidence["status"] != "partial_pass" or not evidence["runs_passed"]:
        raise RuntimeError("One foreground formal audit must pass before persistent launch")
    for run in evidence["runs_passed"]:
        report = json.loads((EXP / "logs/stage3_audit/runs" / f"{run}.json").read_text())
        if report["status"] != "pass" or not all(report["checks"].values()):
            raise RuntimeError("Foreground formal audit did not pass")
    for lock in (job / "run.lock", EXP / "logs/stage3_audit/stage3_completion_audit/run.lock"):
        if lock.exists() and process_is_alive(int(json.loads(lock.read_text())["pid"])):
            raise RuntimeError(f"Audit is already running: {lock}")
    session = "wdro_stage3_completed_audit_v1"
    socket = EXP / "tmp/wdro_stage3.sock"
    tmux = ["tmux", "-S", str(socket)]
    env = dict(os.environ, TMUX_TMPDIR=str(EXP / "tmp"), WDRO_TMUX_SESSION=session,
               CUDA_VISIBLE_DEVICES="")
    if subprocess.run([*tmux, "has-session", "-t", session], capture_output=True, env=env).returncode == 0:
        raise RuntimeError("Completed-run audit tmux session already exists")
    preflight = snapshot_cpu("stage3_completed_audit_persistent_launch_or_recovery")
    env["WDRO_HOST_RUNTIME_MANIFEST"] = str(preflight)
    # Existing tmux servers retain their own environment. Pass these explicitly
    # so the second session cannot inherit the training session's job identity.
    command = ["env", f"WDRO_TMUX_SESSION={session}", f"WDRO_HOST_RUNTIME_MANIFEST={preflight}",
               "CUDA_VISIBLE_DEVICES=", "bash", str(ROOT / "scripts/stage3_env.sh"), sys.executable,
               str(ROOT / "scripts/audit_stage3_completed.py")]
    log = job / "controller.log"
    launch = [*tmux, "new-session", "-d", "-s", session, "-c", str(CODEX),
              f"exec {shlex.join(command)} >> {shlex.quote(str(log))} 2>&1"]
    record = {"started_at_utc": utc_now(), "session": session,
              "socket_logical": str(socket), "socket_resolved": str(socket.resolve()),
              "exact_command": shlex.join(command), "tmux_command": shlex.join(launch),
              "recovery_command": shlex.join(["bash", str(ROOT / "scripts/stage3_env.sh"),
                  sys.executable, str(Path(__file__))]),
              "monitor_command": f"cat {job / 'heartbeat.json'}", "preflight": str(preflight),
              "log_logical": str(log), "log_resolved": str(log.resolve()),
              "launcher_sha256": file_sha256(Path(__file__)), "research_protocol_override": False}
    atomic_write_json(job / "launch.json", record)
    subprocess.run(launch, check=True, env=env)
    for _ in range(30):
        pid_path = job / "pid.json"
        if pid_path.exists():
            pid = int(json.loads(pid_path.read_text())["pid"])
            if process_is_alive(pid):
                record["pid"] = pid
                break
        time.sleep(0.5)
    if "pid" not in record:
        raise RuntimeError(f"Auditor did not start; inspect {log}")
    record.update(status="running", post_launch_gpus=gpu_sample(),
                  post_launch_compute_processes=compute_processes())
    atomic_write_json(job / "launch.json", record)
    print(json.dumps({k: record[k] for k in ("status", "session", "pid", "preflight", "monitor_command")}))


if __name__ == "__main__":
    main()
