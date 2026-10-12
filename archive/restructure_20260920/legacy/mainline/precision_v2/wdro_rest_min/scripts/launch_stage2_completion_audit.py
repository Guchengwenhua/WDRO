#!/usr/bin/env python3
"""Fresh preflight, isolated tmux socket, persistent audit, and launch metadata."""
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
CODEX = PROJECT.parent
EXP = CODEX / "experiments"
sys.path.insert(0, str(PROJECT))
from src.io_utils import atomic_write_json
from src.job_control import utc_now
from capture_host_runtime import compute_processes, gpu_sample


def main():
    env = dict(os.environ)
    env.update({key: str(EXP / "tmp") for key in ("TMPDIR", "TMP", "TEMP", "TMUX_TMPDIR")})
    env.update(PYTHONDONTWRITEBYTECODE="1", XDG_CACHE_HOME=str(EXP / "cache/runtime"),
               TORCH_HOME=str(EXP / "cache/runtime/torch"), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="", WDRO_TMUX_SESSION="wdro_stage2_audit_v1")
    tmux = ["tmux", "-L", "wdro_stage2_audit"]
    session = env["WDRO_TMUX_SESSION"]
    if subprocess.run([*tmux, "has-session", "-t", session], env=env, capture_output=True).returncode == 0:
        raise RuntimeError("audit tmux session is already alive")
    preflight = subprocess.run([sys.executable, str(PROJECT / "scripts/capture_host_runtime.py"),
                                "--purpose", "stage2_full_completion_audit_cpu", "--measured-peak-mib", "4224",
                                "--safety-margin-mib", "2048"], env=env, check=True, text=True, capture_output=True)
    runtime = json.loads(preflight.stdout)
    command = [sys.executable, str(PROJECT / "scripts/audit_stage2_formal.py"),
               "--host-runtime-manifest", runtime["manifest"]]
    job_root = EXP / "logs/stage2/stage2_formal_audit_job"
    job_root.mkdir(parents=True, exist_ok=True)
    log = job_root / "stdout.log"
    tmux_command = [*tmux, "new-session", "-d", "-s", session, "-c", str(PROJECT),
                    f"exec {shlex.join(command)} >> {shlex.quote(str(log))} 2>&1"]
    launch = {"started_at_utc": utc_now(), "tmux_session": session, "tmux_socket_name": "wdro_stage2_audit",
              "exact_command": shlex.join(command), "tmux_launch_command": shlex.join(tmux_command),
              "stdout_log": str(log), "job": str(job_root / "job.json"),
              "heartbeat": str(job_root / "heartbeat.json"), "host_runtime": runtime,
              "environment": {key: env[key] for key in ("TMPDIR", "TMP", "TEMP", "TMUX_TMPDIR", "PYTHONDONTWRITEBYTECODE",
                  "XDG_CACHE_HOME", "TORCH_HOME", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "CUDA_VISIBLE_DEVICES")},
              "monitor_command": f"tail -f {log}",
              "recovery_command": f"{sys.executable} {PROJECT / 'scripts/launch_stage2_completion_audit.py'}",
              "artifact_root": {"logical_path": str(EXP), "resolved_path": str(EXP.resolve())},
              "gpu_mapping": None, "device": "cpu", "status": "launching"}
    atomic_write_json(job_root / "launch.json", launch)
    subprocess.run(tmux_command, env=env, check=True)
    for _ in range(20):
        if (job_root / "job.json").exists():
            launch["pid"] = json.loads((job_root / "job.json").read_text())["pid"]
            break
        time.sleep(0.25)
    launch.update(status="launched", post_launch_at_utc=utc_now(), post_launch_gpus=gpu_sample(),
                  post_launch_compute_processes=compute_processes())
    atomic_write_json(job_root / "launch.json", launch)
    print(json.dumps({key: launch[key] for key in ("status", "pid", "stdout_log", "tmux_session", "host_runtime")}))


if __name__ == "__main__":
    main()
