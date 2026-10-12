#!/usr/bin/env python3
"""Refresh capacity evidence and launch/recover the stage-3 tmux pipeline."""
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];CODEX=ROOT.parent;EXP=CODEX/"experiments"
sys.path.insert(0,str(ROOT))
from src.io_utils import atomic_write_json
from src.job_control import process_is_alive,utc_now
from capture_host_runtime import gpu_sample,compute_processes
from stage3_pipeline import verify_sources


def main():
    verify_sources()
    evidence=json.loads((EXP/"logs/stage3_smoke/audit_final.json").read_text())
    gate=json.loads((EXP/"logs/stage3_audit/final_smoke_gate.json").read_text())
    if evidence["status"]!="pass" or gate["status"]!="pass":raise RuntimeError("final smoke gates must pass")
    env=dict(os.environ)
    env["TMUX_TMPDIR"]=str(EXP/"tmp")
    env["WDRO_TMUX_SESSION"]="wdro_stage3_baselines_v1"
    socket=EXP/"tmp/wdro_stage3.sock"
    tmux=["tmux","-S",str(socket)]
    if subprocess.run([*tmux,"has-session","-t",env["WDRO_TMUX_SESSION"]],capture_output=True,env=env).returncode==0:
        raise RuntimeError("stage-3 tmux pipeline already exists")
    job=EXP/"logs/stage3_pipeline";job.mkdir(parents=True,exist_ok=True)
    if (job/"run.lock").exists():
        lock=json.loads((job/"run.lock").read_text())
        if process_is_alive(int(lock["pid"])):raise RuntimeError("stage-3 controller PID is alive")
    capture=subprocess.run([sys.executable,str(ROOT/"scripts/capture_host_runtime.py"),
        "--purpose","stage3_pipeline_launch_or_recovery","--measured-peak-mib","4626","--safety-margin-mib","2048"],
        capture_output=True,text=True,check=True,env=env)
    runtime=json.loads(capture.stdout)
    env["WDRO_HOST_RUNTIME_MANIFEST"]=runtime["manifest"]
    command=["bash",str(ROOT/"scripts/stage3_env.sh"),sys.executable,str(ROOT/"scripts/stage3_pipeline.py")]
    log=job/"controller.log"
    launch_command=[*tmux,"new-session","-d","-s",env["WDRO_TMUX_SESSION"],"-c",str(ROOT),f"exec {shlex.join(command)} >> {shlex.quote(str(log))} 2>&1"]
    record={"started_at_utc":utc_now(),"session":env["WDRO_TMUX_SESSION"],"socket_logical":str(socket),"socket_resolved":str(socket.resolve()),
        "exact_command":shlex.join(command),"tmux_command":shlex.join(launch_command),"preflight":runtime,
        "log":str(log),"log_resolved":str(log.resolve()),"heartbeat":str(job/"heartbeat.json"),
        "monitor_command":f"cat {job/'heartbeat.json'}", "recovery_command":shlex.join(["bash",str(ROOT/"scripts/stage3_env.sh"),sys.executable,str(Path(__file__))])}
    atomic_write_json(job/"launch.json",record)
    subprocess.run(launch_command,check=True,env=env)
    for _ in range(30):
        if (job/"pid.json").exists():
            record["pid"]=json.loads((job/"pid.json").read_text())["pid"]
            if process_is_alive(record["pid"]):break
        time.sleep(0.25)
    record["post_launch_gpus"]=gpu_sample();record["post_launch_processes"]=compute_processes()
    record["status"]="running" if process_is_alive(int(record.get("pid",-1))) else "launch_failed"
    atomic_write_json(job/"launch.json",record)
    print(json.dumps({k:record[k] for k in ("status","session","pid","log","heartbeat","preflight")}))


if __name__=="__main__":main()
