#!/usr/bin/env python3
"""Persistent stage 3 only: training -> fresh CPU audit preflight -> export."""
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
from src.job_control import JobLease,utc_now,process_is_alive
from src.rest import file_sha256
from capture_host_runtime import gpu_sample,compute_processes,disk_snapshot

JOB=EXP/"logs/stage3_pipeline"


def snapshot_cpu(purpose):
    record={"kind":"stage3_cpu_audit_preflight","purpose":purpose,"at_utc":utc_now(),
        "gpus":gpu_sample(),"compute_processes":compute_processes(),
        "disk":{"codex":disk_snapshot(CODEX),"experiments":disk_snapshot(EXP)},"research_protocol_override":False}
    record["passed"]=record["disk"]["experiments"]["free_gib"]>=100
    target=JOB/f"cpu_preflight_{utc_now().replace(':','').replace('.','_')}.json"
    atomic_write_json(target,record)
    if not record["passed"]:raise RuntimeError("CPU audit artifact disk floor failed")
    return target


def verify_sources():
    failure=EXP/"logs/stage3_completed_audit/failure.json"
    if failure.exists():raise RuntimeError(f"unresolved completed-run audit failure: {failure}")
    contract=ROOT/"manifests/stage3_launch_contract_v1.json"
    frozen=json.loads(contract.read_text())
    expected=dict(frozen["source_and_config_hashes"])
    operations=ROOT/"manifests/stage3_operations_patch_v1.json"
    if operations.exists():
        patch=json.loads(operations.read_text())
        allowed={"wdro_rest_min/scripts/run_stage3_baseline_queue.py","wdro_rest_min/scripts/stage3_pipeline.py","wdro_rest_min/tests/test_stage3_operations.py"}
        if patch["base_contract_sha256"]!=file_sha256(contract) or not set(patch["operation_hashes"])<=allowed:
            raise RuntimeError("invalid host-operations patch; training/config overrides are forbidden")
        expected.update(patch["operation_hashes"])
    operations_v2=ROOT/"manifests/stage3_operations_patch_v2.json"
    if operations_v2.exists():
        patch=json.loads(operations_v2.read_text())
        allowed={"wdro_rest_min/scripts/stage3_pipeline.py","wdro_rest_min/scripts/audit_stage3_completed.py","wdro_rest_min/tests/test_stage3_completed_audit.py"}
        if patch["base_contract_sha256"]!=file_sha256(contract) or patch["previous_patch_sha256"]!=file_sha256(operations) or not set(patch["operation_hashes"])<=allowed:
            raise RuntimeError("invalid completed-audit operations patch")
        expected.update(patch["operation_hashes"])
    for logical,digest in expected.items():
        if file_sha256(CODEX/logical)!=digest:raise RuntimeError(f"frozen source drift: {logical}")
    for logical,digest in frozen["legacy_hashes"].items():
        if file_sha256(Path(logical))!=digest:raise RuntimeError(f"legacy source drift: {logical}")


def main():
    exact=shlex.join([sys.executable,str(Path(__file__)),*sys.argv[1:]])
    recovery=shlex.join(["bash",str(ROOT/"scripts/stage3_env.sh"),sys.executable,str(ROOT/"scripts/launch_stage3.py")])
    with JobLease(JOB,run_id="stage3_pipeline_v1",exact_command=exact,resume_command=recovery,
        monitor_command=f"cat {JOB/'heartbeat.json'}",expected_updates=3,physical_gpu_index=None,
        host_runtime_manifest=os.environ.get("WDRO_HOST_RUNTIME_MANIFEST")) as lease:
        stages=[("training",[sys.executable,str(ROOT/"scripts/run_precision_training_queue.py"),"--stage","3",
                             "--max-simultaneous","8","--max-per-gpu","2","--launch-stagger-seconds","15"]),
                ("audit",[sys.executable,str(ROOT/"scripts/audit_stage3.py"),"--workers","8"]),
                ("export",[sys.executable,str(ROOT/"scripts/export_stage3.py")])]
        for index,(name,command) in enumerate(stages):
            verify_sources()
            previous_job=JOB/f"{name}_job.json"
            if previous_job.exists():
                previous=json.loads(previous_job.read_text())
                pid=int(previous.get("pid",-1))
                cmdline=Path(f"/proc/{pid}/cmdline")
                if process_is_alive(pid) and cmdline.exists() and command[1].encode() in cmdline.read_bytes():
                    while process_is_alive(pid) and cmdline.exists() and command[1].encode() in cmdline.read_bytes():
                        lease.update(update=index,status="running",detail={"substep":name,"adopted_pid":pid,"log":previous["log"]})
                        time.sleep(15)
                    # Re-enter the same idempotent command: completed work is
                    # verified/skipped, incomplete cursors resume in place.
            env=dict(os.environ)
            if name=="audit":
                # Early audit and final audit share hash-bound reconstruction
                # cursors. Wait for the early job to avoid concurrent writers.
                ahead=EXP/"logs/stage3_completed_audit/run.lock"
                while ahead.exists():
                    pid=int(json.loads(ahead.read_text()).get("pid",-1))
                    if not process_is_alive(pid):break
                    lease.update(update=index,status="running",detail={"substep":"waiting_completed_audit","pid":pid})
                    time.sleep(15)
                verify_sources()
                env["WDRO_HOST_RUNTIME_MANIFEST"]=str(snapshot_cpu("stage3_complete_input_audit"))
                env["CUDA_VISIBLE_DEVICES"]=""
                p=CODEX/"01_WDRO-REST_项目进度.md"
                s=p.read_text().replace("| 下一动作 | 3.2 正式 45-run 队列运行；训练完成后自动 3.3 全量审计与导出，止于阶段 3 |",
                    "| 下一动作 | 3.2 的 45-run 训练完成；正在 3.3 全量训练/候选重建审计，通过后自动导出并停止于阶段 3 |")
                p.write_text(s)
            log=JOB/f"{name}.log"
            with log.open("a") as out:
                child=subprocess.Popen(command,env=env,stdout=out,stderr=subprocess.STDOUT)
                atomic_write_json(JOB/f"{name}_job.json",{"stage":name,"pid":child.pid,"exact_command":shlex.join(command),"log":str(log),"log_resolved":str(log.resolve()),"started_at_utc":utc_now()})
                while child.poll() is None:
                    lease.update(update=index,status="running",detail={"substep":name,"pid":child.pid,"log":str(log)})
                    time.sleep(15)
                if child.returncode:
                    raise RuntimeError(f"stage3 {name} failed with exit {child.returncode}: {log}")
        lease.finish(update=3,status="complete",detail={"completion":str(ROOT/"manifests/stage3_completion_v1.json"),"stage4_started":False})


if __name__=="__main__":main()
