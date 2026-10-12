"""Live-capacity scheduler for isolated GPU experiments."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
import subprocess
import shutil
import time
from pathlib import Path
from .common import ROOT,EXP,atomic_json,append_json,sha256

PYTHON="/remote-home/Guilin/miniconda3/envs/py311/bin/python"

def process_active(pid, start):
    try:
        fields=Path(f"/proc/{pid}/stat").read_text().split()
        return fields[2] not in {"Z","X"} and fields[21]==str(start)
    except FileNotFoundError:return False

class AdoptedProcess:
    def __init__(self,launch,out):
        self.pid=launch["pid"];self.start=launch["process_start_ticks"];self.out=out
    def poll(self):
        if process_active(self.pid,self.start):return None
        return 0 if (self.out/"complete.json").exists() else 1

def valid_complete(path):
    if not path.exists():return False
    obj=json.loads(path.read_text())
    if obj.get("status")!="pass":raise RuntimeError(f"Nonpass completion: {path}")
    for item in obj["artifacts"]:
        if sha256(item["path"])!=item["sha256"]:raise RuntimeError(f"Artifact mismatch: {item['path']}")
    identity=obj.get("identity",{})
    protocol_hash=identity.get("protocol_sha256",identity.get("protocol",{}).get("sha256"))
    if protocol_hash and protocol_hash!=sha256(ROOT/"protocol.json"):
        raise RuntimeError("Completed protocol identity changed")
    for name,digest in identity.get("source_hashes",identity.get("source",{})).items():
        if sha256(ROOT/"adv_v3"/name)!=digest:raise RuntimeError("Completed task source changed")
    return True

def gpu_snapshot():
    output=subprocess.check_output([
        "nvidia-smi","--query-gpu=index,uuid,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits"],text=True)
    rows=[]
    for line in output.strip().splitlines():
        i,uuid,total,used,free,util=[s.strip() for s in line.split(",")]
        rows.append(dict(index=int(i),uuid=uuid,total=int(total),used=int(used),free=int(free),util=int(util)))
    return rows

def task_list(phase):
    if phase=="fixed":
        return [dict(id=f"{parent}_d{degree}_s{seed}",parent=parent,degree=degree,seed=seed)
            for degree in [50,40,30,20,10] for seed in [101,102,103] for parent in ["erm","wdro_rest"]]
    methods=["random_top4","critic_calibrated4","rl_frozen4","rl_joint4","random64","es64"]
    return [dict(id=f"{method}_d30_s{seed}",method=method,degree=30,seed=seed)
        for seed in [101,102,103] for method in methods]

def run(args):
    folder=EXP/"queue"/args.phase;folder.mkdir(parents=True,exist_ok=True)
    handle=(folder/"queue.lock").open("a+")
    fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    tasks=task_list(args.phase);active={};attempts={};completed=set()
    for task in tasks:
        out=EXP/args.phase/task["id"]
        launch_path=out/"launch.json"
        if launch_path.exists():
            launch=json.loads(launch_path.read_text())
            if process_active(launch["pid"],launch["process_start_ticks"]):
                active[task["id"]]=dict(process=AdoptedProcess(launch,out),
                    gpu=launch["physical_gpu_index"],started=time.monotonic(),log=None,
                    reference_workers=launch.get("reference_workers",3))
                append_json(folder/"events.jsonl",dict(event="adopt",task=task["id"],pid=launch["pid"],at=time.time()))
                continue
        if valid_complete(out/"complete.json"):completed.add(task["id"])
    atomic_json(folder/"tasks.json",tasks)
    atomic_json(folder/"launch.json",dict(pid=os.getpid(),at=time.time(),command=os.sys.argv,
        process_start_ticks=Path(f"/proc/{os.getpid()}/stat").read_text().split()[21]))
    samples=[]
    try:
        while True:
            for name,a in list(active.items()):
                rc=a["process"].poll()
                if rc is None:continue
                if a["log"] is not None:a["log"].close()
                append_json(folder/"events.jsonl",dict(event="exit",task=name,code=rc,gpu=a["gpu"],at=time.time()))
                active.pop(name)
                if rc==0 and valid_complete(EXP/args.phase/name/"complete.json"):completed.add(name)
            pending=[t for t in tasks if t["id"] not in completed and t["id"] not in active]
            if not pending and not active:
                atomic_json(folder/"complete.json",dict(status="pass",tasks=len(tasks),at=time.time()))
                break
            failed=[t for t in pending if attempts.get(t["id"],0)>=3]
            if failed and not active:
                atomic_json(folder/"blocked.json",dict(status="failed_tasks",tasks=failed,at=time.time()))
                raise RuntimeError("Tasks failed three attempts; preserve evidence for repair")
            gpus=gpu_snapshot();samples.append(gpus);samples=samples[-3:]
            atomic_json(folder/"heartbeat.json",dict(pid=os.getpid(),at=time.time(),
                active=[dict(task=k,gpu=v["gpu"],pid=v["process"].pid) for k,v in active.items()],
                pending=len(pending),gpus=gpus))
            if len(samples)>=3 and shutil.disk_usage(EXP).free/2**30>=100:
                for task in pending:
                    if len(active)>=args.max_concurrent:break
                    if attempts.get(task["id"],0)>=3:continue
                    reference_workers=(args.reference_workers if args.reference_workers else
                        (3 if task["degree"]>=40 else (2 if task["degree"]>=20 else 1)))
                    if sum(a["reference_workers"] for a in active.values())+reference_workers>args.cpu_worker_budget:continue
                    options=[]
                    for g in gpus:
                        idx=g["index"]
                        on_gpu=[a for a in active.values() if a["gpu"]==idx]
                        if len(on_gpu)>=args.max_per_gpu:continue
                        # Conservatively reserve every admitted job's future peak. Namespace PID
                        # mappings may hide CUDA registration; no time-based release is safe.
                        pending_memory=len(on_gpu)*(args.peak_mib+1024)
                        util=max(next(row["util"] for row in s if row["index"]==idx) for s in samples)
                        free=min(next(row["free"] for row in s if row["index"]==idx) for s in samples)
                        if free-pending_memory<args.peak_mib+3072:continue
                        if util>=95 and not args.share_busy_gpus:continue
                        options.append((len(on_gpu),util,-free,idx))
                    if not options:break
                    gpu=min(options)[-1]
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),ADV_V3_REFERENCE_WORKERS=str(reference_workers))
                    module="adv_v3.worker" if args.phase=="fixed" else "adv_v3.joint"
                    cmd=["bash",str(ROOT/"run_env.sh"),PYTHON,"-u","-m",module,
                        "--degree",str(task["degree"]),"--seed",str(task["seed"]),
                        "--physical-gpu-index",str(gpu)]
                    cmd += ["--parent",task["parent"]] if args.phase=="fixed" else ["--method",task["method"]]
                    attempt=attempts.get(task["id"],0)+1;attempts[task["id"]]=attempt
                    logpath=folder/f'{task["id"]}_{time.time_ns()}_a{attempt}.log'
                    log=logpath.open("w")
                    proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                    active[task["id"]]=dict(process=proc,gpu=gpu,started=time.monotonic(),log=log,
                        reference_workers=reference_workers)
                    append_json(folder/"events.jsonl",dict(event="launch",task=task["id"],gpu=gpu,pid=proc.pid,
                        command=cmd,at=time.time(),log=str(logpath),admission=gpus))
            time.sleep(5)
    finally:
        # Never orphan launched workers on controller error: their own checkpoint/lock remains authoritative.
        atomic_json(folder/"controller_exit.json",dict(at=time.time(),pid=os.getpid(),
            active_pids=[a["process"].pid for a in active.values()]))
        fcntl.flock(handle,fcntl.LOCK_UN);handle.close()

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--phase",choices=["fixed","joint"],required=True)
    p.add_argument("--max-concurrent",type=int,default=16)
    p.add_argument("--max-per-gpu",type=int,default=5)
    p.add_argument("--peak-mib",type=int,default=6000)
    p.add_argument("--reference-workers",type=int,default=1)
    p.add_argument("--cpu-worker-budget",type=int,default=28)
    p.add_argument("--share-busy-gpus",action="store_true",
        help="Allow CUDA time sharing when GPU compute is busy; all memory and CPU admission limits remain enforced")
    run(p.parse_args())
