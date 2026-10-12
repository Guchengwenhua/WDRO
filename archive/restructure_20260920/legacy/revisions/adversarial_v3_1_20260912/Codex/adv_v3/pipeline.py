"""Durable staged execution: fixed opponents, audit, joint training, delivery."""
import argparse
import fcntl
import json
import os
import subprocess
import time
from .common import ROOT,EXP,atomic_json,append_json,sha256

PYTHON="/remote-home/Guilin/miniconda3/envs/py311/bin/python"

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--fixed-concurrency",type=int,default=16)
    parser.add_argument("--joint-concurrency",type=int,default=8)
    parser.add_argument("--peak-mib",type=int,default=2500)
    parser.add_argument("--share-busy-gpus",action="store_true")
    args=parser.parse_args()
    folder=EXP/"pipeline";folder.mkdir(parents=True,exist_ok=True)
    lock=(folder/"pipeline.lock").open("a+");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    atomic_json(folder/"launch.json",dict(pid=os.getpid(),at=time.time(),command=os.sys.argv,
        code_sha256=sha256(__file__),
        process_start_ticks=open(f"/proc/{os.getpid()}/stat").read().split()[21]))
    commands=[
        ("fixed",["-m","adv_v3.queue","--phase","fixed","--max-concurrent",str(args.fixed_concurrency),
            "--max-per-gpu","6","--peak-mib",str(args.peak_mib),"--reference-workers","0"]),
        ("fixed_audit",["-m","adv_v3.report","--phase","fixed"]),
        ("joint",["-m","adv_v3.queue","--phase","joint","--max-concurrent",str(args.joint_concurrency),
            "--max-per-gpu","3","--peak-mib",str(args.peak_mib),"--reference-workers","3"]),
        ("final_audit",["-m","adv_v3.report","--phase","all"]),
    ]
    if args.share_busy_gpus:
        for phase,tail in commands:
            if phase in {"fixed","joint"}:tail.append("--share-busy-gpus")
    try:
        for phase,tail in commands:
            cmd=["bash",str(ROOT/"run_env.sh"),PYTHON,"-u"]+tail
            logpath=folder/f"{phase}_{time.time_ns()}.log"
            atomic_json(folder/"state.json",dict(phase=phase,status="running",at=time.time(),command=cmd))
            append_json(folder/"events.jsonl",dict(event="phase_start",phase=phase,at=time.time(),command=cmd))
            with logpath.open("w") as log:
                proc=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                    start_new_session=True)
                while proc.poll() is None:
                    atomic_json(folder/"heartbeat.json",dict(pid=os.getpid(),child_pid=proc.pid,phase=phase,at=time.time()))
                    time.sleep(10)
                code=proc.returncode
            append_json(folder/"events.jsonl",dict(event="phase_exit",phase=phase,code=code,at=time.time(),
                log=str(logpath),log_sha256=sha256(logpath)))
            if code:
                atomic_json(folder/"state.json",dict(phase=phase,status="failed",code=code,at=time.time()))
                raise SystemExit(code)
        atomic_json(folder/"complete.json",dict(status="pass",at=time.time()))
        atomic_json(folder/"state.json",dict(phase="all",status="complete",at=time.time()))
    finally:
        atomic_json(folder/"exit.json",dict(pid=os.getpid(),at=time.time(),
            completed=(folder/"complete.json").exists()))
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()

if __name__=="__main__":main()
