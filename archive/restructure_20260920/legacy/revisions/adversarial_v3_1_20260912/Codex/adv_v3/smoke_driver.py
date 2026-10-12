import json
import os
import subprocess
import time
from .common import ROOT,EXP,atomic_json,sha256

def main():
    commands=json.loads((ROOT/"manifests/smoke_commands.json").read_text())
    jobs=[]
    for item in commands:
        logpath=EXP/"logs"/f"gpu_smoke_d{item['degree']}_{time.time_ns()}.log"
        log=logpath.open("w")
        proc=subprocess.Popen(item["cmd"],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(item["gpu"])),
            stdout=log,stderr=subprocess.STDOUT)
        jobs.append((proc,log,logpath,item))
    results=[]
    for proc,log,path,item in jobs:
        code=proc.wait();log.close()
        results.append(dict(degree=item["degree"],gpu=item["gpu"],pid=proc.pid,exit_code=code,
            log=str(path),log_sha256=sha256(path)))
    passed=all(x["exit_code"]==0 for x in results)
    atomic_json(ROOT/"manifests/gpu_smoke.json",dict(status="pass" if passed else "fail",results=results))
    raise SystemExit(0 if passed else 1)

if __name__=="__main__":main()
