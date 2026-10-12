#!/usr/bin/env python3
"""Foreground, individually preflighted baseline memory and resume acceptance."""
from __future__ import annotations
import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODEX = ROOT.parent
sys.path.insert(0, str(ROOT))
from src.io_utils import atomic_write_json, append_jsonl
from src.job_control import utc_now

PYTHON = sys.executable
ENV = ROOT / "scripts/stage3_env.sh"
LOGS = CODEX / "experiments/logs/stage3_smoke"
LOGS.mkdir(parents=True, exist_ok=True)


def launch(method, degree, run_id, stop, resume=False):
    result_path = CODEX / 'experiments/logs/stage3' / run_id / 'result.json'
    if result_path.exists():
        previous = json.loads(result_path.read_text())
        from src.geosteiner import reference_identity
        assert previous['run_identity']['reference_solver'] == reference_identity()
        if previous['completed_updates'] >= stop:
            return previous
    if (CODEX / 'experiments/checkpoints/stage3' / run_id / 'latest.pt').exists():
        resume = True
    capture = subprocess.run([PYTHON, str(ROOT / "scripts/capture_host_runtime.py"),
        "--purpose", f"stage3_smoke_{run_id}_stop{stop}",
        "--measured-peak-mib", "3602", "--safety-margin-mib", "8192"],
        capture_output=True, text=True, check=True)
    summary = json.loads(capture.stdout.strip().splitlines()[-1])
    manifest = json.loads(Path(summary["manifest"]).read_text())
    eligible = [r for r in manifest["gpu_decisions"] if r["eligible"]]
    gpu = max(eligible, key=lambda r: r["minimum_free_memory_mib"])["index"]
    command = ["bash", str(ENV), PYTHON, str(ROOT / "scripts/train_baseline.py"),
        "--method", method, "--degree", str(degree), "--seed", "101", "--scope", "stage3_smoke",
        "--run-id", run_id, "--stop-after", str(stop), "--physical-gpu-index", str(gpu),
        "--host-runtime-manifest", summary["manifest"], "--geosteiner-workers", "3"]
    if resume:
        command.append("--resume")
    log = LOGS / f"{run_id}_to{stop}.log"
    with log.open("a") as out:
        process = subprocess.Popen(command, stdout=out, stderr=subprocess.STDOUT)
        record = {"at_utc": utc_now(), "run_id": run_id, "pid": process.pid,
                  "command": command, "gpu": gpu, "preflight": summary, "log": str(log),
                  "log_resolved": str(log.resolve())}
        append_jsonl(LOGS / "launches.jsonl", record)
        print(json.dumps(record), flush=True)
        # Post-launch samples are captured immediately, including compute PIDs.
        subprocess.run([PYTHON, str(ROOT / "scripts/capture_host_runtime.py"),
            "--purpose", f"stage3_smoke_post_launch_{run_id}", "--measured-peak-mib", "3576",
            "--safety-margin-mib", "8192"], stdout=out, stderr=subprocess.STDOUT)
        code = process.wait()
    if code:
        raise RuntimeError(f"smoke failed ({code}): {log}")
    result_path = CODEX / "experiments/logs/stage3" / run_id / "result.json"
    result = json.loads(result_path.read_text())
    print(json.dumps({"run_id": run_id, "update": result["completed_updates"],
        "peak_reserved_mib": result["peak_memory_reserved_bytes"] / 2**20,
        "seconds": result["segment_wall_seconds"]}), flush=True)
    return result


def assert_same(left, right, path="root"):
    import torch
    import numpy as np
    if isinstance(left, torch.Tensor):
        if not torch.equal(left.cpu(), right.cpu()):
            raise AssertionError(path)
    elif isinstance(left, np.ndarray):
        if not np.array_equal(left, right):
            raise AssertionError(path)
    elif isinstance(left, dict):
        if left.keys() != right.keys():
            raise AssertionError(path + ".keys")
        for k in left:
            assert_same(left[k], right[k], path+"."+str(k))
    elif isinstance(left, (list, tuple)):
        if len(left) != len(right):
            raise AssertionError(path + ".length")
        for i, (a,b) in enumerate(zip(left,right)):
            assert_same(a,b,path+f"[{i}]")
    elif left != right:
        raise AssertionError(path)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=["mixed_erm", "group_dro", "hard_mining"])
    parser.add_argument("--tag", default="v1")
    parser.add_argument("--reuse-memory-profile", type=Path)
    args=parser.parse_args()
    audit={"status":"running", "started_at_utc":utc_now(), "methods":{}, "profiles":{}}
    for method in args.methods:
        continuous=f"smoke_{args.tag}_{method}_d30_continuous"
        split=f"smoke_{args.tag}_{method}_d30_split"
        a=launch(method,30,continuous,12)
        launch(method,30,split,10)
        b=launch(method,30,split,12,True)
        import torch
        ca=torch.load(a["latest_checkpoint"],map_location="cpu",weights_only=False)
        cb=torch.load(b["latest_checkpoint"],map_location="cpu",weights_only=False)
        fields=["actor_state_dict","critic_state_dict","optimizer_state_dict","scheduler_state_dict",
            "rng_state","training_stream_hash","method_state","call_ledger","initialization","best"]
        for field in fields:
            assert_same(ca[field],cb[field],field)
        audit["methods"][method]={"resume":"pass", "compared_fields":fields,
            "continuous":a["latest_checkpoint"],"split":b["latest_checkpoint"],
            "candidate_refresh_boundary_crossed": method=="hard_mining"}
        audit["profiles"][method]={"30":{"peak_reserved_mib":math.ceil(max(a["peak_memory_reserved_bytes"],b["peak_memory_reserved_bytes"])/2**20), "evidence":a["event_log"]}}
        if args.reuse_memory_profile:
            previous=json.loads(args.reuse_memory_profile.read_text())
            for degree in (10,20,40,50):
                audit["profiles"][method][str(degree)]=previous["profiles"][method][str(degree)]
        for degree in (() if args.reuse_memory_profile else (10,20,40,50)):
            result=launch(method,degree,f"smoke_{args.tag}_{method}_d{degree}_memory",2)
            audit["profiles"][method][str(degree)]={"peak_reserved_mib":math.ceil(result["peak_memory_reserved_bytes"]/2**20),"evidence":result["event_log"]}
        atomic_write_json(LOGS/f"audit_{args.tag}.json",audit)
    audit.update(status="pass",completed_at_utc=utc_now())
    atomic_write_json(LOGS/f"audit_{args.tag}.json",audit)
    print(json.dumps({"status":"pass","audit":str(LOGS/f"audit_{args.tag}.json")}),flush=True)


if __name__=="__main__":
    main()
