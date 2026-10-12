"""Bounded real-GPU main-policy 4-step versus 2+2-step recovery diagnostic.

The only overrides are output roots and authorized smoke counts. Every Actor,
REST and GeoSteiner call is real, and the resumed segment starts in a new OS
process. This script never writes into formal or ordinary smoke task roots.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _child(args):
    import torch
    from adv_v3 import joint

    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.gpu):
        raise RuntimeError("physical GPU binding does not match diagnostic request")
    if os.environ.get("ADV_V3_REFERENCE_WORKERS") != "2":
        raise RuntimeError("diagnostic requires exactly two reference workers")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("diagnostic requires one visible real CUDA GPU")
    original_settings = joint._settings

    def settings(smoke):
        if not smoke:
            raise RuntimeError("diagnostic may only run smoke settings")
        value = original_settings(True)
        value.update(updates=4, batch_size=4, validation_interval=2,
                     validation_count=4, evaluation_count=8,
                     independent_attack_count=2, evaluation_batch_size=4)
        return value

    joint._settings = settings
    branch = "continuous" if args.part == "continuous" else "resumed"
    joint.EXP = Path(args.output) / branch
    task = SimpleNamespace(degree=30, seed=101, method="random_top4",
                           physical_gpu_index=args.gpu, smoke=True)
    print(json.dumps({"event": "real_gpu_child_start", "part": args.part,
                      "pid": os.getpid(), "gpu": args.gpu,
                      "device": torch.cuda.get_device_name(0),
                      "reference_workers": 2, "fake_oracle": False}), flush=True)
    result = joint.run(task, stop_after=2 if args.part == "pause" else None)
    expected = "paused" if args.part == "pause" else "pass"
    if result["status"] != expected:
        raise RuntimeError(f"unexpected child result: {result['status']}")
    output = {"part": args.part, "status": result["status"], "pid": os.getpid(),
              "gpu": args.gpu, "updates": result.get("updates", result.get("update")),
              "best_update": result.get("best_update"), "ledger": result.get("ledger"),
              "fake_oracle": False}
    _write(Path(args.output) / f"{args.part}_process.json", output)
    print(json.dumps(output), flush=True)


def _compare(left, right, path, errors, counts):
    import numpy as np
    import torch

    if isinstance(left, torch.Tensor):
        counts["tensor_leaves"] += 1
        if not isinstance(right, torch.Tensor) or left.shape != right.shape or left.dtype != right.dtype or not torch.equal(left, right):
            errors.append(path)
    elif isinstance(left, np.ndarray):
        counts["array_leaves"] += 1
        if not isinstance(right, np.ndarray) or left.dtype != right.dtype or not np.array_equal(left, right):
            errors.append(path)
    elif isinstance(left, dict):
        if not isinstance(right, dict) or set(left) != set(right):
            errors.append(path + ".keys")
            return
        for key in left:
            _compare(left[key], right[key], f"{path}.{key}", errors, counts)
    elif isinstance(left, (tuple, list)):
        if type(left) is not type(right) or len(left) != len(right):
            errors.append(path + ".shape")
            return
        for index, (a, b) in enumerate(zip(left, right)):
            _compare(a, b, f"{path}[{index}]", errors, counts)
    else:
        counts["scalar_leaves"] += 1
        if left != right:
            errors.append(path)


def _parent(args):
    import torch

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ("continuous", "resumed")):
        raise FileExistsError("diagnostic output already contains runs; preserve it and use a new output root")
    gpu_query = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,uuid,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
        text=True, capture_output=True, check=True,
    ).stdout
    row = next(line for line in gpu_query.splitlines() if line.split(",")[0].strip() == str(args.gpu))
    values = [value.strip() for value in row.split(",")]
    if int(values[4]) < 8192 or int(values[5]) > 20:
        raise RuntimeError(f"GPU preflight admission failed: {row}")
    preflight = {"at_unix": time.time(), "all_gpus": gpu_query, "selected_gpu": row,
                 "reference_workers": 2, "fake_oracle": False,
                 "source": {name: _sha(ROOT / "adv_v3" / name) for name in ("joint.py", "common.py", "search.py")},
                 "diagnostic_source_sha256": _sha(__file__)}
    _write(output / "preflight.json", preflight)
    began = time.monotonic()
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), ADV_V3_REFERENCE_WORKERS="2",
                       OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONDONTWRITEBYTECODE="1")
    for part in ("continuous", "pause", "resume"):
        print(json.dumps({"event": "launch", "part": part, "at_unix": time.time()}), flush=True)
        command = [sys.executable, "-B", str(Path(__file__).resolve()), "--gpu", str(args.gpu),
                   "--output", str(output.resolve()), "--part", part]
        with (output / f"{part}.log").open("w") as log:
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
            _write(output / f"{part}_launch.json", {"pid": process.pid, "command": command,
                                                    "at_unix": time.time(), "gpu": args.gpu})
            code = process.wait()
        if code != 0:
            _write(output / "comparison.json", {"status": "fail", "failed_part": part, "exit_code": code,
                                               "fake_oracle": False, "log": str(output / f"{part}.log")})
            raise RuntimeError(f"{part} exited {code}; see {output / f'{part}.log'}")
    relative = Path("joint_smoke/random_top4_d30_s101")
    directories = [output / name / relative for name in ("continuous", "resumed")]
    latest = [torch.load(directory / "latest.pt", map_location="cpu", weights_only=False) for directory in directories]
    completed = [json.loads((directory / "complete.json").read_text()) for directory in directories]
    errors, counts = [], {"tensor_leaves": 0, "array_leaves": 0, "scalar_leaves": 0}
    keys = ("update", "price", "actor_state_dict", "critic_state_dict", "optimizer_state_dict", "rng_state")
    for key in keys:
        _compare(latest[0]["current"][key], latest[1]["current"][key], f"current.{key}", errors, counts)
    for key in ("best_update", "initial_price", "final_price", "best_price", "rho", "ledger"):
        _compare(completed[0][key], completed[1][key], key, errors, counts)
    for i, (a, b) in enumerate(zip(completed[0]["history"], completed[1]["history"])):
        for key in ("update", "input_sha256", "selected_sha256", "lambda_before", "lambda_after", "mean_cost",
                    "mean_search_score", "mean_search_regret", "search_ledger_delta", "sampled"):
            _compare(a[key], b[key], f"history[{i}].{key}", errors, counts)
    _compare(latest[0]["best"]["actor_state_dict"], latest[1]["best"]["actor_state_dict"], "best.actor_state_dict", errors, counts)
    source_after = {name: _sha(ROOT / "adv_v3" / name) for name in preflight["source"]}
    if source_after != preflight["source"]:
        errors.append("frozen_source_changed_during_diagnostic")
    for done in completed:
        if done["updates"] != 4 or len(done["history"]) != 4:
            errors.append("incorrect_completed_updates")
        for key in ("actor_requested", "actor_completed", "rest_requested", "rest_completed", "reference_requests"):
            if done["ledger"].get(key) != 688:
                errors.append(f"unexpected_{key}_total")
        for key in ("sampled_actor_requested", "sampled_actor_completed", "sampled_rest_requested",
                    "sampled_rest_completed", "sampled_reference_reuses"):
            if done["ledger"].get(key) != 16:
                errors.append(f"unexpected_{key}_total")
    result = {
        "status": "pass" if not errors else "fail", "fake_oracle": False,
        "description": "Real parent ERM Actor/Critic, real REST and precision_v2 GeoSteiner; three separate OS processes",
        "degree": 30, "seed": 101, "method": "random_top4", "physical_gpu": args.gpu,
        "continuous_updates": 4, "resumed_updates": [2, 2], "reference_workers": 2,
        "compared": list(keys) + ["best_actor", "lambda", "center_hashes", "selection_hashes", "sampled_losses", "true_call_ledger"],
        "comparison_counts": counts, "mismatch_paths": errors,
        "continuous_ledger": completed[0]["ledger"], "resumed_ledger": completed[1]["ledger"],
        "best_update": [done["best_update"] for done in completed],
        "final_price": [done["final_price"] for done in completed],
        "source": source_after, "diagnostic_seconds": time.monotonic() - began,
        "artifacts": [{"path": str(directory / name), "sha256": _sha(directory / name)}
                      for directory in directories for name in ("latest.pt", "best.pt", "complete.json", "training.jsonl")],
        "processes": [json.loads((output / f"{part}_process.json").read_text()) for part in ("continuous", "pause", "resume")],
    }
    _write(output / "comparison.json", result)
    print(json.dumps(result), flush=True)
    if errors:
        raise RuntimeError("real-GPU interrupted recovery did not match continuous execution")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=3)
    parser.add_argument("--output", required=True)
    parser.add_argument("--part", choices=("continuous", "pause", "resume"))
    args = parser.parse_args()
    if args.part:
        _child(args)
    else:
        _parent(args)


if __name__ == "__main__":
    main()
