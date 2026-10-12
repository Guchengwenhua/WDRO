"""Bounded GPU recovery diagnostic; fake rewards, four real GeoSteiner solves.

Run with CUDA_VISIBLE_DEVICES=2 through this version's run_env.sh.  The diagnostic
never updates a smoke/parent checkpoint or starts a formal experiment task.
"""
from __future__ import annotations

import argparse
import datetime
import gc
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def preflight(physical_gpu, output):
    snapshots = []
    for index in range(3):
        raw = subprocess.check_output([
            "nvidia-smi", "--query-gpu=index,uuid,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits"], text=True)
        rows = []
        for line in raw.strip().splitlines():
            gpu, uuid, total, used, free, utilization = [x.strip() for x in line.split(",")]
            rows.append({"index": int(gpu), "uuid": uuid, "total_mib": int(total),
                         "used_mib": int(used), "free_mib": int(free), "utilization": int(utilization)})
        snapshots.append({"at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "gpus": rows})
        if index != 2:
            time.sleep(1)
    processes = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader,nounits"], text=True)
    evidence = {"physical_gpu_index": physical_gpu, "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "samples": snapshots, "gpu_processes": processes,
                "disk": subprocess.check_output(["df", "-h", str(output.parent)], text=True)}
    output.write_text(json.dumps(evidence, indent=2) + "\n")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical_gpu):
        raise RuntimeError("CUDA_VISIBLE_DEVICES must match the diagnostic's physical GPU")
    selected = [next(g for g in sample["gpus"] if g["index"] == physical_gpu) for sample in snapshots]
    if any(g["free_mib"] < 6144 or g["utilization"] >= 85 for g in selected):
        raise RuntimeError("insufficient currently available GPU capacity; no work was launched")
    return evidence


def assert_equal(left, right, location="root"):
    """Recursively demand bitwise equality, including optimizer and RNG tensors."""
    if torch.is_tensor(left):
        if not torch.is_tensor(right) or not torch.equal(left.cpu(), right.cpu()):
            raise AssertionError(f"tensor mismatch: {location}")
    elif isinstance(left, np.ndarray):
        if not np.array_equal(left, right):
            raise AssertionError(f"array mismatch: {location}")
    elif isinstance(left, dict):
        if left.keys() != right.keys():
            raise AssertionError(f"dictionary keys differ: {location}")
        for key in left:
            assert_equal(left[key], right[key], f"{location}.{key}")
    elif isinstance(left, (list, tuple)):
        if len(left) != len(right):
            raise AssertionError(f"sequence lengths differ: {location}")
        for index, (first, second) in enumerate(zip(left, right)):
            assert_equal(first, second, f"{location}[{index}]")
    elif left != right:
        raise AssertionError(f"value mismatch: {location}: {left!r} vs {right!r}")


class FakeGpuOracle:
    """Deterministic GPU arithmetic rewards, explicitly not REST regret."""
    def __init__(self, device):
        self.device = device
        self.actor = nn.Identity().to(device)
        self.ledger = {"fake_gpu_reward_requests": 0}

    def evaluate(self, pins):
        points = torch.as_tensor(pins, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            reward = points.square().mean(dim=(1, 2)) + 0.15 * points[:, :, 0].mean(dim=1)
        self.ledger["fake_gpu_reward_requests"] += len(points)
        return {"regret": reward.cpu().numpy().astype(np.float64)}


def reset_rng(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run(output, physical_gpu):
    output.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S")
    run_dir = output.parent / "gpu_recovery_runs" / stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    preflight_path = run_dir / "gpu_preflight.json"
    before = preflight(physical_gpu, preflight_path)
    from adv_v3 import common, critic, rl
    from src.geosteiner import GeoSteinerEvaluator, ParallelGeoSteinerEvaluator
    from src.rest import build_critic

    started = time.monotonic()
    if not torch.cuda.is_available():
        raise RuntimeError("GPU diagnostic requires CUDA")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda:0")
    source_paths = [Path(__file__), Path(common.__file__), Path(critic.__file__), Path(rl.__file__)]
    source_hashes = {str(path): file_hash(path) for path in source_paths}
    smoke = common.EXP / "smoke/wdro_rest_d10_s101"
    critic_source = smoke / "critic/best.pt"
    rl_source = smoke / "rl/best.pt"
    critic_summary = json.loads((smoke / "critic/summary.json").read_text())
    rl_summary = json.loads((smoke / "rl/summary.json").read_text())
    smoke_artifacts = json.loads((smoke / "complete.json").read_text())["artifacts"]
    for path in (critic_source, rl_source):
        expected = next(record["sha256"] for record in smoke_artifacts if record["path"] == str(path))
        if file_hash(path) != expected:
            raise RuntimeError("smoke source checkpoint hash changed")
    source = torch.load(critic_source, map_location="cpu", weights_only=False)
    parent = build_critic(10, device)
    parent.load_state_dict({name.removeprefix("critic."): value
                            for name, value in source["model_state_dict"].items()}, strict=True)
    parent.eval()
    parent_before = {name: value.detach().cpu().clone() for name, value in parent.state_dict().items()}
    critic_config = dict(critic_summary["identity"]["settings"])
    critic_config.update(train_count=32, validation_count=16, epochs=2, batch_size=8,
                         namespace="gpu_recovery_critic")
    critic_seed = int(critic_summary["identity"]["seed"])
    full_oracle, split_oracle = FakeGpuOracle(device), FakeGpuOracle(device)
    reset_rng(91217)
    critic_full, full_summary = critic.train_surrogate(
        full_oracle, parent, 10, critic_seed, run_dir / "critic_continuous", critic_config)
    original_save = common.atomic_torch

    def stop_at_first_epoch(path, state):
        original_save(path, state)
        if Path(path).name == "latest.pt" and state.get("epoch") == 1:
            raise InterruptedError("diagnostic interruption after epoch one checkpoint")

    reset_rng(91217)
    interrupted = False
    with patch.object(common, "atomic_torch", stop_at_first_epoch):
        try:
            critic.train_surrogate(split_oracle, parent, 10, critic_seed,
                                   run_dir / "critic_resumed", critic_config)
        except InterruptedError:
            interrupted = True
    if not interrupted:
        raise AssertionError("Critic interruption point was not reached")
    critic_resumed, resumed_summary = critic.train_surrogate(
        split_oracle, parent, 10, critic_seed, run_dir / "critic_resumed", critic_config)
    critic_full_state = torch.load(run_dir / "critic_continuous/latest.pt", map_location="cpu", weights_only=False)
    critic_split_state = torch.load(run_dir / "critic_resumed/latest.pt", map_location="cpu", weights_only=False)
    critic_checks = ["model_state_dict", "optimizer_state_dict", "rng_state", "history", "best_mse", "epoch"]
    for name in critic_checks:
        assert_equal(critic_full_state[name], critic_split_state[name], f"critic.{name}")
    assert_equal(critic_full.state_dict(), critic_resumed.state_dict(), "critic.best_returned_model")
    assert_equal(parent_before, parent.state_dict(), "critic.parent_unchanged")
    assert_equal(full_oracle.ledger, split_oracle.ledger, "critic.no_repeated_labels")
    if full_oracle.ledger["fake_gpu_reward_requests"] != 48:
        raise AssertionError("wrong Critic fake label budget")
    critic_result = {"status": "pass", "degree": 10, "config": critic_config, "seed": critic_seed,
                     "completed_epochs": 2, "interrupt_after_epoch": 1, "strict_equal": critic_checks,
                     "best_returned_model_exact": True, "parent_unchanged": True,
                     "continuous_requests": full_oracle.ledger, "resumed_requests": split_oracle.ledger,
                     "best_epoch": full_summary["best_epoch"], "validation_mse": full_summary["validation"]["mse"]}

    rl_config = dict(rl_summary["identity"]["config"])
    rl_config.update(updates=4, batch_size=8, validation_interval=1, validation_count=8)
    rl_seed, price = int(rl_summary["identity"]["seed"]), float(rl_summary["identity"]["price"])
    full_rl_oracle, split_rl_oracle = FakeGpuOracle(device), FakeGpuOracle(device)
    policy_full, rl_full_summary = rl.train_policy(
        full_rl_oracle, 10, rl_seed, price, run_dir / "rl_continuous", rl_config)
    _, partial = rl.train_policy(split_rl_oracle, 10, rl_seed, price,
                                 run_dir / "rl_resumed", {**rl_config, "stop_after": 2})
    if partial["update"] != 2 or partial["status"] != "partial":
        raise AssertionError("RL two-update checkpoint was not reached")
    policy_resumed, rl_resumed_summary = rl.train_policy(
        split_rl_oracle, 10, rl_seed, price, run_dir / "rl_resumed", rl_config)
    rl_full_state = torch.load(run_dir / "rl_continuous/latest.pt", map_location="cpu", weights_only=False)
    rl_split_state = torch.load(run_dir / "rl_resumed/latest.pt", map_location="cpu", weights_only=False)
    rl_checks = ["policy_state_dict", "optimizer_state_dict", "rng_state", "update", "best_update",
                 "best_f", "counts", "validation_history", "training_records", "validation_centers_sha256"]
    for name in rl_checks:
        assert_equal(rl_full_state[name], rl_split_state[name], f"rl.{name}")
    assert_equal(policy_full.state_dict(), policy_resumed.state_dict(), "rl.best_returned_model")
    assert_equal(full_rl_oracle.ledger, split_rl_oracle.ledger, "rl.no_repeated_reward_requests")
    rl_result = {"status": "pass", "degree": 10, "config": rl_config, "seed": rl_seed,
                 "price": price, "continuous_updates": 4, "resumed_updates": [2, 2],
                 "strict_equal": rl_checks, "best_returned_model_exact": True,
                 "continuous_requests": full_rl_oracle.ledger, "resumed_requests": split_rl_oracle.ledger,
                 "best_update": rl_full_summary["best_update"], "counts": rl_full_summary["counts"]}

    pins = common.make_data(10, 2, "uniform", "gpu_recovery_reference", seed=91217)
    serial = GeoSteinerEvaluator()
    serial_lengths = serial.lengths(pins)
    with ParallelGeoSteinerEvaluator(2) as parallel:
        parallel_lengths = parallel.lengths(pins)
    if not np.array_equal(serial_lengths, parallel_lengths):
        raise AssertionError("serial and Parallel2 GeoSteiner differ")
    common.atomic_npz(run_dir / "reference_serial_parallel2.npz", pins=pins,
                      serial=serial_lengths, parallel2=parallel_lengths)
    geo_result = {"status": "pass", "degree": 10, "nets": 2, "float32_inputs": str(pins.dtype),
                  "serial_lengths": serial_lengths.tolist(), "parallel2_lengths": parallel_lengths.tolist(),
                  "exactly_equal": True, "real_geosteiner_net_solves": 4, "real_actor_calls": 0,
                  "real_rest_calls": 0}
    torch.cuda.synchronize()
    peak = {"allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "reserved_mib": torch.cuda.max_memory_reserved() / 2**20}
    dependencies_unchanged = source_hashes == {str(path): file_hash(path) for path in source_paths}
    report = {"status": "pass", "kind": "gpu_exact_resume_and_reference_parallelism_diagnostic",
              "at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "physical_gpu_index": physical_gpu,
              "device_name": torch.cuda.get_device_name(0), "torch_version": torch.__version__,
              "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
              "cudnn_deterministic": torch.backends.cudnn.deterministic, "cudnn_benchmark": torch.backends.cudnn.benchmark,
              "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
              "scope": "Fake GPU arithmetic rewards; no attacker-quality claim and no formal training task",
              "preflight_path": str(preflight_path), "source_hashes": source_hashes,
              "sources_unchanged_during_run": dependencies_unchanged,
              "smoke_critic_checkpoint": {"path": str(critic_source), "sha256": file_hash(critic_source)},
              "smoke_rl_checkpoint": {"path": str(rl_source), "sha256": file_hash(rl_source)},
              "critic_initialization": "calibrated smoke backbone; independent copy",
              "rl_initialization": "fresh deterministic policy using completed smoke base configuration",
              "critic": critic_result, "rl": rl_result, "reference": geo_result,
              "peak_gpu_memory": peak, "wall_seconds": time.monotonic() - started, "run_dir": str(run_dir),
              "artifacts": [{"path": str(path), "sha256": file_hash(path)}
                            for path in sorted(run_dir.rglob("*")) if path.is_file()]}
    common.atomic_json(output, report)
    del critic_full, critic_resumed, parent, policy_full, policy_resumed
    gc.collect()
    torch.cuda.empty_cache()
    print(json.dumps({"status": report["status"], "evidence": str(output), "peak_gpu_memory": peak,
                      "sources_unchanged_during_run": dependencies_unchanged}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--physical-gpu-index", type=int, default=2)
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/diagnostics/gpu_recovery.json")
    args = parser.parse_args()
    try:
        run(args.output, args.physical_gpu_index)
    except Exception as exc:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        failure = {"status": "fail", "error": repr(exc), "script_sha256": file_hash(__file__)}
        args.output.with_name("gpu_recovery_failure.json").write_text(json.dumps(failure, indent=2) + "\n")
        raise
