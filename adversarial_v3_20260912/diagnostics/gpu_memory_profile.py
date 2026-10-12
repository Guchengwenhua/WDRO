"""Actual degree-50 networks at formal batch sizes; synthetic diagnostic labels."""
from __future__ import annotations

import argparse
import datetime
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from gpu_recovery import FakeGpuOracle, file_hash, preflight, reset_rng

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def memory(gpu_uuid):
    torch.cuda.synchronize()
    process_rows = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader,nounits"], text=True)
    mine = []
    for row in process_rows.strip().splitlines():
        fields = [part.strip() for part in row.split(",")]
        if fields and fields[1] == gpu_uuid:
            mine.append({"nvidia_pid": int(fields[0]), "gpu_uuid": fields[1], "used_mib": int(fields[2]),
                         "matches_visible_python_pid": fields[0] == str(os.getpid())})
    return {"allocated_mib": torch.cuda.memory_allocated() / 2**20,
            "reserved_mib": torch.cuda.memory_reserved() / 2**20,
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
            "nvidia_processes": mine, "visible_python_pid": os.getpid(),
            "nvidia_process_scope": "all processes on the selected physical GPU; host PID may differ from container PID"}


def run(output, gpu):
    output.parent.mkdir(parents=True, exist_ok=True)
    before = preflight(gpu, output.with_name("gpu_memory_profile_preflight.json"))
    gpu_uuid = next(row["uuid"] for row in before["samples"][-1]["gpus"] if row["index"] == gpu)
    from adv_v3 import common, critic, rl
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for memory profile")
    device = torch.device("cuda:0")
    reset_rng(9121750)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    actor, main_critic, _, parent = common.load_parent("wdro_rest", 50, 101, device)
    calibrated = critic.InputCritic(main_critic, calibrated=True).to(device)
    raw = critic.InputCritic(main_critic, calibrated=False).to(device)
    policy = rl.Policy(50).to(device)
    cal_optimizer = torch.optim.Adam(calibrated.parameters(), lr=1e-4)
    main_optimizer = torch.optim.Adam(list(actor.parameters()) + list(main_critic.parameters()), lr=1e-5)
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    pins = common.make_data(50, 128, "uniform", "gpu_degree50_memory_profile", seed=9121750)
    fake = FakeGpuOracle(device)
    phases = []
    phases.append({"phase": "all_networks_resident", **memory(gpu_uuid)})

    calibrated.train()
    target = torch.as_tensor(fake.evaluate(pins)["regret"], dtype=torch.float32, device=device)
    prediction = calibrated(pins)
    loss = (prediction - target).square().mean()
    if not torch.isfinite(loss):
        raise AssertionError("nonfinite calibrated Critic profiling loss")
    cal_optimizer.zero_grad(set_to_none=True)
    loss.backward()
    cal_optimizer.step()
    phases.append({"phase": "critic_calibrated_batch128_adam_step", "batch_size": 128,
                   "loss": float(loss.detach().cpu()), **memory(gpu_uuid)})
    del prediction, loss, target

    actor.train()
    main_critic.train()
    outputs, log_prob = actor(pins[:64], False)
    target = torch.as_tensor(fake.evaluate(pins[:64])["regret"], dtype=torch.float32, device=device)
    baseline = main_critic(pins[:64])
    actor_loss = ((target - baseline).detach() * log_prob).mean()
    critic_loss = (baseline - target).square().mean()
    loss = actor_loss + critic_loss
    if not torch.isfinite(loss):
        raise AssertionError("nonfinite main REINFORCE profiling loss")
    main_optimizer.zero_grad(set_to_none=True)
    loss.backward()
    main_optimizer.step()
    phases.append({"phase": "actor_and_main_critic_batch64_reinforce_adam_step", "batch_size": 64,
                   "actor_loss": float(actor_loss.detach().cpu()), "main_critic_mse": float(critic_loss.detach().cpu()),
                   "sampled_topology_shape": list(outputs.shape), **memory(gpu_uuid)})
    del outputs, log_prob, target, baseline, actor_loss, critic_loss, loss

    policy_detail = rl.online_update(policy, policy_optimizer, fake, pins[:64], 0.2)
    phases.append({"phase": "rl_batch64_reinforce_fake_gpu_reward", "batch_size": 64,
                   "detail": policy_detail, **memory(gpu_uuid)})
    final_memory = memory(gpu_uuid)
    result = {"status": "pass", "kind": "degree50_formal_batch_memory_profile_with_fake_labels",
              "at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "physical_gpu_index": gpu,
              "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"), "pid": os.getpid(),
              "device_name": torch.cuda.get_device_name(0), "torch_version": torch.__version__,
              "scope": "One step per component; real architectures, fake finite labels/rewards; not formal training or quality evaluation",
              "degree": 50, "parent_checkpoint": parent["best"],
              "resident_networks": ["REST Actor", "main Critic", "calibrated InputCritic", "raw InputCritic", "RL Policy"],
              "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
              "cudnn_deterministic": torch.backends.cudnn.deterministic, "cudnn_benchmark": torch.backends.cudnn.benchmark,
              "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
              "real_actor_scoring_calls": 0, "real_rest_calls": 0, "real_geosteiner_calls": 0,
              "fake_gpu_reward_requests": fake.ledger["fake_gpu_reward_requests"],
              "phases": phases, "combined_memory": final_memory, "wall_seconds": time.monotonic() - started,
              "preflight_path": str(output.with_name("gpu_memory_profile_preflight.json")),
              "source_hashes": {str(path): file_hash(path) for path in
                                 [Path(__file__), Path(critic.__file__), Path(rl.__file__), Path(common.__file__)]}}
    common.atomic_json(output, result)
    del actor, main_critic, calibrated, raw, policy, cal_optimizer, main_optimizer, policy_optimizer
    gc.collect()
    torch.cuda.empty_cache()
    print(json.dumps({"status": "pass", "evidence": str(output), "combined_memory": final_memory}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--physical-gpu-index", type=int, default=2)
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/diagnostics/gpu_memory_profile.json")
    args = parser.parse_args()
    try:
        run(args.output, args.physical_gpu_index)
    except Exception as exc:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.with_name("gpu_memory_profile_failure.json").write_text(json.dumps({"status": "fail", "error": repr(exc)}, indent=2) + "\n")
        raise
