#!/usr/bin/env python3
"""CPU-only smoke test for the legacy REST model, checkpoint, and evaluator."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
REST_ROOT = WORKSPACE_ROOT / "DRO" / "WDRO-REST"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--degree", type=int, default=10)
    args = parser.parse_args()

    if args.degree not in {10, 20, 30, 40, 50}:
        parser.error("degree must be one of 10, 20, 30, 40, 50")

    sys.path.insert(0, str(REST_ROOT))
    previous_cwd = Path.cwd()
    try:
        os.chdir(REST_ROOT)
        from models.actor_critic import Actor
        from utils.rsmt_utils import Evaluator

        checkpoint_path = REST_ROOT / "save" / "DAC21" / f"rsmt{args.degree}b.pt"
        # These repository-owned legacy checkpoints contain NumPy scalar metadata.
        # PyTorch 2.6 therefore cannot load them with weights_only=True.
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        actor = Actor(args.degree, torch.device("cpu"))
        actor.load_state_dict(checkpoint["actor_state_dict"], strict=True)
        actor.eval()

        rng = np.random.default_rng(20260904)
        pins = rng.random((1, args.degree, 2), dtype=np.float32)
        with torch.no_grad():
            topology, _ = actor(pins, deterministic=True)

        topology_np = topology.cpu().numpy()
        expected_shape = (1, 2 * (args.degree - 1))
        if topology_np.shape != expected_shape:
            raise RuntimeError(f"unexpected topology shape {topology_np.shape}; expected {expected_shape}")
        if topology_np.min() < 0 or topology_np.max() >= args.degree:
            raise RuntimeError("actor emitted an invalid terminal index")

        evaluator = Evaluator()
        rest_length = float(evaluator.eval_batch(pins, topology_np, args.degree)[0])
        optimal_length = float(evaluator.gst_rsmt(pins[0])[0])
        if not (math.isfinite(rest_length) and math.isfinite(optimal_length) and optimal_length > 0):
            raise RuntimeError("evaluator returned a non-finite or non-positive length")

        result = {
            "status": "pass",
            "device": "cpu",
            "degree": args.degree,
            "checkpoint": str(checkpoint_path.relative_to(WORKSPACE_ROOT)),
            "checkpoint_batch_idx": int(checkpoint["batch_idx"]),
            "topology_shape": list(topology_np.shape),
            "rest_length": rest_length,
            "geosteiner_length": optimal_length,
        }
        print(json.dumps(result, sort_keys=True))
        return 0
    finally:
        os.chdir(previous_cwd)


if __name__ == "__main__":
    raise SystemExit(main())

