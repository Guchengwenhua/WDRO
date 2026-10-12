#!/usr/bin/env python3
"""Compare the semantic training state of continuous and resumed checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.checkpoints import load_checkpoint
from src.io_utils import atomic_write_json
from src.rest import file_sha256


def first_difference(left: Any, right: Any, path: str = "root") -> str | None:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        if left.dtype != right.dtype or left.shape != right.shape or not torch.equal(left.cpu(), right.cpu()):
            return path
        return None
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return None if left.dtype == right.dtype and np.array_equal(left, right) else path
    if type(left) is not type(right):
        return path
    if isinstance(left, dict):
        if set(left) != set(right):
            return path
        for key in sorted(left, key=str):
            difference = first_difference(left[key], right[key], f"{path}.{key}")
            if difference:
                return difference
        return None
    if isinstance(left, (list, tuple)):
        if len(left) != len(right):
            return path
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            difference = first_difference(left_item, right_item, f"{path}[{index}]")
            if difference:
                return difference
        return None
    return None if left == right else path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resumed", type=Path, required=True)
    parser.add_argument("--continuous", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    resumed = load_checkpoint(args.resumed, map_location="cpu")
    continuous = load_checkpoint(args.continuous, map_location="cpu")
    fields = [
        "run_identity",
        "update",
        "actor_state_dict",
        "critic_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "rng_state",
        "best",
        "call_ledger",
        "training_stream_hash",
        "initialization",
        "inverse_length_audit",
    ]
    differences = {}
    for field in fields:
        difference = first_difference(resumed[field], continuous[field], field)
        if difference:
            differences[field] = difference
    result = {
        "schema_version": 1,
        "kind": "checkpoint_resume_determinism_audit",
        "captured_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "resumed_checkpoint": str(args.resumed.resolve()),
        "resumed_checkpoint_sha256": file_sha256(args.resumed),
        "continuous_checkpoint": str(args.continuous.resolve()),
        "continuous_checkpoint_sha256": file_sha256(args.continuous),
        "compared_fields": fields,
        "differences": differences,
        "status": "pass" if not differences else "fail",
        "exact_semantic_state_match": not differences,
    }
    atomic_write_json(args.output, result, mode=0o444)
    print(json.dumps(result, sort_keys=True))
    return 0 if not differences else 2


if __name__ == "__main__":
    raise SystemExit(main())
