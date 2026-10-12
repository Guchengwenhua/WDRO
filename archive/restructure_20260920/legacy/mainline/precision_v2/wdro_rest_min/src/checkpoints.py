"""Atomic checkpoint IO and complete random-number-generator state capture."""

from __future__ import annotations

import os
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": [],
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    if set(state) != required:
        raise ValueError(f"checkpoint RNG fields differ: {sorted(state)}")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # ``torch.load(..., map_location=cuda)`` also moves serialized RNG byte
    # tensors.  Both CPU and CUDA generator setters require CPU ByteTensors.
    torch.set_rng_state(state["torch_cpu"].detach().cpu())
    if torch.cuda.is_available():
        cuda_states = state["torch_cuda"]
        if len(cuda_states) != torch.cuda.device_count():
            raise ValueError("checkpoint CUDA RNG device count differs from the active runtime")
        torch.cuda.set_rng_state_all([item.detach().cpu() for item in cuda_states])


def atomic_torch_save(value: Any, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".partial", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(value, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_checkpoint(path: str | Path, *, map_location: torch.device | str) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    checkpoint = torch.load(source, map_location=map_location, weights_only=False)
    required = {
        "schema_version",
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
    }
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(f"checkpoint is missing fields: {sorted(missing)}")
    if checkpoint["schema_version"] != 1:
        raise ValueError("unsupported checkpoint schema")
    return checkpoint
