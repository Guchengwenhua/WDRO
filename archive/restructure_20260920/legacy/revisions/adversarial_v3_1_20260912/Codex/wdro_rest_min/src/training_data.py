"""Stateless, immutable training-pool materialization for deterministic resume."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from .datasets import array_bundle_sha256, build_synthetic_dataset


@dataclass(frozen=True)
class TrainingBatch:
    pins: np.ndarray
    net_ids: np.ndarray
    offsets: np.ndarray
    scales: np.ndarray
    metadata: dict[str, Any]
    bundle_sha256: str


def materialize_training_batch(
    *,
    protocol: dict[str, Any],
    degree: int,
    training_seed: int,
    update: int,
    batch_size: int,
    distribution: str = "uniform",
) -> TrainingBatch:
    """Materialize one disjoint slice of the frozen virtual training pool."""

    if update <= 0:
        raise ValueError("training update is one-indexed")
    if training_seed not in protocol["data"]["seeds"]["training"]:
        raise ValueError("training seed is outside protocol_v1")
    if degree not in protocol["scope"]["degrees"]:
        raise ValueError("degree is outside protocol_v1")
    if distribution not in protocol["method_contracts"]["erm"]["training_groups"]:
        raise ValueError("ERM may only materialize the nominal training group")
    split = f"train_u{update:06d}"
    arrays, metadata = build_synthetic_dataset(
        protocol_id=protocol["protocol_id"],
        split=split,
        distribution=distribution,
        degree=degree,
        base_seed=training_seed,
        count=batch_size,
        parameters=protocol["data"]["distributions"][distribution],
    )
    metadata = dict(metadata)
    metadata.update(
        {
            "pool_kind": "stateless_immutable_per_update",
            "update": update,
            "global_index_start": (update - 1) * batch_size,
            "global_index_stop_exclusive": update * batch_size,
        }
    )
    return TrainingBatch(
        pins=arrays["pins"],
        net_ids=arrays["net_ids"],
        offsets=arrays["offsets"],
        scales=arrays["scales"],
        metadata=metadata,
        bundle_sha256=array_bundle_sha256(arrays),
    )


def extend_stream_hash(previous_hex: str, batch_bundle_sha256: str) -> str:
    if len(previous_hex) != 64 or len(batch_bundle_sha256) != 64:
        raise ValueError("stream hash inputs must be SHA-256 hex digests")
    digest = hashlib.sha256()
    digest.update(bytes.fromhex(previous_hex))
    digest.update(bytes.fromhex(batch_bundle_sha256))
    return digest.hexdigest()


EMPTY_STREAM_SHA256 = hashlib.sha256(b"").hexdigest()
