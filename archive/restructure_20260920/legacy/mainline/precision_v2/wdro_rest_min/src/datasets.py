"""Deterministic synthetic datasets, IDs, hashes, and immutable NPZ storage."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .geometry import (
    MINIMUM_L1_SPACING,
    NORMALIZED_LOWER,
    NORMALIZED_UPPER,
    ZERO_SPAN_EPSILON,
    has_exact_duplicate,
    has_spacing_violation,
)


def namespaced_seed(
    protocol_id: str,
    base_seed: int,
    split: str,
    distribution: str,
    degree: int,
    attempt: int = 0,
) -> int:
    identity = f"{protocol_id}\0{base_seed}\0{split}\0{distribution}\0{degree}\0{attempt}"
    return int.from_bytes(hashlib.sha256(identity.encode("utf-8")).digest()[:8], "little")


def generate_terminals(
    n_nets: int,
    degree: int,
    seed: int,
    distribution: str,
    parameters: dict[str, float | int],
) -> np.ndarray:
    """Generate the four frozen distributions in [0,1]^2."""

    if n_nets <= 0 or degree < 2:
        raise ValueError("n_nets must be positive and degree must be at least two")
    rng = np.random.default_rng(seed)
    nonuniform = float(parameters.get("nonuniform", 0.0))
    clusters = int(parameters.get("clusters", 1))
    sigma = float(parameters.get("sigma", 0.0))
    if not 0.0 <= nonuniform <= 1.0:
        raise ValueError("nonuniform must lie in [0,1]")

    if distribution == "uniform":
        values = rng.random((n_nets, degree, 2))
    elif distribution == "mix_gauss":
        if clusters <= 0 or sigma <= 0:
            raise ValueError("mix_gauss requires positive clusters and sigma")
        centers = rng.random((clusters, 2))
        values = rng.random((n_nets, degree, 2))
        mask = rng.random((n_nets, degree)) < nonuniform
        center_ids = rng.integers(0, clusters, size=(n_nets, degree))
        gaussian = centers[center_ids] + rng.normal(0.0, sigma, size=(n_nets, degree, 2))
        values[mask] = gaussian[mask]
    elif distribution == "edge_bias":
        values = rng.random((n_nets, degree, 2))
        values **= 1.0 + nonuniform * 3.0
        flip = rng.random((n_nets, degree, 2)) > 0.5
        values[flip] = 1.0 - values[flip]
    elif distribution == "grid_cluster":
        if clusters < 2 or sigma <= 0:
            raise ValueError("grid_cluster requires clusters >= 2 and positive sigma")
        values = rng.random((n_nets, degree, 2))
        aligned = rng.random((n_nets, degree)) < nonuniform
        rows = np.linspace(0.1, 0.9, clusters)
        row_ids = rng.integers(0, clusters, size=(n_nets, degree))
        aligned_x = rng.random((n_nets, degree))
        aligned_y = rows[row_ids] + rng.normal(0.0, sigma * 0.3, size=(n_nets, degree))
        values[:, :, 0][aligned] = aligned_x[aligned]
        values[:, :, 1][aligned] = aligned_y[aligned]
    else:
        raise ValueError(f"unknown frozen distribution: {distribution}")
    return np.round(np.clip(values, 0.0, 1.0), 8)


def normalize_fixed_degree_batch(
    raw_pins: np.ndarray,
    *,
    minimum_l1_spacing: float = MINIMUM_L1_SPACING,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str | None]]:
    """Vectorized bbox transform plus per-net legality checks."""

    raw = np.asarray(raw_pins, dtype=np.float64)
    if raw.ndim != 3 or raw.shape[-1] != 2 or raw.shape[1] < 2:
        raise ValueError("raw_pins must have shape (count,degree,2), degree >= 2")
    count = len(raw)
    finite = np.isfinite(raw).all(axis=(1, 2))
    safe = np.where(np.isfinite(raw), raw, 0.0)
    offsets = safe.min(axis=1)
    spans = safe.max(axis=1) - offsets
    longest = spans.max(axis=1)
    valid_span = longest > ZERO_SPAN_EPSILON
    scales = np.zeros(count, dtype=np.float64)
    scales[valid_span] = (NORMALIZED_UPPER - NORMALIZED_LOWER) / longest[valid_span]
    normalized = (safe - offsets[:, None, :]) * scales[:, None, None] + NORMALIZED_LOWER
    normalized = np.clip(normalized, NORMALIZED_LOWER, NORMALIZED_UPPER)

    valid = finite & valid_span
    reasons: list[str | None] = [None] * count
    for index in range(count):
        if not finite[index]:
            reasons[index] = "nonfinite_coordinate"
            continue
        if not valid_span[index]:
            reasons[index] = "zero_bbox_span"
            continue
        if has_exact_duplicate(normalized[index]):
            valid[index] = False
            reasons[index] = "degree_changed_by_deduplication"
            continue
        if has_spacing_violation(normalized[index], minimum_l1_spacing):
            valid[index] = False
            reasons[index] = "minimum_spacing_violation"
    return normalized, offsets, scales, valid, reasons


def make_net_id(split: str, distribution: str, degree: int, base_seed: int, index: int) -> str:
    return f"{split}:{distribution}:d{degree}:s{base_seed}:n{index:06d}"


def array_bundle_sha256(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(b"\0")
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_synthetic_dataset(
    *,
    protocol_id: str,
    split: str,
    distribution: str,
    degree: int,
    base_seed: int,
    count: int,
    parameters: dict[str, float | int],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    accepted_pins: list[np.ndarray] = []
    accepted_offsets: list[np.ndarray] = []
    accepted_scales: list[np.ndarray] = []
    rejection_counts: Counter[str] = Counter()
    attempt = 0
    while sum(len(batch) for batch in accepted_pins) < count:
        already = sum(len(batch) for batch in accepted_pins)
        needed = count - already
        generated_count = max(needed, min(128, count))
        seed = namespaced_seed(protocol_id, base_seed, split, distribution, degree, attempt)
        raw = generate_terminals(generated_count, degree, seed, distribution, parameters)
        normalized, offsets, scales, valid, reasons = normalize_fixed_degree_batch(raw)
        for reason in reasons:
            if reason is not None:
                rejection_counts[reason] += 1
        if valid.any():
            accepted_pins.append(normalized[valid][:needed])
            accepted_offsets.append(offsets[valid][:needed])
            accepted_scales.append(scales[valid][:needed])
        attempt += 1
        if attempt > 100:
            raise RuntimeError("unable to generate enough legal nets after 100 deterministic attempts")

    pins = np.concatenate(accepted_pins, axis=0)[:count].astype(np.float32)
    offsets = np.concatenate(accepted_offsets, axis=0)[:count].astype(np.float64)
    scales = np.concatenate(accepted_scales, axis=0)[:count].astype(np.float64)
    ids = np.asarray(
        [make_net_id(split, distribution, degree, base_seed, index).encode("utf-8") for index in range(count)],
        dtype="S80",
    )
    arrays = {
        "net_ids": ids,
        "offsets": offsets,
        "pins": pins,
        "scales": scales,
    }
    metadata = {
        "schema_version": 1,
        "kind": "synthetic_fixed_split",
        "protocol_id": protocol_id,
        "split": split,
        "distribution": distribution,
        "parameters": parameters,
        "degree": degree,
        "base_seed": base_seed,
        "count": count,
        "attempt_streams_used": attempt,
        "rejections": dict(sorted(rejection_counts.items())),
        "array_bundle_sha256": array_bundle_sha256(arrays),
    }
    return arrays, metadata


def save_dataset(path: str | Path, arrays: dict[str, np.ndarray], metadata: dict[str, Any]) -> dict[str, Any]:
    destination = Path(path)
    meta_path = destination.with_suffix(".meta.json")
    if destination.exists() or meta_path.exists():
        raise FileExistsError(f"refusing to overwrite frozen dataset {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".npz", delete=False) as temporary:
        temporary_path = Path(temporary.name)
        np.savez_compressed(temporary, **arrays)
    try:
        os.replace(temporary_path, destination)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    final_metadata = dict(metadata)
    final_metadata["array_bundle_sha256"] = array_bundle_sha256(arrays)
    final_metadata["npz_sha256"] = file_sha256(destination)
    meta_path.write_text(json.dumps(final_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    destination.chmod(0o444)
    meta_path.chmod(0o444)
    return final_metadata


def load_dataset(path: str | Path, *, verify: bool = True) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    source = Path(path)
    metadata = json.loads(source.with_suffix(".meta.json").read_text(encoding="utf-8"))
    with np.load(source, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    if verify:
        if file_sha256(source) != metadata["npz_sha256"]:
            raise ValueError(f"NPZ file hash mismatch: {source}")
        if array_bundle_sha256(arrays) != metadata["array_bundle_sha256"]:
            raise ValueError(f"array bundle hash mismatch: {source}")
    return arrays, metadata
