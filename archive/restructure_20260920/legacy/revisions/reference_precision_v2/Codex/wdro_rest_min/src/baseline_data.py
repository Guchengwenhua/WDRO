"""Stage-3 sampling; reuse the frozen virtual pool and shared preprocessing."""
from __future__ import annotations

import numpy as np

from .datasets import array_bundle_sha256, build_synthetic_dataset
from .training_data import TrainingBatch, materialize_training_batch

GROUPS = ("uniform", "mix_gauss", "edge_bias", "grid_cluster")


def balanced_training_batch(*, protocol, degree, training_seed, update, batch_size):
    if batch_size % 4 or update <= 0:
        raise ValueError("four equal groups require a positive update and batch divisible by four")
    if degree not in protocol["scope"]["degrees"] or training_seed not in protocol["data"]["seeds"]["training"]:
        raise ValueError("training identity outside frozen protocol")
    groups = []
    for name in GROUPS:
        # Generate the same 256-net per-group slice as other methods, select its
        # first 64. Generating 64 directly changes RNG ordering for some shifts.
        arrays, _ = build_synthetic_dataset(
            protocol_id=protocol["protocol_id"], split=f"train_u{update:06d}",
            distribution=name, degree=degree, base_seed=training_seed,
            count=batch_size, parameters=protocol["data"]["distributions"][name])
        groups.append({key: value[:batch_size // 4] for key, value in arrays.items()})
    arrays = {key: np.concatenate([g[key] for g in groups]) for key in groups[0]}
    return TrainingBatch(
        pins=arrays["pins"], net_ids=arrays["net_ids"], offsets=arrays["offsets"],
        scales=arrays["scales"], metadata={"update": update, "groups": GROUPS,
        "group_counts": [batch_size // 4] * 4}, bundle_sha256=array_bundle_sha256(arrays))


def method_training_batch(method, **kwargs):
    if method in ("mixed_erm", "group_dro"):
        return balanced_training_batch(**kwargs)
    if method == "hard_mining":
        return materialize_training_batch(**kwargs)
    raise ValueError(f"unsupported baseline {method}")


def update_group_weights(log_weights, group_losses, eta):
    losses = np.asarray(group_losses, dtype=np.float64)
    old = np.asarray(log_weights, dtype=np.float64)
    if old.shape != (4,) or losses.shape != (4,) or not np.isfinite(losses).all() or not np.isfinite(old).all() or eta <= 0:
        raise ValueError("invalid GroupDRO state or losses")
    new = old + eta * losses
    maximum = new.max()
    new -= maximum + np.log(np.exp(new - maximum).sum())
    return new, np.exp(new)
