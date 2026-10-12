"""Permutation-invariant rectilinear costs between equal-degree pin sets."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def _validate_pair(x: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left = np.asarray(x, dtype=np.float64)
    right = np.asarray(z, dtype=np.float64)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1:] != (2,) or right.shape[1:] != (2,):
        raise ValueError(f"pin sets must have shapes (degree,2), got {left.shape} and {right.shape}")
    if left.shape != right.shape:
        raise ValueError(f"pin sets must have the same degree, got {left.shape} and {right.shape}")
    if len(left) == 0:
        raise ValueError("pin sets must be non-empty")
    if not (np.isfinite(left).all() and np.isfinite(right).all()):
        raise ValueError("pin sets must contain only finite coordinates")
    return left, right


def hungarian_l1_mean(x: np.ndarray, z: np.ndarray) -> float:
    left, right = _validate_pair(x, z)
    pairwise_l1 = np.abs(left[:, None, :] - right[None, :, :]).sum(axis=-1)
    rows, columns = linear_sum_assignment(pairwise_l1)
    return float(pairwise_l1[rows, columns].mean())


def symmetric_chamfer_l1(x: np.ndarray, z: np.ndarray) -> float:
    left, right = _validate_pair(x, z)
    pairwise_l1 = np.abs(left[:, None, :] - right[None, :, :]).sum(axis=-1)
    return float(0.5 * (pairwise_l1.min(axis=1).mean() + pairwise_l1.min(axis=0).mean()))


def batch_cost(x: np.ndarray, z: np.ndarray, *, kind: str = "hungarian_l1_mean") -> np.ndarray:
    left = np.asarray(x)
    right = np.asarray(z)
    if left.ndim != 3 or right.ndim != 3 or left.shape != right.shape:
        raise ValueError("batches must have identical shapes (batch,degree,2)")
    functions = {
        "hungarian_l1_mean": hungarian_l1_mean,
        "symmetric_chamfer_l1": symmetric_chamfer_l1,
    }
    if kind not in functions:
        raise ValueError(f"unknown cost kind: {kind}")
    function = functions[kind]
    return np.asarray([function(a, b) for a, b in zip(left, right, strict=True)], dtype=np.float64)

