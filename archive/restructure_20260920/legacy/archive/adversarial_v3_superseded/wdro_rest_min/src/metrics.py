"""Relative-regret metrics and paired uncertainty estimates."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def relative_regret(rest_lengths: np.ndarray, optimal_lengths: np.ndarray) -> np.ndarray:
    rest = np.asarray(rest_lengths, dtype=np.float64)
    optimal = np.asarray(optimal_lengths, dtype=np.float64)
    if rest.shape != optimal.shape:
        raise ValueError("REST and optimal lengths must have identical shapes")
    if not (np.isfinite(rest).all() and np.isfinite(optimal).all()):
        raise ValueError("lengths must be finite")
    if np.any(optimal <= 0):
        raise ValueError("all optimal lengths must be positive")
    return (rest - optimal) / optimal


def summarize_regret(regrets: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(regrets, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("regrets must be a non-empty vector")
    if not np.isfinite(values).all():
        raise ValueError("regrets must be finite")
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "p95": float(np.quantile(values, 0.95)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
    }


def summarize_groups(regrets: np.ndarray, groups: Iterable[str]) -> dict[str, object]:
    values = np.asarray(regrets, dtype=np.float64)
    labels = np.asarray(list(groups), dtype=str)
    if values.ndim != 1 or labels.shape != values.shape:
        raise ValueError("regrets and groups must be equal-length vectors")
    per_group = {}
    for group in sorted(np.unique(labels)):
        per_group[str(group)] = summarize_regret(values[labels == group])
    worst_group = max(per_group, key=lambda group: per_group[group]["mean"])
    return {
        "overall": summarize_regret(values),
        "groups": per_group,
        "worst_group": worst_group,
        "worst_group_mean": per_group[worst_group]["mean"],
    }


def paired_bootstrap_mean_difference(
    candidate_regrets: np.ndarray,
    baseline_regrets: np.ndarray,
    *,
    confidence_level: float = 0.95,
    samples: int = 10000,
    seed: int = 0,
    chunk_size: int = 256,
) -> dict[str, float | int]:
    candidate = np.asarray(candidate_regrets, dtype=np.float64)
    baseline = np.asarray(baseline_regrets, dtype=np.float64)
    if candidate.ndim != 1 or candidate.shape != baseline.shape or len(candidate) == 0:
        raise ValueError("paired regrets must be non-empty equal-length vectors")
    if not (np.isfinite(candidate).all() and np.isfinite(baseline).all()):
        raise ValueError("paired regrets must be finite")
    if not 0 < confidence_level < 1 or samples <= 0:
        raise ValueError("invalid bootstrap settings")
    difference = candidate - baseline
    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(samples, dtype=np.float64)
    position = 0
    while position < samples:
        count = min(chunk_size, samples - position)
        indices = rng.integers(0, len(difference), size=(count, len(difference)))
        bootstrap_means[position : position + count] = difference[indices].mean(axis=1)
        position += count
    alpha = (1.0 - confidence_level) / 2.0
    return {
        "count": int(len(difference)),
        "bootstrap_samples": int(samples),
        "mean_difference": float(difference.mean()),
        "ci_low": float(np.quantile(bootstrap_means, alpha)),
        "ci_high": float(np.quantile(bootstrap_means, 1.0 - alpha)),
        "confidence_level": float(confidence_level),
    }

