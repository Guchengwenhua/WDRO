"""Shared net normalization, inverse transform, legality, and identity rules."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable

import numpy as np


NORMALIZED_LOWER = 0.1
NORMALIZED_UPPER = 0.9
ZERO_SPAN_EPSILON = 1e-12
MINIMUM_L1_SPACING = 1e-6


class InvalidNetError(ValueError):
    """A net violates the frozen data contract."""

    def __init__(self, reason: str, detail: str = "") -> None:
        self.reason = reason
        self.detail = detail
        message = reason if not detail else f"{reason}: {detail}"
        super().__init__(message)


@dataclass(frozen=True)
class NetTransform:
    offset_xy: np.ndarray
    scale: float
    original_degree: int
    unique_degree: int
    duplicate_count: int


def _as_pin_array(pins: np.ndarray | Iterable[Iterable[float]]) -> np.ndarray:
    array = np.asarray(pins, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise InvalidNetError("invalid_shape", f"expected (degree,2), got {array.shape}")
    if not np.isfinite(array).all():
        raise InvalidNetError("nonfinite_coordinate")
    return np.ascontiguousarray(array)


def stable_deduplicate(pins: np.ndarray) -> tuple[np.ndarray, int]:
    """Remove exact duplicate coordinates without changing the remaining order."""

    array = _as_pin_array(pins)
    keep: list[int] = []
    seen: set[tuple[float, float]] = set()
    for index, point in enumerate(array):
        key = (float(point[0]), float(point[1]))
        if key not in seen:
            seen.add(key)
            keep.append(index)
    unique = array[np.asarray(keep, dtype=np.int64)]
    return unique, len(array) - len(unique)


def has_exact_duplicate(pins: np.ndarray) -> bool:
    array = _as_pin_array(pins)
    return len({(float(point[0]), float(point[1])) for point in array}) != len(array)


def has_spacing_violation(pins: np.ndarray, minimum_l1_spacing: float = MINIMUM_L1_SPACING) -> bool:
    """Check close pairs in O(n log n) for the small fixed-degree point sets."""

    array = _as_pin_array(pins)
    if minimum_l1_spacing <= 0:
        raise ValueError("minimum_l1_spacing must be positive")
    order = np.argsort(array[:, 0], kind="stable")
    ordered = array[order]
    for left in range(len(ordered)):
        right = left + 1
        while right < len(ordered) and ordered[right, 0] - ordered[left, 0] < minimum_l1_spacing:
            if np.abs(ordered[right] - ordered[left]).sum() < minimum_l1_spacing:
                return True
            right += 1
    return False


def minimum_pair_l1_distance(pins: np.ndarray) -> float:
    array = _as_pin_array(pins)
    if len(array) < 2:
        return float("inf")
    differences = np.abs(array[:, None, :] - array[None, :, :]).sum(axis=-1)
    np.fill_diagonal(differences, np.inf)
    return float(differences.min())


def normalize_net(
    pins: np.ndarray | Iterable[Iterable[float]],
    *,
    expected_degree: int | None = None,
    minimum_l1_spacing: float = MINIMUM_L1_SPACING,
) -> tuple[np.ndarray, NetTransform]:
    """Normalize one unordered pin set under the frozen data contract."""

    original = _as_pin_array(pins)
    original_degree = len(original)
    unique, duplicate_count = stable_deduplicate(original)
    if len(unique) < 2:
        raise InvalidNetError("degree_below_two")
    if expected_degree is not None and len(unique) != expected_degree:
        raise InvalidNetError(
            "degree_changed_by_deduplication",
            f"expected {expected_degree}, found {len(unique)} unique pins",
        )

    offset = unique.min(axis=0)
    spans = unique.max(axis=0) - offset
    longest_span = float(spans.max())
    if longest_span <= ZERO_SPAN_EPSILON:
        raise InvalidNetError("zero_bbox_span")

    scale = (NORMALIZED_UPPER - NORMALIZED_LOWER) / longest_span
    normalized = (unique - offset) * scale + NORMALIZED_LOWER
    normalized = np.clip(normalized, NORMALIZED_LOWER, NORMALIZED_UPPER)
    if has_spacing_violation(normalized, minimum_l1_spacing):
        raise InvalidNetError("minimum_spacing_violation")

    transform = NetTransform(
        offset_xy=offset.copy(),
        scale=scale,
        original_degree=original_degree,
        unique_degree=len(unique),
        duplicate_count=duplicate_count,
    )
    return normalized, transform


def inverse_net(normalized_pins: np.ndarray, transform: NetTransform) -> np.ndarray:
    normalized = _as_pin_array(normalized_pins)
    if transform.scale <= 0 or not np.isfinite(transform.scale):
        raise ValueError("transform scale must be finite and positive")
    return (normalized - NORMALIZED_LOWER) / transform.scale + transform.offset_xy


def validate_normalized_net(
    pins: np.ndarray,
    *,
    expected_degree: int | None = None,
    minimum_l1_spacing: float = MINIMUM_L1_SPACING,
) -> None:
    array = _as_pin_array(pins)
    if expected_degree is not None and len(array) != expected_degree:
        raise InvalidNetError("invalid_shape", f"expected degree {expected_degree}, got {len(array)}")
    tolerance = 16 * np.finfo(np.float64).eps
    if array.min() < NORMALIZED_LOWER - tolerance or array.max() > NORMALIZED_UPPER + tolerance:
        raise InvalidNetError("coordinate_out_of_normalized_bounds")
    if has_spacing_violation(array, minimum_l1_spacing):
        raise InvalidNetError("minimum_spacing_violation")


def project_normalized_net(
    pins: np.ndarray,
    *,
    minimum_l1_spacing: float = MINIMUM_L1_SPACING,
    max_shells: int = 128,
) -> np.ndarray:
    """Clip a candidate and deterministically separate colliding pins."""

    array = _as_pin_array(pins).copy()
    array = np.clip(array, NORMALIZED_LOWER, NORMALIZED_UPPER)
    accepted: list[np.ndarray] = []
    directions = np.asarray(
        [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)],
        dtype=np.float64,
    )
    for point in array:
        if not accepted or all(np.abs(point - other).sum() >= minimum_l1_spacing for other in accepted):
            accepted.append(point.copy())
            continue
        placed = None
        for shell in range(1, max_shells + 1):
            for direction in directions:
                candidate = np.clip(
                    point + direction * minimum_l1_spacing * shell,
                    NORMALIZED_LOWER,
                    NORMALIZED_UPPER,
                )
                if all(np.abs(candidate - other).sum() >= minimum_l1_spacing for other in accepted):
                    placed = candidate
                    break
            if placed is not None:
                break
        if placed is None:
            raise InvalidNetError("minimum_spacing_violation", "deterministic projection exhausted")
        accepted.append(placed)
    projected = np.asarray(accepted, dtype=np.float64)
    validate_normalized_net(projected, expected_degree=len(array), minimum_l1_spacing=minimum_l1_spacing)
    return projected


def pin_set_sha256(pins: np.ndarray) -> str:
    """Hash an unordered pin set using canonical lexicographic float64 bytes."""

    array = _as_pin_array(pins)
    order = np.lexsort((array[:, 1], array[:, 0]))
    canonical = np.asarray(array[order], dtype="<f8", order="C")
    digest = hashlib.sha256()
    digest.update(np.asarray(canonical.shape, dtype="<i8").tobytes())
    digest.update(canonical.tobytes())
    return digest.hexdigest()
