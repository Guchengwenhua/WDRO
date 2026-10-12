"""Cost/skew evaluation through the MSPD C++ implementation."""

from __future__ import annotations

from typing import Sequence, Tuple, Union

import numpy as np

try:
    from . import _mspd
except ImportError as exc:  # pragma: no cover - exercised before extension build
    raise ImportError(
        "The MSPD extension is not built. Run `pip install -e .` from the "
        "repository root."
    ) from exc


Objective = Union[int, float]


def objective_weights(objective: Objective) -> Tuple[float, float]:
    """Return ``(wirelength_weight, skew_weight)``.

    The three integer objectives follow the experiments in the original code:
    1 -> (0.5, 0.5), 2 -> (0.25, 0.75), 3 -> (0.75, 0.25).
    A floating-point value in [0, 1] is interpreted directly as the
    wirelength weight from Eq. (3) of the paper.
    """

    if isinstance(objective, (int, np.integer)) or float(objective).is_integer():
        objective_id = int(objective)
        presets = {1: (0.5, 0.5), 2: (0.25, 0.75), 3: (0.75, 0.25)}
        if objective_id in presets:
            return presets[objective_id]

    wirelength_weight = float(objective)
    if not 0.0 <= wirelength_weight <= 1.0:
        raise ValueError("objective must be 1, 2, 3, or a weight in [0, 1]")
    return wirelength_weight, 1.0 - wirelength_weight


def _integer_coordinates(points: np.ndarray, coordinate_scale: float) -> np.ndarray:
    points = np.asarray(points)
    if points.ndim != 3 or points.shape[-1] != 2:
        raise ValueError("points must have shape [batch, degree, 2]")
    if np.issubdtype(points.dtype, np.floating):
        return (points * coordinate_scale).astype(np.int64)
    return points.astype(np.int64, copy=False)


def evaluate_batch(
    points: np.ndarray,
    hub_indices: Union[np.ndarray, Sequence[Sequence[int]]],
    objective: Objective = 1,
    coordinate_scale: float = 1000.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate selected hubs and return wirelength, skew, and objective.

    Floating-point coordinates are assumed to be normalized and are scaled to
    integers before MSPD is called. Returned values are divided by the same
    scale, matching the training code used for the paper.

    Index 0 is the root. Zero entries in ``hub_indices`` are ignored because
    the adaptive Conductor uses them as padding after it stops selecting hubs.
    """

    if coordinate_scale <= 0:
        raise ValueError("coordinate_scale must be positive")

    integer_points = _integer_coordinates(points, coordinate_scale)
    hubs = np.asarray(hub_indices)
    if hubs.ndim == 1:
        hubs = hubs[None, :]
    if hubs.ndim != 2 or hubs.shape[0] != integer_points.shape[0]:
        raise ValueError("hub_indices must have shape [batch, max_hubs]")

    degree = integer_points.shape[1]
    wirelengths, skews, objectives = [], [], []
    wirelength_weight, skew_weight = objective_weights(objective)

    for coordinates, selected in zip(integer_points, hubs):
        sources = []
        for index in selected.tolist():
            index = int(index)
            if index == 0:
                continue
            if not 0 < index < degree:
                raise ValueError(f"hub index {index} is outside [1, {degree - 1}]")
            if index not in sources:
                sources.append(index)

        candidates = _mspd.solve_single(
            coordinates.reshape(-1).tolist(), sources, False
        )
        wirelength, skew = min(
            candidates,
            key=lambda pair: wirelength_weight * pair[0] + skew_weight * pair[1],
        )
        wirelength /= coordinate_scale
        skew /= coordinate_scale
        wirelengths.append(wirelength)
        skews.append(skew)
        objectives.append(wirelength_weight * wirelength + skew_weight * skew)

    return (
        np.asarray(wirelengths, dtype=np.float32),
        np.asarray(skews, dtype=np.float32),
        np.asarray(objectives, dtype=np.float32),
    )


def evaluate_instance(
    points: np.ndarray,
    hub_indices: Sequence[int],
    objective: Objective = 1,
    coordinate_scale: float = 1000.0,
) -> Tuple[float, float, float]:
    """Convenience wrapper for one routing instance."""

    wirelength, skew, value = evaluate_batch(
        np.asarray(points)[None, ...],
        np.asarray(hub_indices)[None, ...],
        objective,
        coordinate_scale,
    )
    return float(wirelength[0]), float(skew[0]), float(value[0])
