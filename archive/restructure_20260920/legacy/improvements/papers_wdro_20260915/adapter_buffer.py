"""HarRL component pilot: harmonic candidates plus physical RC evaluation.

Upstream is source-pinned in sources/RLBufferTree. Adaptations are explicit in
vendor_harrl/PATCHES.md. This is not a reproduction of Shi--Li optimal buffering.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from vendor_harrl.actor_critic import Actor
from vendor_harrl.evaluator import Evaluator, TimingTree


class CategoricalPhase(nn.Module):
    """A score-function gradient for every phase, including the final edge."""

    def __init__(self, query_dim: int, bins: int = 16):
        super().__init__()
        self.logits = nn.Linear(query_dim, bins)
        self.bins = bins
        self.greedy = False
        self.log_probabilities: list[torch.Tensor] = []
        self.actions: list[torch.Tensor] = []

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        distribution = Categorical(logits=self.logits(query))
        choice = distribution.logits.argmax(-1) if self.greedy else distribution.sample()
        self.log_probabilities.append(distribution.log_prob(choice))
        self.actions.append(choice)
        return choice.to(query.dtype) * (2.0 * math.pi / self.bins)


class ContinuousPhaseTree(TimingTree):
    """Preserve phase origins through splits and reversed overlapping segments."""

    def __init__(self, period: float):
        self.period = period

    def insert_steiner(self, steiner_points, coordinates, pairs):
        # The published helper copies the unshifted phase to every split. We
        # instead use local arclength so the wave continues across corners.
        points = set(coordinates.values())
        raw = []
        omega = 2.0 * math.pi / self.period
        for v, h, phase in pairs:
            start, end = coordinates[v], coordinates[h]
            corner = (start[0], end[1])
            points.add(corner)
            segments = [(start, corner, phase),
                        (corner, end, phase + omega * abs(start[1] - end[1]))]
            raw.extend((a, b, phi) for a, b, phi in segments if a != b)
        # Crossings and collinear endpoints split the graph into unique edges.
        for a, b, _ in raw:
            for c, d, _ in raw:
                if a[0] == b[0] and c[1] == d[1]:
                    crossing = (a[0], c[1])
                    if (min(a[1], b[1]) <= crossing[1] <= max(a[1], b[1])
                            and min(c[0], d[0]) <= crossing[0] <= max(c[0], d[0])):
                        points.add(crossing)
        inverse = {xy: i for i, xy in coordinates.items()}
        for point in sorted(points):
            if point not in inverse:
                inverse[point] = len(coordinates)
                coordinates[inverse[point]] = point
        grouped: dict[tuple[int, int], list[float]] = {}
        for a, b, phase in raw:
            axis = 1 if a[0] == b[0] else 0
            fixed = 1 - axis
            on_segment = sorted((p for p in points if p[fixed] == a[fixed]
                and min(a[axis], b[axis]) <= p[axis] <= max(a[axis], b[axis])),
                key=lambda p: p[axis])
            direction = 1.0 if b[axis] > a[axis] else -1.0
            for lo, hi in zip(on_segment, on_segment[1:]):
                phase_at_lo = phase + omega * abs(lo[axis] - a[axis])
                local_phase = phase_at_lo if direction > 0 else math.pi - phase_at_lo
                key = (inverse[lo], inverse[hi])
                grouped.setdefault(key, []).append(local_phase % (2.0 * math.pi))
        return coordinates, [[a, b, *phases] for (a, b), phases in grouped.items()]


class Adapter(nn.Module):
    """Uniform API for ERM versus finite-candidate WDRO training.

    Each input uses index zero as a fixed source and fixed sink capacitance;
    the returned objective is (max_delay * 1e18 + 0.25 * buffer_count) / 10.
    This scale is fixed before evaluation and does not claim optimal regret.
    """

    def __init__(self, degree: int = 10, device: Any = 'cpu', params=None, **kwargs):
        super().__init__()
        config = dict(params or {})
        config.update(kwargs)
        self.degree = degree
        self.device = torch.device(device)
        self.phase_bins = int(config.get('phase_bins', 16))
        self.buffer_penalty = float(config.get('buffer_penalty', 0.25))
        self.objective_scale = float(config.get('objective_scale', 10.0))
        self.delay_scale = float(config.get('delay_scale', 1e18))
        self.sink_capacitance = float(config.get('sink_capacitance', 0.5))
        if (self.phase_bins < 2 or self.objective_scale <= 0 or self.delay_scale <= 0
                or not math.isfinite(self.objective_scale) or not math.isfinite(self.delay_scale)):
            raise ValueError('finite positive scales and at least two phase bins required')
        self.actor = Actor(3, self.device)
        self.actor.bufferloc = CategoricalPhase(self.actor.d_query, self.phase_bins).to(self.device)
        # A fixed source needs no start-pointer learning.
        for parameter in self.actor.start_ptr.parameters():
            parameter.requires_grad_(False)
        self.evaluator = Evaluator(degree)
        self.period = math.sqrt(2 * self.evaluator.buffer_capacitance * self.evaluator.buffer_resistance
            / (self.evaluator.wire_capacitance * self.evaluator.wire_resistance))
        self.evaluator.tree = ContinuousPhaseTree(self.period)
        self.last_outputs = None

    def evaluate(self, pins: np.ndarray, greedy: bool = False):
        pins = np.asarray(pins, dtype=np.float32)
        if pins.ndim != 3 or pins.shape[1:] != (self.degree, 2) or not np.isfinite(pins).all():
            raise ValueError('finite pins [B,degree,2] required')
        augmented = np.concatenate([pins, np.full((*pins.shape[:2], 1), self.sink_capacitance,
            dtype=np.float32)], axis=-1)
        phase_head = self.actor.bufferloc
        phase_head.log_probabilities = []
        phase_head.actions = []
        phase_head.greedy = bool(greedy)
        was_training = self.actor.training
        if greedy:
            self.actor.eval()
        try:
            outputs, topology_log_prob = self.actor(augmented, bool(greedy))
        finally:
            self.actor.train(was_training)
        phase_log_prob = torch.stack(phase_head.log_probabilities, dim=0).sum(dim=0)
        log_prob = topology_log_prob + phase_log_prob
        array = outputs.detach().cpu().numpy()
        self.last_outputs = array.copy()
        rows = []
        for example, decisions in zip(augmented, array, strict=True):
            coords = {i: tuple(map(float, example[i, :2])) for i in range(self.degree)}
            caps = {i: float(example[i, 2]) * 1e-15 for i in range(self.degree)}
            connections = [(int(decisions[3*j]), int(decisions[3*j+1]), float(decisions[3*j+2]))
                           for j in range(self.degree - 1)]
            rows.append(self.evaluator.eval_func(coords, caps, connections))
        values = np.asarray(rows, dtype=np.float64)
        if not np.isfinite(values).all() or (values < -1e-12).any():
            raise RuntimeError('invalid physical buffering metrics')
        delay, skew = values[:, 0] * self.delay_scale, values[:, 1] * self.delay_scale
        objective = (delay + self.buffer_penalty * values[:, 3]) / self.objective_scale
        metrics = {'delay_raw_seconds': values[:, 0], 'delay_scaled': delay, 'skew_scaled': skew,
                   'length': values[:, 2], 'buffers': values[:, 3], 'objective': objective,
                   'phase_mean': array[:, 2::3].mean(axis=1)}
        return torch.as_tensor(objective, dtype=torch.float32, device=self.device), log_prob, metrics

    def metadata(self):
        return {'paper': '25_ICCAD', 'method': 'HarRL component pilot with categorical phase',
                'upstream_commit': 'c4f921a45a46eaec07b1f4afe1b5a73ec5a38e31',
                'phase_bins': self.phase_bins, 'period': self.period,
                'objective': f'(max_delay_seconds * {self.delay_scale} + {self.buffer_penalty} * buffers) / {self.objective_scale}',
                'delay_scale': self.delay_scale, 'buffer_penalty': self.buffer_penalty,
                'objective_scale': self.objective_scale,
                'buffer_solver': 'upstream local delay-gain greedy, not Shi-Li DP',
                'sink_capacitance_farad': self.sink_capacitance * 1e-15,
                'source_index': 0, 'physical_scale': 'normalized synthetic coordinates and fixed upstream RC'}
