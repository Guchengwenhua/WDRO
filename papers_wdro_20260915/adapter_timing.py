"""ICCAD 2023 official TripleGAT Actor, adapted for a rooted WDRO pilot.

The imported network and path evaluator are commit-pinned official source.
Only import names are rewritten in memory to avoid three repositories' `models`
packages colliding. Wirelength is a NumPy translation of algorithms/eval.c.
Score = ((1-alpha)*wirelength + alpha*max_source_sink_path)/(2*(n-1));
this is a fixed-scale routing objective, NOT optimality regret or an RC delay.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "sources" / "Timing-RST"


def _package(name: str, directory: Path) -> None:
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(directory)]
        package.__package__ = name
        sys.modules[name] = package


def _load_source(name: str, path: Path, replacements=()):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        source = path.read_text()
        for old, new in replacements:
            source = source.replace(old, new)
        exec(compile(source, str(path), "exec"), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _official_network():
    prefix = "_paper23_timing_models"
    _package(prefix, SOURCE / "models")
    for stem in ("utils", "self_attn", "gcn", "gat", "model_sorted"):
        module = _load_source(
            prefix + "." + stem, SOURCE / "models" / (stem + ".py"),
            (("from models.", "from " + prefix + "."),),
        )
    return module


def official_wirelength(pins: np.ndarray, edge_pairs: np.ndarray) -> np.ndarray:
    """Exact arithmetic translation of official eval.c (vertical then horizontal)."""
    points = np.asarray(pins, dtype=np.float64)
    edges = np.asarray(edge_pairs, dtype=np.int64)
    batch, degree, _ = points.shape
    if edges.shape != (batch, degree - 1, 2):
        raise ValueError("expected n-1 directed pin edges")
    values = np.empty(batch, dtype=np.float64)
    for b, (net, pairs) in enumerate(zip(points, edges)):
        low_x, high_x = net[:, 0].copy(), net[:, 0].copy()
        low_y, high_y = net[:, 1].copy(), net[:, 1].copy()
        for u, v in pairs:
            low_y[u] = min(low_y[u], net[v, 1])
            high_y[u] = max(high_y[u], net[v, 1])
            low_x[v] = min(low_x[v], net[u, 0])
            high_x[v] = max(high_x[v], net[u, 0])
        values[b] = np.sum(high_y - low_y + high_x - low_x)
    return values


def valid_pin_tree(edge_pairs: np.ndarray, degree: int) -> np.ndarray:
    result = []
    for edges in np.asarray(edge_pairs):
        parent = list(range(degree))
        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        valid = len(edges) == degree - 1
        for u, v in edges:
            if not (0 <= u < degree and 0 <= v < degree) or u == v:
                valid = False
                break
            pu, pv = find(int(u)), find(int(v))
            if pu == pv:
                valid = False
                break
            parent[pu] = pv
        result.append(valid and len({find(i) for i in range(degree)}) == 1)
    return np.asarray(result, dtype=bool)


class Adapter(nn.Module):
    paper = "23_ICCAD"
    score_kind = "fixed_scale_wirelength_max_path"

    def __init__(self, degree: int, device="cpu", params=None):
        super().__init__()
        self.degree = int(degree)
        self.params = dict(params or {})
        self.alpha = float(self.params.get("alpha", self.params.get("weight", 0.5)))
        self.normalizer = float(self.params.get("normalizer", 2 * (degree - 1)))
        if self.degree < 2 or not 0 <= self.alpha <= 1 or self.normalizer <= 0:
            raise ValueError("invalid degree, weight, or normalizer")
        self.actor = _official_network().Actor()
        self._distance = _load_source(
            "_paper23_official_distance", SOURCE / "utils" / "distance.py"
        ).get_distance
        self.to(device)

    def evaluate(self, pins: np.ndarray, greedy: bool):
        points = np.asarray(pins, dtype=np.float32)
        if points.ndim != 3 or points.shape[1:] != (self.degree, 2):
            raise ValueError("pins must have shape [batch, degree, 2]")
        if not np.isfinite(points).all():
            raise ValueError("non-finite coordinates")
        device = next(self.actor.parameters()).device
        tensor = torch.as_tensor(points, device=device)
        _, log_prob, indexes = self.actor(tensor, deterministic=bool(greedy))
        pairs = torch.stack(indexes, dim=1).reshape(len(points), -1, 2).detach().cpu().numpy()
        self.last_edges = pairs.copy()
        legal = valid_pin_tree(pairs, self.degree)
        if not legal.all():
            raise RuntimeError("official Actor emitted an invalid pin tree")
        lengths = official_wirelength(points, pairs)
        max_paths = []
        for net, edges in zip(points.astype(np.float64), pairs):
            nodes = list(map(tuple, net))
            graph = [[nodes[int(u)], nodes[int(v)]] for u, v in edges]
            distances, radius = self._distance(graph, nodes[0])
            if len(distances) != len(set(nodes)):
                raise RuntimeError("official path evaluator did not reach all pins")
            max_paths.append(radius)
        radii = np.asarray(max_paths, dtype=np.float64)
        objective = (1.0 - self.alpha) * lengths + self.alpha * radii
        score = objective / self.normalizer
        if not np.isfinite(score).all():
            raise RuntimeError("non-finite timing cost")
        metrics = {
            "wirelength": lengths, "max_path": radii,
            "raw_objective": objective, "score": score,
            "edge_count": np.full(len(points), self.degree - 1, dtype=np.int64), "legal": legal,
            "policy_reward": np.zeros(len(points), dtype=np.float64),
        }
        return torch.as_tensor(score, dtype=torch.float32, device=device).detach(), log_prob, metrics
