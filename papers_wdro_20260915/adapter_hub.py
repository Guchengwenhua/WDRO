"""DAC 2025 original Selector-Conductor + original C++ MSPD adapter.

Uses the exact released CPU MSPD construction (including its 11-point
Prim-Dijkstra alpha sweep and scale-1000 integer coordinates). The WDRO
adversarial candidate set remains external. This is not a new attacker.
The optional hub-count shaping reward defaults to zero so inner and outer
optimization share the physical cost-skew objective in this pilot.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

from adapter_timing import _package, _load_source

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "sources" / "An-Adaptive-Learning-Approach-for-Hub-Node-Selection"


def _official_components():
    prefix = "_paper25_hub_models"
    _package(prefix, SOURCE / "models")
    network = _load_source(prefix + ".selector_conductor", SOURCE / "models" / "selector_conductor.py")
    extension_name = prefix + "._mspd"
    if extension_name not in sys.modules:
        candidates = sorted((ROOT / "artifacts" / "build").glob("_mspd*.so"))
        if len(candidates) != 1:
            raise RuntimeError("build the original MSPD C++ extension into artifacts/build/_mspd*.so first")
        spec = importlib.util.spec_from_file_location(extension_name, candidates[0])
        extension = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(extension)
        sys.modules[extension_name] = extension
        setattr(sys.modules[prefix], "_mspd", extension)
    evaluator = _load_source(prefix + ".evaluate", SOURCE / "models" / "evaluate.py")
    return network, evaluator


class Adapter(nn.Module):
    paper = "25_DAC"
    score_kind = "fixed_scale_wirelength_skew"

    def __init__(self, degree: int, device="cpu", params=None):
        super().__init__()
        self.degree = int(degree)
        self.params = dict(params or {})
        self.beta = float(self.params.get("beta", self.params.get("weight", 0.5)))
        self.normalizer = float(self.params.get("normalizer", 2 * (degree - 1)))
        self.max_hubs = int(self.params.get("max_hubs", min(5, degree - 1)))
        self.coordinate_scale = float(self.params.get("coordinate_scale", 1000.0))
        self.policy_bonus_weight = float(self.params.get("policy_bonus_weight", 0.0))
        if self.degree < 2 or not 0 <= self.beta <= 1 or self.normalizer <= 0:
            raise ValueError("invalid degree, weight, or normalizer")
        if not 1 <= self.max_hubs <= self.degree - 1:
            raise ValueError("max_hubs must be between 1 and degree-1")
        network, self._evaluator = _official_components()
        self.actor = network.Actor(self.degree, torch.device(device), float(self.params.get("reward_gamma", 0.5)))
        self.to(device)

    def evaluate(self, pins: np.ndarray, greedy: bool):
        points = np.asarray(pins, dtype=np.float32)
        if points.ndim != 3 or points.shape[1:] != (self.degree, 2):
            raise ValueError("pins must have shape [batch, degree, 2]")
        if not np.isfinite(points).all():
            raise ValueError("non-finite coordinates")
        device = next(self.actor.parameters()).device
        self.actor.device = device
        selected, log_prob, reward = self.actor(
            torch.as_tensor(points, device=device), self.max_hubs, deterministic=bool(greedy)
        )
        hubs = selected.detach().cpu().numpy()
        self.last_hubs = hubs.copy()
        legal = np.asarray([
            all(0 <= int(h) < self.degree for h in row)
            and len([int(h) for h in row if h]) == len({int(h) for h in row if h})
            and (not np.any(row == 0) or not np.any(row[np.flatnonzero(row == 0)[0]:] != 0))
            for row in hubs
        ], dtype=bool)
        if not legal.all():
            raise RuntimeError("official Selector emitted invalid hub indices")
        # Avoid the public preset parser's ambiguity for 1.0: obtain L and S
        # with an internal, explicit weighted candidate selection in that case.
        if self.beta == 1.0:
            integers = self._evaluator._integer_coordinates(points, self.coordinate_scale)
            pairs = [min(self._evaluator._mspd.solve_single(net.reshape(-1).tolist(), [int(h) for h in hs if h], False), key=lambda p:p[0]) for net,hs in zip(integers,hubs)]
            lengths, skews = np.asarray(pairs, dtype=np.float64).T / self.coordinate_scale
            objective = lengths.copy()
        else:
            lengths, skews, objective = self._evaluator.evaluate_batch(
                points, hubs, objective=self.beta, coordinate_scale=self.coordinate_scale
            )
        score = np.asarray(objective, dtype=np.float64) / self.normalizer
        raw_reward = reward.detach().cpu().numpy().astype(np.float64)
        if not np.isfinite(score).all():
            raise RuntimeError("non-finite MSPD cost")
        metrics = {
            "wirelength": lengths, "skew": skews,
            "raw_objective": objective, "score": score,
            "hub_count": np.sum(hubs != 0, axis=1),
            "legal": legal, "raw_policy_reward": raw_reward,
            "policy_reward": self.policy_bonus_weight * raw_reward / self.normalizer,
        }
        return torch.as_tensor(score, dtype=torch.float32, device=device).detach(), log_prob, metrics
