"""Finite-budget coordinate adversaries using only the Actor's own tree lengths.

No reference solver or reference objective occurs in this module. All final
selection uses the same greedy Actor length and direct labeled-coordinate cost.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from torch import nn

from geometry import costs, legalize


class SetRegressor(nn.Module):
    """Permutation-invariant, unconstrained scalar prediction (no output ReLU)."""

    def __init__(self, width: int = 128):
        super().__init__()
        self.point = nn.Sequential(nn.Linear(2, width), nn.ReLU(),
                                   nn.Linear(width, width), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(2 * width, width), nn.ReLU(),
                                  nn.Linear(width, 1))

    def forward(self, pins: torch.Tensor) -> torch.Tensor:
        h = self.point(pins)
        return self.head(torch.cat((h.mean(1), h.amax(1)), -1)).squeeze(-1)


class ProposalPolicy(nn.Module):
    """Permutation-equivariant Gaussian proposal; legality is a fixed pushforward."""

    def __init__(self, epsilon: float, sigma: float, width: int = 64):
        super().__init__()
        self.epsilon = float(epsilon)
        self.point = nn.Sequential(nn.Linear(2, width), nn.ReLU(),
                                   nn.Linear(width, width), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(3 * width, width), nn.ReLU(),
                                  nn.Linear(width, 2))
        # Starting proposals have zero mean. Learned scale is scalar and shared.
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.log_sigma = nn.Parameter(torch.tensor(float(np.log(sigma))))

    def distribution(self, centers: torch.Tensor) -> torch.distributions.Normal:
        h = self.point(centers)
        context = torch.cat((h.mean(1), h.amax(1)), -1)
        context = context[:, None, :].expand(-1, h.shape[1], -1)
        mu = self.epsilon * self.head(torch.cat((h, context), -1)).tanh()
        sigma = self.log_sigma.clamp(np.log(1e-4), np.log(self.epsilon)).exp()
        return torch.distributions.Normal(mu, sigma)


@dataclass
class SearchResult:
    selected: np.ndarray
    selected_cost: np.ndarray
    candidates: np.ndarray
    scores: dict[str, np.ndarray]
    transport: np.ndarray
    selected_index: np.ndarray
    proposal_loss: torch.Tensor | None = None


def _rms_unit(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.sqrt(np.mean(x * x, axis=(-2, -1), keepdims=True)), 1e-12)


class CoordinateSearcher:
    def __init__(self, protocol: dict, surrogate: nn.Module,
                 proposal: ProposalPolicy, device: torch.device):
        self.p = protocol
        self.surrogate = surrogate
        self.proposal = proposal
        self.device = device

    def _legal(self, centers: np.ndarray, deltas: np.ndarray) -> np.ndarray:
        # Legalize is shared by every method; callers also enforce the frozen cap.
        cap = float(self.p["epsilon"])
        delta = np.clip(deltas, -cap, cap)
        b, k, n, d = delta.shape
        tiled = np.broadcast_to(centers[:, None], delta.shape).reshape(b * k, n, d)
        return legalize(tiled, delta.reshape(b * k, n, d)).reshape(b, k, n, d)

    def _score(self, scorer: Callable, candidates: np.ndarray) -> dict:
        b, k, n, d = candidates.shape
        flat = scorer(candidates.reshape(b * k, n, d))
        return {name: np.asarray(value).reshape(b, k, *np.asarray(value).shape[1:])
                for name, value in flat.items()}

    def search(self, method: str, centers: np.ndarray, lam: float,
               scorer: Callable, rng: np.random.Generator,
               candidate_count: int | None = None) -> SearchResult:
        b, n, _ = centers.shape
        k = int(candidate_count or self.p["candidates"])
        if k < 2:
            raise ValueError("A search includes its center and at least one proposal")
        sigma = float(self.p["search_sigma"])
        proposal_loss = None
        if method in ("wdro_random", "random"):
            delta = rng.normal(0, sigma, (b, k - 1, n, 2)).astype(np.float32)
            moved = self._legal(centers, delta)
            candidates = np.concatenate((centers[:, None], moved), 1)
            scores = self._score(scorer, candidates)
        elif method == "wdro_es":
            if k != 8:
                raise ValueError("Frozen ES budget requires center+6 antithetic+1 moved = 8")
            directions = _rms_unit(rng.normal(size=(b, 3, n, 2))).astype(np.float32)
            pairs = np.stack((directions, -directions), 2).reshape(b, 6, n, 2)
            candidates = np.concatenate((centers[:, None], self._legal(centers, sigma * pairs)), 1)
            scores = self._score(scorer, candidates)
            pair_values = (scores["length"] - lam * costs(centers, candidates))[:, 1:].reshape(b, 3, 2)
            derivative = (pair_values[:, :, 0] - pair_values[:, :, 1]) / (2 * sigma)
            direction = _rms_unit(np.mean(derivative[:, :, None, None] * directions, 1))
            moved = self._legal(centers, float(self.p["es_step_size"]) * direction[:, None])
            last = self._score(scorer, moved)
            candidates = np.concatenate((candidates, moved), 1)
            scores = {name: np.concatenate((value, last[name]), 1) for name, value in scores.items()}
        elif method == "wdro_critic":
            delta = rng.normal(0, sigma, (b, k - 1, n, 2)).astype(np.float32)
            points = self._legal(centers, delta)
            ct = torch.as_tensor(np.broadcast_to(centers[:, None], points.shape).copy(),
                                 dtype=torch.float32, device=self.device).reshape(-1, n, 2)
            # The direct indexwise L1 penalty is the same cost used for selection.
            # Legality remains a fixed projection/rejection after each update.
            for _ in range(int(self.p["critic_steps"])):
                pt = torch.tensor(points.reshape(-1, n, 2), device=self.device,
                                  dtype=torch.float32, requires_grad=True)
                objective = self.surrogate(pt) - lam * (pt - ct).abs().sum(-1).mean(-1)
                grad, = torch.autograd.grad(objective.sum(), pt)
                grad = grad / grad.square().mean((1, 2), keepdim=True).sqrt().clamp_min(1e-12)
                delta = (pt.detach() + float(self.p["critic_step_size"]) * grad - ct).cpu().numpy()
                points = self._legal(centers, delta.reshape(b, k - 1, n, 2))
            candidates = np.concatenate((centers[:, None], points), 1)
            scores = self._score(scorer, candidates)
        elif method == "wdro_rl":
            ct = torch.as_tensor(centers, dtype=torch.float32, device=self.device)
            distribution = self.proposal.distribution(ct)
            # rsample is deliberately detached: the estimator is score-function
            # REINFORCE through log_prob, not a pathwise derivative of hard scores.
            draws = distribution.sample((k - 1,))
            log_prob = distribution.log_prob(draws).sum((-1, -2)).transpose(0, 1)
            delta = draws.permute(1, 0, 2, 3).cpu().numpy()
            moved = self._legal(centers, delta)
            candidates = np.concatenate((centers[:, None], moved), 1)
            scores = self._score(scorer, candidates)
            rewards = torch.as_tensor((scores["length"] - lam * costs(centers, candidates))[:, 1:],
                                      device=self.device, dtype=torch.float32)
            if k <= 2:
                raise ValueError("Leave-one-out proposal baseline needs at least two draws")
            loo = (rewards.sum(1, keepdim=True) - rewards) / (k - 2)
            proposal_loss = -((rewards - loo).detach() * log_prob).mean()
        else:
            raise ValueError(f"Unsupported search method: {method}")
        transport = costs(centers, candidates)
        objective = scores["length"] - lam * transport
        if not np.isfinite(objective).all():
            raise FloatingPointError("Nonfinite search score")
        # First occurrence wins ties, so no-move wins exact ties.
        chosen = np.argmax(objective, axis=1)
        rows = np.arange(b)
        return SearchResult(candidates[rows, chosen], transport[rows, chosen], candidates,
                            scores, transport, chosen, proposal_loss)
