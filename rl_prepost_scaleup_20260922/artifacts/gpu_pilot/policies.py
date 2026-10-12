"""Small, budget-matched learned adversaries for a black-box REST scorer.

The only reward is ``length(z) - lam * mean_pin_L1(x, z)``.  Every search
evaluates the center once and K-1 proposals.  Coordinates, reward values and
likelihoods used in an update are from the policy before that update.

Stochastic policies act in a latent Gaussian space; clipping/legalization is
part of the environment.  PPO ratios therefore use the *latent* actions, not
a fictitious density on the clipped coordinates.  Pin order is exchangeable.
The MAPPO method cooperates among attacking pins, not between REST and attack.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, fields
import math
from typing import Callable

import numpy as np
import torch
from torch import nn


@dataclass
class Config:
    width: int = 64
    lr: float = 3e-4
    sigma: float = .04
    clip: float = .2
    epochs: int = 4
    epsilon: float = .08
    move_clip: float = .04
    grad_clip: float = 1.0
    value_coef: float = .5
    entropy_coef: float = 0.0
    replay_capacity: int = 8192
    replay_batch: int = 128
    ddpg_updates: int = 1
    polyak: float = .005


def costs(centers: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Labeled-coordinate mean-pin L1 cost, identical to rl_only geometry."""
    x = np.asarray(centers, dtype=np.float64)
    z = np.asarray(candidates, dtype=np.float64)
    return np.abs(z - (x[:, None] if z.ndim == 4 else x)).sum(-1).mean(-1)


def legalize(centers: np.ndarray, deltas: np.ndarray, epsilon: float = .08) -> np.ndarray:
    """Apply relative L-infinity limit, box clipping, and duplicate fallback."""
    x = np.asarray(centers, dtype=np.float32)
    d = np.clip(np.asarray(deltas), -epsilon, epsilon)
    z = np.clip((x[:, None] if d.ndim == 4 else x) + d, .1, .9).astype(np.float32)
    shape = z.shape
    flat = z.reshape(-1, x.shape[1], 2)
    for i, row in enumerate(flat):
        distances = np.abs(row[:, None].astype(np.float64) - row[None].astype(np.float64)).sum(-1)
        np.fill_diagonal(distances, np.inf)
        if np.any(distances < 1e-6):
            flat[i] = x[i // shape[1] if d.ndim == 4 else i]
    return flat.reshape(shape)


def replacement_advantages(objectives: np.ndarray) -> np.ndarray:
    """Exact max-of-K score-function advantages for columns 1..K-1.

    Column zero is the fixed center.  Removing proposal j gives an
    action-independent conditional baseline b_j.  R-b_j is nonzero only for
    a strict winning proposal, and equals its margin above the runner-up.
    """
    values = np.asarray(objectives, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("objectives must have shape (B, K), K >= 2")
    maximum = values.max(axis=1)
    return np.stack([maximum - np.delete(values, j, axis=1).max(axis=1)
                     for j in range(1, values.shape[1])], axis=1)


def _features(x: torch.Tensor, displacement: torch.Tensor, lam: torch.Tensor,
              context: torch.Tensor, epsilon: float) -> torch.Tensor:
    # context = [fraction of moves already used, current gain, best gain].
    scalar = torch.cat((lam[:, None] / (1 + lam[:, None]), context), dim=-1)
    return torch.cat((x - .5, displacement / epsilon,
                      scalar[:, None].expand(-1, x.shape[1], -1)), dim=-1)


class PinActor(nn.Module):
    """Shared pin actor using its local features and invariant global context."""
    def __init__(self, config: Config):
        super().__init__()
        self.epsilon = config.epsilon
        w = config.width
        self.encoder = nn.Sequential(nn.Linear(8, w), nn.Tanh(), nn.Linear(w, w), nn.Tanh())
        self.head = nn.Sequential(nn.Linear(3 * w, w), nn.Tanh(), nn.Linear(w, 2))
        nn.init.normal_(self.head[-1].weight, std=.01)
        nn.init.zeros_(self.head[-1].bias)
        self.log_std = nn.Parameter(torch.full((2,), math.log(config.sigma)))

    def forward(self, x, displacement, lam, context):
        h = self.encoder(_features(x, displacement, lam, context, self.epsilon))
        pooled = torch.cat((h.mean(dim=1), h.amax(dim=1)), dim=-1)
        return self.epsilon * torch.tanh(self.head(torch.cat(
            (h, pooled[:, None].expand(-1, x.shape[1], -1)), dim=-1)))

    def std(self):
        return self.log_std.clamp(math.log(.002), math.log(.2)).exp()


class CentralValue(nn.Module):
    """Permutation-invariant state value or joint-action value."""
    def __init__(self, config: Config):
        super().__init__()
        self.epsilon = config.epsilon
        w = config.width
        self.encoder = nn.Sequential(nn.Linear(8, w), nn.Tanh(), nn.Linear(w, w), nn.Tanh())
        self.head = nn.Sequential(nn.Linear(2 * w, w), nn.Tanh(), nn.Linear(w, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x, displacement, lam, context):
        h = self.encoder(_features(x, displacement, lam, context, self.epsilon))
        return self.head(torch.cat((h.mean(1), h.amax(1)), dim=-1)).squeeze(-1)


class LearnedAttacker:
    """Interface used by fixed-opponent and coevolution research runners.

    ``search(centers, lam, scorer, rng, train=False, k=8)`` returns numpy
    arrays.  Scorer accepts (M,N,2) float32 coordinates and returns (M,).
    ``lam`` is a nonnegative scalar or length-B vector.  The caller owns the
    numpy Generator and must checkpoint its bit_generator.state.  One seed
    from it controls all torch sampling in a search; replay sampling also
    consumes that generator.  Constructor initialization uses torch's RNG.

    ``state_dict`` includes networks, optimizers, target networks, replay,
    configuration and update counters.  There is no retained computation
    graph.  Evaluation neither updates parameters nor mutates replay.
    """
    METHODS = {"reinforce", "ppo", "bestof_ppo", "sequential_ppo", "mappo", "maddpg_bandit"}

    def __init__(self, method: str, degree: int, device="cpu", config=None):
        self.method = {"ddpg": "maddpg_bandit", "maddpg": "maddpg_bandit"}.get(method, method)
        if self.method not in self.METHODS:
            raise ValueError(f"Unknown attacker: {method}")
        supplied = {} if config is None else (asdict(config) if isinstance(config, Config) else dict(config))
        unknown = set(supplied) - {f.name for f in fields(Config)}
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        self.config = Config(**supplied)
        c = self.config
        if (degree < 2 or c.width < 1 or c.sigma <= 0 or c.epsilon <= 0 or c.move_clip <= 0
                or c.lr <= 0 or c.epochs < 1 or not 0 < c.clip < 1 or c.grad_clip <= 0
                or c.replay_capacity < 1 or c.replay_batch < 1 or c.ddpg_updates < 1
                or not 0 < c.polyak <= 1):
            raise ValueError("Invalid attacker configuration")
        self.degree = int(degree)
        self.device = torch.device(device)
        self.actor = PinActor(c).to(self.device)
        self.value = CentralValue(c).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=c.lr)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=c.lr)
        self.target_actor = deepcopy(self.actor).requires_grad_(False)
        self.target_value = deepcopy(self.value).requires_grad_(False)
        self.replay = None
        self.updates = 0

    def _tensor(self, value):
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)

    def _sample(self, x, displacement, lam, context, generator):
        with torch.no_grad():
            mean = self.actor(x, displacement, lam, context)
            action = mean + self.actor.std() * torch.randn(
                mean.shape, generator=generator, device=self.device)
            log_pin = torch.distributions.Normal(mean, self.actor.std()).log_prob(action).sum(-1)
        return action, log_pin

    def _clip_grad_step(self, loss, optimizer, parameters):
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite policy/value loss")
        parameters = list(parameters)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, self.config.grad_clip, error_if_nonfinite=True)
        optimizer.step()
        return float(norm.detach().cpu())

    def _ppo_update(self, x, displacement, lam, context, actions, old_log_pin,
                    targets, advantages=None, per_pin=False):
        c = self.config
        with torch.no_grad():
            advantage = targets - self.value(x, displacement, lam, context) if advantages is None else advantages
            advantage = advantage.detach()
        out = {}
        for _ in range(c.epochs):
            mean = self.actor(x, displacement, lam, context)
            distribution = torch.distributions.Normal(mean, self.actor.std())
            log_pin = distribution.log_prob(actions).sum(-1)
            log_ratio = log_pin - old_log_pin if per_pin else (log_pin - old_log_pin).sum(-1)
            ratio = log_ratio.clamp(-30, 30).exp()
            adv = advantage[:, None] if per_pin else advantage
            surrogate = torch.minimum(ratio * adv, ratio.clamp(1-c.clip, 1+c.clip) * adv)
            # Joint ratios use the joint entropy; per-pin ratios use per-pin entropy.
            entropy = distribution.entropy().sum(-1)
            if not per_pin:
                entropy = entropy.sum(-1)
            actor_loss = -surrogate.mean() - c.entropy_coef * entropy.mean()
            grad = self._clip_grad_step(actor_loss, self.actor_optimizer, self.actor.parameters())
            value_loss = .5 * (self.value(x, displacement, lam, context) - targets).square().mean()
            self._clip_grad_step(c.value_coef * value_loss, self.value_optimizer, self.value.parameters())
            self.updates += 1
            out = dict(actor_loss=float(actor_loss.detach().cpu()), value_loss=float(value_loss.detach().cpu()),
                       actor_grad_norm=grad, approximate_kl=float(((ratio - 1) - log_ratio).mean().detach().cpu()),
                       clip_fraction=float(((ratio-1).abs() > c.clip).float().mean().detach().cpu()),
                       advantage_mean=float(advantage.mean().cpu()))
        return out

    def _reinforce_update(self, x, lam, actions, objective):
        b, j, n, _ = actions.shape
        reward = objective[:, 1:]
        # Other sampled proposals are independent of action j conditional on x.
        baseline = (reward.sum(1, keepdims=True) - reward) / (j - 1) if j > 1 else objective[:, :1]
        advantage = self._tensor(reward - baseline).reshape(-1)
        xx = x[:, None].expand(-1, j, -1, -1).reshape(-1, n, 2)
        ll = lam[:, None].expand(-1, j).reshape(-1)
        zeros = torch.zeros_like(xx)
        context = torch.zeros((b*j, 3), device=self.device)
        mean = self.actor(xx, zeros, ll, context)
        log_probability = torch.distributions.Normal(mean, self.actor.std()).log_prob(actions.reshape(-1, n, 2)).sum((1, 2))
        loss = -(log_probability * advantage).mean()
        norm = self._clip_grad_step(loss, self.actor_optimizer, self.actor.parameters())
        self.updates += 1
        return dict(actor_loss=float(loss.detach().cpu()), actor_grad_norm=norm,
                    advantage_mean=float(advantage.mean().cpu()))

    def _push_replay(self, x, actions, lam, rewards):
        batch = {"x": np.asarray(x, dtype=np.float32).copy(),
                 "a": np.asarray(actions, dtype=np.float32).copy(),
                 "lam": np.asarray(lam, dtype=np.float32).copy(),
                 "reward": np.asarray(rewards, dtype=np.float32).copy()}
        self.replay = {name: (v if self.replay is None else np.concatenate((self.replay[name], v), axis=0))[-self.config.replay_capacity:]
                       for name, v in batch.items()}

    def _ddpg_update(self, rng):
        out = {}
        c = self.config
        for _ in range(c.ddpg_updates):
            indices = rng.choice(len(self.replay["reward"]), size=min(c.replay_batch, len(self.replay["reward"])), replace=False)
            x, action, lam, reward = (self._tensor(self.replay[name][indices]) for name in ("x", "a", "lam", "reward"))
            context = torch.zeros((len(indices), 3), device=self.device)
            # All transitions terminate: gamma * (1-done) = 0.  Reward is the
            # exact Bellman target; adding target-Q here would change the task.
            value_loss = .5 * (self.value(x, action, lam, context) - reward).square().mean()
            self._clip_grad_step(value_loss, self.value_optimizer, self.value.parameters())
            self.value.requires_grad_(False)
            try:
                mu = self.actor(x, torch.zeros_like(x), lam, context)
                executed = torch.clamp(x + mu, .1, .9) - x
                actor_loss = -self.value(x, executed, lam, context).mean()
                norm = self._clip_grad_step(actor_loss, self.actor_optimizer, self.actor.parameters())
            finally:
                self.value.requires_grad_(True)
            with torch.no_grad():
                for target, source in ((self.target_actor, self.actor), (self.target_value, self.value)):
                    for t, s in zip(target.parameters(), source.parameters()):
                        t.lerp_(s, c.polyak)
            self.updates += 1
            out = dict(actor_loss=float(actor_loss.detach().cpu()), value_loss=float(value_loss.detach().cpu()),
                       actor_grad_norm=norm, replay_size=len(self.replay["reward"]))
        return out

    def search(self, centers: np.ndarray, lam, scorer: Callable, rng: np.random.Generator,
               train: bool = False, k: int = 8) -> dict:
        centers = np.asarray(centers, dtype=np.float32)
        if (centers.ndim != 3 or centers.shape[1:] != (self.degree, 2)
                or not len(centers) or not np.isfinite(centers).all()
                or np.any(centers < .1) or np.any(centers > .9)):
            raise ValueError("centers must be finite legal (B,degree,2) coordinates")
        if not isinstance(k, (int, np.integer)) or k < 2:
            raise ValueError("k must be an integer >= 2, including the center")
        b, n, _ = centers.shape
        distances = np.abs(centers[:, :, None].astype(np.float64) - centers[:, None].astype(np.float64)).sum(-1)
        distances[:, np.arange(n), np.arange(n)] = np.inf
        if np.any(distances < 1e-6):
            raise ValueError("centers must have distinct pins separated by L1 >= 1e-6")
        lambdas = np.broadcast_to(np.asarray(lam, dtype=np.float64), (b,)).copy()
        if not np.isfinite(lambdas).all() or np.any(lambdas < 0):
            raise ValueError("lam must be finite and nonnegative")
        x, ll = self._tensor(centers), self._tensor(lambdas)
        generator = torch.Generator(device=self.device)
        generator.manual_seed(int(rng.integers(0, 2**63 - 1)))
        scorer_batches = 0

        def score(points):
            nonlocal scorer_batches
            result = np.asarray(scorer(np.asarray(points, dtype=np.float32)), dtype=np.float64)
            if result.shape != (len(points),) or not np.isfinite(result).all():
                raise ValueError("scorer must return one finite length per coordinate set")
            scorer_batches += 1
            return result

        candidates = np.empty((b, k, n, 2), dtype=np.float32)
        lengths = np.empty((b, k), dtype=np.float64)
        objective = np.empty((b, k), dtype=np.float64)
        candidates[:, 0] = centers
        lengths[:, 0] = objective[:, 0] = score(centers)
        diagnostics = {}
        if self.method in {"sequential_ppo", "mappo"}:
            displacement = torch.zeros_like(x)
            best = objective[:, 0].copy()
            previous = best.copy()
            records, rewards = [], []
            for j in range(1, k):
                context = self._tensor(np.column_stack((np.full(b, (j-1)/(k-1)),
                                                        previous-objective[:, 0], best-objective[:, 0])))
                action, log_pin = self._sample(x, displacement, ll, context, generator)
                delta = displacement.cpu().numpy() + action.clamp(-self.config.move_clip, self.config.move_clip).cpu().numpy()
                proposal = legalize(centers, delta, self.config.epsilon)
                candidates[:, j] = proposal
                lengths[:, j] = score(proposal)
                objective[:, j] = lengths[:, j] - lambdas * costs(centers, proposal)
                next_best = np.maximum(best, objective[:, j])
                rewards.append(next_best - best)
                records.append((displacement, context, action, log_pin))
                displacement = self._tensor(proposal - centers)
                previous, best = objective[:, j], next_best
            increments = np.stack(rewards, axis=1)
            returns = np.flip(np.cumsum(np.flip(increments, axis=1), axis=1), axis=1).copy()
            diagnostics["telescoping_error"] = float(np.abs(increments.sum(1) - (objective.max(1)-objective[:, 0])).max())
            diagnostics["mean_return"] = float(returns[:, 0].mean())
            if train:
                # Time-major flattened order matches torch.cat(records).
                xx = x.repeat(k-1, 1, 1)
                dd, cc, aa, lp = (torch.cat([row[col] for row in records], dim=0) for col in range(4))
                diagnostics.update(self._ppo_update(xx, dd, ll.repeat(k-1), cc, aa, lp,
                                                     self._tensor(returns.T.copy()).reshape(-1),
                                                     per_pin=self.method == "mappo"))
        else:
            j = k - 1
            xx = x[:, None].expand(-1, j, -1, -1).reshape(-1, n, 2)
            lam_flat = ll[:, None].expand(-1, j).reshape(-1)
            context = torch.zeros((b*j, 3), device=self.device)
            actions, log_pin = self._sample(xx, torch.zeros_like(xx), lam_flat, context, generator)
            proposal = legalize(centers, actions.cpu().numpy().reshape(b, j, n, 2), self.config.epsilon)
            candidates[:, 1:] = proposal
            lengths[:, 1:] = score(proposal.reshape(-1, n, 2)).reshape(b, j)
            objective[:, 1:] = lengths[:, 1:] - lambdas[:, None] * costs(centers, proposal)
            rewards = objective[:, 1:] - objective[:, :1]
            if train:
                if self.method == "reinforce":
                    diagnostics.update(self._reinforce_update(x, ll, actions.reshape(b, j, n, 2), objective))
                elif self.method == "maddpg_bandit":
                    self._push_replay(xx.cpu().numpy(), proposal.reshape(-1, n, 2)-xx.cpu().numpy(),
                                      lam_flat.cpu().numpy(), rewards.reshape(-1))
                    diagnostics.update(self._ddpg_update(rng))
                else:
                    advantages = self._tensor(replacement_advantages(objective)).reshape(-1) if self.method == "bestof_ppo" else None
                    if advantages is not None:
                        diagnostics["replacement_advantage_mean"] = float(advantages.mean().cpu())
                        diagnostics["replacement_positive_fraction"] = float((advantages > 0).float().mean().cpu())
                    diagnostics.update(self._ppo_update(xx, torch.zeros_like(xx), lam_flat, context, actions, log_pin,
                                                         self._tensor(rewards).reshape(-1), advantages=advantages))
        index = objective.argmax(axis=1)
        selected = candidates[np.arange(b), index]
        diagnostics.update(method=self.method, k=int(k), scored_coordinates=int(b*k), scorer_batches=scorer_batches,
                           optimizer_updates=int(self.updates), training=bool(train),
                           center_objective_mean=float(objective[:, 0].mean()),
                           best_gain_mean=float((objective.max(1)-objective[:, 0]).mean()),
                           selection_noncenter_fraction=float((index != 0).mean()))
        return dict(selected=selected, selected_cost=costs(centers, selected), candidates=candidates,
                    lengths=lengths, objective=objective, index=index, diagnostics=diagnostics)

    def state_dict(self):
        # Snapshot owns its tensors/arrays, so later training cannot mutate it.
        return deepcopy(dict(format_version=1, method=self.method, degree=self.degree, config=asdict(self.config),
                             actor=self.actor.state_dict(), value=self.value.state_dict(),
                             target_actor=self.target_actor.state_dict(), target_value=self.target_value.state_dict(),
                             actor_optimizer=self.actor_optimizer.state_dict(), value_optimizer=self.value_optimizer.state_dict(),
                             replay=self.replay, updates=self.updates))

    def load_state_dict(self, state):
        if state.get("format_version") != 1 or state["method"] != self.method or state["degree"] != self.degree:
            raise ValueError("Checkpoint method/degree/format mismatch")
        if state["config"] != asdict(self.config):
            raise ValueError("Checkpoint config mismatch")
        for name in ("actor", "value", "target_actor", "target_value", "actor_optimizer", "value_optimizer"):
            getattr(self, name).load_state_dict(deepcopy(state[name]))
        self.replay = deepcopy(state["replay"])
        self.updates = int(state["updates"])
