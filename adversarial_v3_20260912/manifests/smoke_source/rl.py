"""One-step, score-function adversarial policy for the frozen v3 protocol.

The action is a Gaussian displacement before clipping and legalization. Its
log-probability is evaluated at a detached sample, so REINFORCE never needs an
input gradient through the real oracle or through the legality mapping. Rewards
are the original F values, with F(X) as an action-independent baseline: there is
no reward normalization, entropy bonus, or surrogate reward.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn


DEFAULTS = {
    "updates": 256, "batch_size": 64, "learning_rate": 3e-4,
    "initial_std": 0.02, "maximum_motion": 0.08,
    "validation_interval": 32, "validation_count": 128, "hidden_size": 64,
}


def _common():
    from . import common
    return common


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _array_digest(value):
    array = np.ascontiguousarray(value, dtype="<f4")
    return hashlib.sha256(str(array.shape).encode() + array.tobytes()).hexdigest()


class Policy(nn.Module):
    """Permutation-equivariant per-pin mean and learned coordinate noise scale."""

    def __init__(self, degree, hidden_size=64, initial_std=0.02, maximum_motion=0.08):
        super().__init__()
        if degree < 2 or hidden_size < 1 or initial_std <= 0 or maximum_motion <= 0:
            raise ValueError("invalid policy geometry or scale")
        self.degree = int(degree)
        self.hidden_size = int(hidden_size)
        self.initial_std = float(initial_std)
        self.maximum_motion = float(maximum_motion)
        self.encoder = nn.Sequential(nn.Linear(3, hidden_size), nn.Tanh(),
                                     nn.Linear(hidden_size, hidden_size), nn.Tanh())
        self.decoder = nn.Sequential(nn.Linear(3 * hidden_size, hidden_size), nn.Tanh(),
                                     nn.Linear(hidden_size, 2))
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)
        self.log_std = nn.Parameter(torch.full((2,), math.log(initial_std)))

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, centers, price):
        """Return Gaussian parameters [B,n,2], conditional on X and lambda."""
        x = torch.as_tensor(centers, dtype=torch.float32, device=self.device)
        if x.ndim != 3 or x.shape[1:] != (self.degree, 2):
            raise ValueError("expected centers [B,degree,2]")
        p = torch.as_tensor(price, dtype=x.dtype, device=x.device)
        if p.numel() == 1:
            p = p.reshape(1, 1, 1).expand(len(x), self.degree, 1)
        elif p.numel() == len(x):
            p = p.reshape(len(x), 1, 1).expand(-1, self.degree, 1)
        else:
            raise ValueError("price must be scalar or per-center")
        if not torch.isfinite(x).all() or not torch.isfinite(p).all() or (p < 0).any():
            raise ValueError("nonfinite centers or invalid nonnegative price")
        h = self.encoder(torch.cat((x, p), dim=-1))
        pooled = torch.cat((h.mean(dim=1, keepdim=True), h.amax(dim=1, keepdim=True)), -1)
        mean = self.decoder(torch.cat((h, pooled.expand(-1, self.degree, -1)), -1))
        std = self.log_std.exp().expand_as(mean)
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
            raise RuntimeError("nonfinite Gaussian policy")
        return mean, std

    def sample(self, centers, price, count=1, generator=None):
        """Return clipped displacements and *unclipped action* log-probabilities.

        Clipping is part of the deterministic environment map. Taking the log
        density at the clipped displacement would be incorrect, since clipping
        puts probability mass at the boundary.
        """
        if not isinstance(count, int) or count < 1:
            raise ValueError("count must be a positive integer")
        mean, std = self(centers, price)
        loc = mean[:, None].expand(-1, count, -1, -1)
        scale = std[:, None].expand_as(loc)
        noise = torch.randn(loc.shape, device=loc.device, dtype=loc.dtype, generator=generator)
        action = loc.detach() + scale.detach() * noise
        log_prob = torch.distributions.Normal(loc, scale).log_prob(action).sum(dim=(-1, -2))
        return action.clamp(-self.maximum_motion, self.maximum_motion), log_prob


def propose(policy, centers, price, count, seed):
    """Generate exactly count candidates per X; no X insertion or real scoring."""
    generator = torch.Generator(device=policy.device).manual_seed(int(seed))
    was_training = policy.training
    policy.eval()
    try:
        with torch.no_grad():
            delta, _ = policy.sample(centers, price, count=count, generator=generator)
        return np.asarray(_common().legalize(np.asarray(centers, dtype=np.float32),
                                           delta.cpu().numpy()), dtype=np.float32)
    finally:
        policy.train(was_training)


def online_update(policy, optimizer, oracle, centers, price):
    """One on-policy update, including B explicit F(X) baseline requests."""
    x = np.asarray(centers, dtype=np.float32)
    if len(x) == 0:
        raise ValueError("empty policy training batch")
    policy.train()
    delta, log_prob = policy.sample(x, price, count=1)
    z = _common().legalize(x, delta[:, 0].detach().cpu().numpy())
    evaluation = oracle.evaluate(np.concatenate((x, z), axis=0))
    regrets = np.asarray(evaluation["regret"], dtype=np.float64)
    if regrets.shape != (2 * len(x),) or not np.isfinite(regrets).all():
        raise RuntimeError("invalid oracle regret in policy update")
    costs = np.asarray(_common().costs(x, np.asarray(z)[:, None]), dtype=np.float64)[:, 0]
    reward = regrets[len(x):] - float(price) * costs
    baseline = regrets[:len(x)]
    advantage = torch.as_tensor(reward - baseline, dtype=torch.float32, device=policy.device)
    loss = -(advantage.detach() * log_prob[:, 0]).mean()
    if not torch.isfinite(loss):
        raise RuntimeError("nonfinite REINFORCE loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradients = [p.grad for p in policy.parameters() if p.grad is not None]
    if not gradients or not all(torch.isfinite(g).all() for g in gradients):
        raise RuntimeError("missing or nonfinite policy gradients")
    gradient_norm = float(torch.sqrt(sum(g.detach().square().sum() for g in gradients)).cpu())
    optimizer.step()
    if not all(torch.isfinite(p).all() for p in policy.parameters()):
        raise RuntimeError("nonfinite updated policy parameters")
    return {
        "loss": float(loss.detach().cpu()), "gradient_norm": gradient_norm,
        "mean_f": float(reward.mean()), "mean_baseline_f": float(baseline.mean()),
        "mean_advantage": float((reward - baseline).mean()),
        "mean_regret": float(regrets[len(x):].mean()), "mean_cost": float(costs.mean()),
        "policy_std": policy.log_std.detach().exp().cpu().tolist(),
        "baseline_requests": len(x), "candidate_requests": len(x),
        "real_score_requests": 2 * len(x), "policy_updates": 1,
        "centers_sha256": _array_digest(x), "candidates_sha256": _array_digest(z),
        "reward_normalization": "none", "baseline": "real_f_of_original_X",
    }


def _rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def _restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([v.cpu() for v in state["cuda"]])


def _load(path, device):
    return torch.load(path, map_location=device, weights_only=False)


def _validation(policy, oracle, centers, price, seed):
    proposed = propose(policy, centers, price, count=1, seed=seed)[:, 0]
    evaluation = oracle.evaluate(np.concatenate((centers, proposed), axis=0))
    regret = np.asarray(evaluation["regret"], dtype=np.float64)
    cost = np.asarray(_common().costs(centers, proposed[:, None]))[:, 0]
    if regret.shape != (2 * len(centers),) or not np.isfinite(regret).all():
        raise RuntimeError("invalid validation regret")
    f = regret[len(centers):] - price * cost
    return {"mean_f": float(f.mean()), "mean_baseline_f": float(regret[:len(centers)].mean()),
            "mean_delta_f": float((f - regret[:len(centers)]).mean()),
            "real_score_requests": 2 * len(centers), "baseline_requests": len(centers),
            "candidate_requests": len(centers), "centers_sha256": _array_digest(centers)}


def train_policy(oracle, degree, seed, price, outdir, config, resume=True, heartbeat=None):
    """Train/resume the 256-update policy; return validation-best policy, summary.

    ``config`` overrides DEFAULTS for explicitly authorized smoke tests. Optional
    ``stop_after`` limits this invocation, but is excluded from run identity so
    an interrupted invocation can resume. Other extra config keys are included
    in identity, allowing the caller to bind a parent checkpoint/protocol hash.
    Oracle.ledger is snapshotted but never restored or rolled back here.
    """
    common = _common()
    settings = {**DEFAULTS, **dict(config)}
    stop_after = min(int(settings.pop("stop_after", settings["updates"])), int(settings["updates"]))
    if min(int(settings[k]) for k in ("updates", "batch_size", "validation_interval", "validation_count")) < 1:
        raise ValueError("policy update and validation counts must be positive")
    if stop_after < 0 or settings["learning_rate"] <= 0:
        raise ValueError("invalid stop_after or learning rate")
    identity = {"schema_version": 1, "algorithm": "v3_one_step_reinforce_raw_f_baseline_X",
                "degree": int(degree), "seed": int(seed), "price": float(price), "config": settings}
    identity_hash = _digest(identity)
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    latest_path, best_path = out / "latest.pt", out / "best.pt"
    log_path = out / "training.jsonl"
    random.seed(seed)
    np.random.seed(int(seed) % (2**32))
    torch.manual_seed(int(seed))
    policy = Policy(degree, hidden_size=settings["hidden_size"], initial_std=settings["initial_std"],
                    maximum_motion=settings["maximum_motion"]).to(oracle.device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=float(settings["learning_rate"]))
    cursor, best_update, best_f = 0, None, None
    counts = {"training_requests": 0, "validation_requests": 0,
              "baseline_requests": 0, "candidate_requests": 0}
    validation_history = []
    training_records = []
    elapsed = 0.0
    if latest_path.exists():
        if not resume:
            raise FileExistsError(f"policy checkpoint exists: {latest_path}")
        saved = _load(latest_path, oracle.device)
        if saved["identity_sha256"] != identity_hash or saved["identity"] != identity:
            raise ValueError("policy checkpoint run identity mismatch")
        policy.load_state_dict(saved["policy_state_dict"], strict=True)
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        cursor, best_update, best_f = saved["update"], saved["best_update"], saved["best_f"]
        counts = dict(saved["counts"])
        validation_history = list(saved["validation_history"])
        training_records = list(saved["training_records"])
        elapsed = float(saved["training_wall_seconds"])
        _restore_rng(saved["rng_state"])
    elif best_path.exists() or log_path.exists():
        raise FileExistsError("policy output has artifacts without latest checkpoint")

    namespace = f"adv_v3_rl_d{degree}_seed{seed}"
    validation = common.make_data(degree, int(settings["validation_count"]), "uniform",
                                  namespace + "_validation", seed=int(seed))
    started = time.monotonic()

    def publish_log():
        # The checkpoint contains every committed record. Rebuilding this small
        # JSONL also repairs a crash after checkpoint commit but before log write.
        temporary = log_path.with_name(log_path.name + ".partial")
        with temporary.open("w", encoding="utf-8") as handle:
            for record in training_records:
                handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
        temporary.replace(log_path)

    def state():
        return {"identity": identity, "identity_sha256": identity_hash, "update": cursor,
                "policy_state_dict": policy.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "rng_state": _rng_state(), "best_update": best_update, "best_f": best_f,
                "counts": dict(counts), "validation_history": list(validation_history),
                "training_records": list(training_records),
                "validation_centers_sha256": _array_digest(validation), "data_namespace": namespace,
                "training_wall_seconds": elapsed + time.monotonic() - started,
                "oracle_ledger_snapshot": dict(oracle.ledger)}

    if not latest_path.exists():
        common.atomic_torch(latest_path, state())
    publish_log()
    # Repair an interrupted best write before later updates can replace latest.
    if best_update == cursor and (not best_path.exists() or _load(best_path, oracle.device)["update"] != best_update):
        common.atomic_torch(best_path, _load(latest_path, oracle.device))
    for update in range(cursor + 1, stop_after + 1):
        centers = common.make_data(degree, int(settings["batch_size"]), "uniform",
                                   namespace + f"_train_update{update:06d}", seed=int(seed))
        detail = online_update(policy, optimizer, oracle, centers, price)
        counts["training_requests"] += detail["real_score_requests"]
        counts["baseline_requests"] += detail["baseline_requests"]
        counts["candidate_requests"] += detail["candidate_requests"]
        cursor = update
        is_best = False
        if update % int(settings["validation_interval"]) == 0 or update == int(settings["updates"]):
            val = _validation(policy, oracle, validation, float(price), int(seed) + 104729)
            counts["validation_requests"] += val["real_score_requests"]
            counts["baseline_requests"] += val["baseline_requests"]
            counts["candidate_requests"] += val["candidate_requests"]
            validation_history.append({"update": update, **val})
            detail["validation"] = val
            if best_f is None or val["mean_f"] > best_f:
                best_f, best_update, is_best = val["mean_f"], update, True
        detail.update({"update": update, "identity_sha256": identity_hash,
                       "counts": dict(counts), "oracle_ledger_snapshot": dict(oracle.ledger)})
        training_records.append(detail)
        # Checkpoints commit the update first. On a crash between the two writes,
        # best can be reconstructed from latest when it contains the best update.
        checkpoint = state()
        common.atomic_torch(latest_path, checkpoint)
        if is_best:
            common.atomic_torch(best_path, checkpoint)
        publish_log()
        if heartbeat is not None:
            heartbeat({"phase": "rl_train", "update": update, "total_updates": int(settings["updates"]),
                       "best_update": best_update, "counts": dict(counts)})

    # Recover a best-file commit interrupted after the authoritative latest save.
    if best_update == cursor and (not best_path.exists() or _load(best_path, oracle.device)["update"] != best_update):
        common.atomic_torch(best_path, _load(latest_path, oracle.device))
    if best_path.exists():
        best = _load(best_path, oracle.device)
        if best["identity_sha256"] != identity_hash or best["update"] != best_update:
            raise ValueError("best policy checkpoint identity/update mismatch")
        policy.load_state_dict(best["policy_state_dict"], strict=True)
    policy.eval()
    summary = {"status": "complete" if cursor == int(settings["updates"]) else "partial",
               "identity": identity, "identity_sha256": identity_hash, "update": cursor,
               "best_update": best_update, "best_f": best_f, "counts": counts,
               "validation_history": validation_history, "validation_centers_sha256": _array_digest(validation),
               "data_namespace": namespace, "training_wall_seconds": elapsed + time.monotonic() - started,
               "oracle_ledger_snapshot": dict(oracle.ledger), "latest_checkpoint": str(latest_path),
               "best_checkpoint": str(best_path) if best_path.exists() else None,
               "reward_normalization": "none", "baseline": "real_f_of_original_X"}
    common.atomic_json(out / "summary.json", summary)
    return policy, summary
