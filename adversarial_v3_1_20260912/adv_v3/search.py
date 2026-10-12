"""Budgeted, true-score adversarial search for the isolated v3 experiment.

The Oracle owns all evaluator accounting. Every requested candidate, including a
repeat, is sent to it. Repeated ES points are never changed to force uniqueness.
"""
from __future__ import annotations

import numpy as np

from .common import costs, derived_seed, legalize
from src.candidates import generate_support, shortlist_support
from src.datasets import array_bundle_sha256
from src.geometry import has_spacing_violation, pin_set_sha256
from src.training_data import TrainingBatch
from stage5.search import adapt_cem, cem_round


BUDGETS = {
    "random_top4": 4,
    "random64": 64,
    "cem64": 64,
    "es64": 64,
    "critic_raw4": 4,
    "critic_calibrated4": 4,
    "critic_calibrated64": 64,
    "rl4": 4,
    "rl64": 64,
}

_SEARCH = {
    "candidate_count": 32,
    "refresh_updates": 10,
    "amplitudes": [0.01, 0.02, 0.04, 0.08],
    "ground_cost": "hungarian_l1_mean",
    "cem": {
        "initial_std": 0.02,
        "minimum_std": 0.001,
        "maximum_motion": 0.08,
    },
}


def antithetic_gradient(positive, negative, directions, mu):
    """Estimate the Gaussian-smoothed gradient from true paired scores.

    directions is (batch, pairs, degree, 2). Clipping or repairing the queried
    points belongs inside the fixed function being smoothed, not in directions.
    """
    positive = np.asarray(positive, dtype=np.float64)
    negative = np.asarray(negative, dtype=np.float64)
    directions = np.asarray(directions, dtype=np.float64)
    if directions.ndim != 4 or directions.shape[-1] != 2:
        raise ValueError("directions must have shape (batch,pairs,degree,2)")
    if positive.shape != directions.shape[:2] or negative.shape != positive.shape:
        raise ValueError("paired scores must match batch and direction counts")
    if directions.shape[1] == 0 or not np.isfinite(mu) or mu <= 0:
        raise ValueError("positive smoothing scale and at least one pair required")
    if not all(np.isfinite(a).all() for a in (positive, negative, directions)):
        raise ValueError("nonfinite paired gradient inputs")
    return np.einsum("bm,bmnc->bnc", positive - negative, directions) / (
        2.0 * directions.shape[1] * mu
    )


def _check_pins(pins, degree):
    if pins.ndim != 4 or pins.shape[2:] != (degree, 2):
        raise ValueError("candidate shape must be (batch,count,degree,2)")
    if not np.isfinite(pins).all():
        raise ValueError("nonfinite search candidate")
    flat = pins.reshape(-1, degree, 2)
    if np.any(flat < np.float32(0.1)) or np.any(flat > np.float32(0.9)):
        raise ValueError("search candidate outside normalized bounds")
    # Tolerance covers the frozen float32 storage contract, not a second
    # normalization: original centers and proposed legal points stay unchanged.
    if not np.allclose(flat.min(axis=1), 0.1, rtol=0, atol=3e-7):
        raise ValueError("search candidate has noncanonical bbox origin")
    if not np.allclose(np.ptp(flat, axis=1).max(axis=1), 0.8, rtol=0, atol=3e-7):
        raise ValueError("search candidate has noncanonical bbox span")
    if any(has_spacing_violation(net) for net in flat):
        raise ValueError("search candidate violates minimum pin spacing")


def _batch(centers):
    count = len(centers)
    net_ids = np.asarray([pin_set_sha256(x).encode("ascii") for x in centers], dtype="S64")
    return TrainingBatch(
        pins=centers,
        net_ids=net_ids,
        offsets=np.zeros((count, 2), dtype=np.float64),
        scales=np.ones(count, dtype=np.float64),
        metadata={"source": "adversarial_v3_content_addressed_search"},
        bundle_sha256=array_bundle_sha256({"pins": centers, "net_ids": net_ids}),
    )


class _History:
    def __init__(self, centers, oracle, price):
        self.centers = centers
        self.oracle = oracle
        self.price = price
        self.chunks = []
        self.calls_per_center = 0

    def evaluate(self, pins):
        pins = np.asarray(pins, dtype=np.float32)
        b, n, _ = self.centers.shape
        _check_pins(pins, n)
        if len(pins) != b or pins.shape[1] == 0:
            raise ValueError("candidate batch must match nonempty center batch")
        k = pins.shape[1]
        true = self.oracle.evaluate(pins.reshape(-1, n, 2))
        record = {"pins": pins.copy()}
        for name in ("regret", "lstar", "length"):
            value = np.asarray(true[name], dtype=np.float64)
            if value.shape != (b * k,) or not np.isfinite(value).all():
                raise ValueError(f"Oracle returned invalid {name}")
            record[name] = value.reshape(b, k)
        if np.any(record["lstar"] <= 0):
            raise ValueError("Oracle reference length must be positive")
        record["cost"] = np.asarray(costs(self.centers, pins), dtype=np.float64)
        if record["cost"].shape != (b, k) or not np.isfinite(record["cost"]).all():
            raise ValueError("invalid transport costs")
        if np.any(record["cost"] < 0):
            raise ValueError("negative transport cost")
        record["score"] = record["regret"] - self.price * record["cost"]
        self.chunks.append(record)
        self.calls_per_center += k
        return record

    def finish(self, method, diagnostics):
        if self.calls_per_center != BUDGETS[method]:
            raise AssertionError("search violated the declared true-score budget")
        whole = {
            name: np.concatenate([part[name] for part in self.chunks], axis=1)
            for name in self.chunks[0]
        }
        np.testing.assert_array_equal(whole["pins"][:, 0], self.centers)
        np.testing.assert_array_equal(whole["cost"][:, 0], np.zeros(len(self.centers)))
        selected = whole["score"].argmax(axis=1)  # earliest candidate wins ties
        rows = np.arange(len(self.centers))
        result = {name: value[rows, selected] for name, value in whole.items()}
        if np.any(result["score"] < whole["score"][:, 0]):
            raise AssertionError("selected true score is worse than nominal")
        result.update(
            original_regret=whole["regret"][:, 0].copy(),
            original_score=whole["score"][:, 0].copy(),
            calls_per_center=self.calls_per_center,
            selected_index=selected,
            # In-memory labels for joint calibration; callers need not persist
            # this support in the fixed-model result artifact.
            evaluated_pins=whole["pins"],
            evaluated_regrets=whole["regret"],
            diagnostics={
                **diagnostics,
                "method": method,
                "legal_fraction": 1.0,
                "scoring_batches_per_center": [p["pins"].shape[1] for p in self.chunks],
                "nominal_repeat_count": int(np.all(
                    whole["pins"][:, 1:] == self.centers[:, None], axis=(2, 3)
                ).sum()),
            },
        )
        return result


def _es(centers, history, seed):
    b, n, _ = centers.shape
    mu, step, maximum_motion = 0.02, 0.01, 0.08
    pairs, rounds = 4, 7
    delta = np.zeros_like(centers, dtype=np.float64)
    orders = [np.lexsort((x[:, 1], x[:, 0])) for x in centers]
    identities = [pin_set_sha256(x) for x in centers]
    history.evaluate(centers[:, None])
    gradient_rms = []
    for round_index in range(rounds):
        directions = np.empty((b, pairs, n, 2), dtype=np.float64)
        for i, order in enumerate(orders):
            rng = np.random.default_rng(derived_seed("es64", seed, identities[i], round_index))
            canonical = rng.normal(size=(pairs, n, 2))
            directions[i][:, order, :] = canonical
        # Fixed clipped latent map: even if two arguments map to the same Z,
        # send both scoring requests and keep their common true function value.
        positive = np.clip(delta[:, None] + mu * directions, -maximum_motion, maximum_motion)
        negative = np.clip(delta[:, None] - mu * directions, -maximum_motion, maximum_motion)
        paired = legalize(centers, np.concatenate((positive, negative), axis=1))
        record = history.evaluate(paired)
        gradient = antithetic_gradient(record["score"][:, :pairs], record["score"][:, pairs:], directions, mu)
        rms = np.sqrt(np.mean(gradient * gradient, axis=(1, 2)))
        normalized = np.divide(gradient, rms[:, None, None], out=np.zeros_like(gradient), where=rms[:, None, None] > 0)
        delta = np.clip(delta + step * normalized, -maximum_motion, maximum_motion)
        history.evaluate(legalize(centers, delta)[:, None])
        gradient_rms.append(float(rms.mean()))
    return {
        "mu": mu,
        "step": step,
        "maximum_motion": maximum_motion,
        "pairs": pairs,
        "rounds": rounds,
        "mean_gradient_rms_by_round": gradient_rms,
        "history_dependent_duplicate_repair": False,
    }


def attack(method, centers, oracle, price, seed, proposer=None):
    """Return the true-score best legal candidate under the fixed v3 budget.

    A learned proposer receives (centers, price, count, seed) and must return
    legal candidate coordinates Z without the nominal center. Its own training
    calls are outside this search budget and must be accounted for by its owner.
    """
    if method not in BUDGETS:
        raise ValueError(f"unknown v3 search method: {method}")
    price = float(price)
    if not np.isfinite(price) or price < 0:
        raise ValueError("price must be finite and nonnegative")
    centers = np.asarray(centers, dtype=np.float32)
    if centers.ndim != 3 or len(centers) == 0 or centers.shape[1] < 2 or centers.shape[2] != 2:
        raise ValueError("centers must have shape (nonempty_batch,degree>=2,2)")
    _check_pins(centers[:, None], centers.shape[1])
    history = _History(centers, oracle, price)
    diagnostics = {}
    if method in ("random_top4", "random64"):
        config = dict(_SEARCH, candidate_count=32 if method == "random_top4" else 63)
        support = generate_support(_batch(centers), training_seed=seed, update=1, search=config)
        pins = support.pins
        if method == "random_top4":
            indices, _ = shortlist_support(support, top_k=4)
            pins = pins[np.arange(len(centers))[:, None], indices]
            diagnostics["geometric_surrogate_candidates_per_center"] = 33
        history.evaluate(pins)
        diagnostics["repaired_candidates"] = support.repaired
    elif method == "cem64":
        batch = _batch(centers)
        mean = np.zeros_like(centers, dtype=np.float64)
        std = np.full_like(mean, 0.02)
        seen = [set() for _ in centers]
        repaired = 0
        for round_index in range(4):
            pins, count = cem_round(
                batch, update=1, training_seed=seed, round_index=round_index,
                mean=mean, std=std, seen=seen, config=_SEARCH,
            )
            record = history.evaluate(pins)
            # rho is unused by the parent's proposal adaptation. No per-net
            # hard radius is introduced: selection remains regret-price*cost.
            mean, std, _ = adapt_cem(batch, pins, record["regret"], price, 1.0, _SEARCH)
            repaired += count
        diagnostics.update(repaired_candidates=repaired, population=16, rounds=4, elite=4)
    elif method == "es64":
        diagnostics.update(_es(centers, history, seed))
    else:
        if proposer is None:
            raise ValueError(f"{method} requires a learned proposer")
        count = BUDGETS[method] - 1
        proposals = np.asarray(proposer(centers, price, count, seed), dtype=np.float32)
        if proposals.shape != (len(centers), count, centers.shape[1], 2):
            raise ValueError("learned proposer returned an incorrect candidate shape")
        history.evaluate(np.concatenate((centers[:, None], proposals), axis=1))
    return history.finish(method, diagnostics)
