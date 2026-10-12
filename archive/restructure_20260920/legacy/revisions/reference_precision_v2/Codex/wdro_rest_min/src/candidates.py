"""Shared Random/Sobol support and exact scoring for Hard Mining / future WDRO.

No transport cost, lambda update, or WDRO trainer is implemented here.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.stats import qmc

from .datasets import array_bundle_sha256, normalize_fixed_degree_batch
from .geometry import pin_set_sha256, project_normalized_net, normalize_net, has_spacing_violation
from .metrics import relative_regret


@dataclass
class CandidateSupport:
    pins: np.ndarray
    sha256: str
    generated: int
    repaired: int
    refresh_block: int


def generate_support(batch, *, training_seed, update, search):
    centers = batch.pins
    count, degree, _ = centers.shape
    k = search["candidate_count"]
    block = (update - 1) // search["refresh_updates"]
    sobol_seed = int.from_bytes(hashlib.sha256(f"support-v1:{training_seed}:{degree}:{block}".encode()).digest()[:4], "little")
    base = qmc.Sobol(d=degree * 2 + 2, scramble=True, seed=sobol_seed).random_base2(int(np.ceil(np.log2(k))))[:k]
    raw = np.empty((count, k, degree, 2), dtype=np.float64)
    for i, (center, net_id) in enumerate(zip(centers, batch.net_ids, strict=True)):
        digest = hashlib.sha256(bytes(net_id) + str(block).encode()).digest()
        shift_rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        u = (base + shift_rng.random(degree * 2 + 2)) % 1.0
        noise = 2 * u[:, :degree * 2].reshape(k, degree, 2) - 1
        order = np.lexsort((center[:, 1], center[:, 0]))
        canonical = center[order].astype(np.float64)
        for j in range(k):
            family = j % 4
            amplitude = search["amplitudes"][(j // 4) % len(search["amplitudes"])]
            motion = noise[j].copy()
            if family == 1:
                mask = np.abs(motion[:, 0]) >= np.quantile(np.abs(motion[:, 0]), 0.75)
                motion *= mask[:, None]
            elif family == 2:
                anchor = canonical[int(u[j, -2] * degree)]
                envelope = np.exp(-np.square(canonical - anchor).sum(axis=1) / (2 * 0.2**2))
                motion = envelope[:, None] * motion[0]
            elif family == 3:
                motion[:, 1 if u[j, -1] < 0.5 else 0] *= 0.1
            moved = np.clip(canonical + amplitude * motion, 0.1, 0.9)
            raw[i, j, order] = moved
    flat = raw.reshape(-1, degree, 2)
    normalized, _, _, valid, _ = normalize_fixed_degree_batch(flat)
    repaired = int((~valid).sum())
    for i in np.flatnonzero(~valid):
        projected = project_normalized_net(flat[i], minimum_l1_spacing=2e-6)
        normalized[i], _ = normalize_net(projected, expected_degree=degree)
    normalized = normalized.astype(np.float32)
    # Check storage precision; repair with extra spacing before conversion.
    for i in range(len(normalized)):
        if has_spacing_violation(normalized[i]):
            normalized[i] = normalize_net(project_normalized_net(normalized[i], minimum_l1_spacing=2e-6), expected_degree=degree)[0]
            repaired += 1
    pins = np.concatenate([centers[:, None], normalized.reshape(count, k, degree, 2)], axis=1)
    # Do not silently accept duplicate support: regenerate deterministically with
    # a tiny legal perturbation while retaining the exact original at index zero.
    for i in range(count):
        seen = {pin_set_sha256(pins[i, 0])}
        for j in range(1, k + 1):
            key = pin_set_sha256(pins[i, j])
            for attempt in range(16):
                if key not in seen:
                    break
                candidate = pins[i, j].astype(np.float64)
                order = np.lexsort((candidate[:, 1], candidate[:, 0]))
                candidate[order[(j + attempt) % degree], attempt % 2] += (attempt + 1) * 1e-4
                candidate = project_normalized_net(candidate, minimum_l1_spacing=2e-6)
                pins[i, j] = normalize_net(candidate, expected_degree=degree)[0]
                key = pin_set_sha256(pins[i, j])
                repaired += 1
            if key in seen:
                raise RuntimeError("failed to produce distinct legal candidate support")
            seen.add(key)
    return CandidateSupport(pins, array_bundle_sha256({"pins": pins, "center_ids": batch.net_ids}), count*k, repaired, block)


def fixed_surrogate(pins):
    """Vectorized Prim L1 MST / HPWL proxy, independent of all model metrics."""
    nets = np.asarray(pins, dtype=np.float64)
    n, degree, _ = nets.shape
    distance = np.full((n, degree), np.inf)
    visited = np.zeros((n, degree), dtype=bool)
    current = np.zeros(n, dtype=np.int64)
    total = np.zeros(n)
    rows = np.arange(n)
    for _ in range(degree - 1):
        visited[rows, current] = True
        edge = np.abs(nets - nets[rows, current, None]).sum(axis=2)
        distance = np.minimum(distance, edge)
        distance[visited] = np.inf
        current = distance.argmin(axis=1)
        total += distance[rows, current]
    hpwl = np.ptp(nets, axis=1).sum(axis=1)
    return total / hpwl - 1.0


def shortlist_support(support, top_k=4):
    b, k, degree, _ = support.pins.shape
    scores = fixed_surrogate(support.pins.reshape(-1, degree, 2)).reshape(b, k)
    # Nominal consumes one of the four calls so the support always includes X.
    ranked = np.argsort(-scores[:, 1:], axis=1, kind="stable")[:, :top_k-1] + 1
    indices = np.concatenate([np.zeros((b, 1), dtype=np.int64), ranked], axis=1)
    return indices, scores


class ReferenceCache:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS lstar (hash TEXT PRIMARY KEY, length REAL NOT NULL)")
        self.db.execute("CREATE TABLE IF NOT EXISTS calls (id INTEGER PRIMARY KEY, requested INTEGER, misses_started INTEGER, misses_completed INTEGER)")
        from .geosteiner import reference_identity
        import json
        identity=json.dumps(reference_identity(),sort_keys=True)
        self.db.execute("CREATE TABLE IF NOT EXISTS reference_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        old=self.db.execute("SELECT value FROM reference_metadata WHERE key='identity'").fetchone()
        if old is None:
            if self.db.execute("SELECT COUNT(*) FROM lstar").fetchone()[0]:
                raise ValueError("unversioned nonempty reference cache cannot be reused")
            with self.db:self.db.execute("INSERT INTO reference_metadata VALUES ('identity',?)",(identity,))
        elif old[0]!=identity:
            raise ValueError("reference cache solver identity mismatch")

    def lengths(self, pins, evaluator):
        keys = [pin_set_sha256(net) for net in pins]
        unique = dict(zip(keys, pins))
        values = {}
        missing = []
        for key in unique:
            row = self.db.execute("SELECT length FROM lstar WHERE hash=?", (key,)).fetchone()
            if row is None:
                missing.append(key)
            else:
                values[key] = row[0]
        with self.db:
            receipt = self.db.execute("INSERT INTO calls (requested,misses_started,misses_completed) VALUES (?,?,0)", (len(pins),len(missing))).lastrowid
        if missing:
            exact = evaluator.lengths(np.asarray([unique[key] for key in missing]))
            if not np.isfinite(exact).all() or (exact <= 0).any():
                raise RuntimeError("invalid exact reference")
            values.update(zip(missing, map(float, exact)))
            with self.db:
                self.db.executemany("INSERT INTO lstar VALUES (?,?)", [(key, values[key]) for key in missing])
                self.db.execute("UPDATE calls SET misses_completed=? WHERE id=?", (len(missing),receipt))
        return np.asarray([values[key] for key in keys]), len(missing)

    def close(self):
        self.db.close()


def exact_score_support(support, *, actor, rest, geosteiner, cache, ledger, search):
    indices, surrogate = shortlist_support(support, search["exact_rescore_top_k"])
    b, _, degree, _ = support.pins.shape
    selected = support.pins[np.arange(b)[:, None], indices]
    flat = selected.reshape(-1, degree, 2)
    outputs = []
    was_training = actor.training
    actor.eval()
    with torch.no_grad():
        for start in range(0, len(flat), search["score_microbatch"]):
            out, _ = actor(flat[start:start+search["score_microbatch"]], True)
            outputs.append(out.cpu().numpy())
    actor.train(was_training)
    lengths = rest.lengths(flat, np.concatenate(outputs))
    lstar, misses = cache.lengths(flat, geosteiner)
    ledger.add("surrogate_evaluator_nets", int(np.prod(support.pins.shape[:2])))
    ledger.add("actor_forward_nets", len(flat))
    ledger.add("exact_rest_evaluator_nets", len(flat))
    ledger.add("geosteiner_nets", misses)
    regrets = relative_regret(lengths, lstar).reshape(b, -1)
    return indices, regrets, lstar.reshape(b, -1), {
        "centers": b, "perturbations": support.generated,
        "surrogate_nets": int(np.prod(support.pins.shape[:2])),
        "exact_rescore_nets": len(flat), "reference_requests": len(flat),
        "reference_cache_misses": misses, "reference_cache_hits": len(flat)-misses,
        "repaired_candidates": support.repaired, "support_sha256": support.sha256,
        "refresh_block": support.refresh_block,
        "shortlist_sha256": array_bundle_sha256({"indices": indices}),
        "exact_shortlist_indices": indices.tolist(),
        "exact_relative_regrets": regrets.tolist(),
    }


def select_hard_mining(indices, true_regrets):
    if indices.shape != true_regrets.shape or not np.isfinite(true_regrets).all():
        raise ValueError("invalid per-center exact scores")
    choice = true_regrets.argmax(axis=1)
    return indices[np.arange(len(indices)), choice], choice
