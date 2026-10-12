"""Shared Random/Sobol support and exact scoring for Hard Mining / future WDRO.

No transport cost, lambda update, or WDRO trainer is implemented here.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.stats import qmc

from src.datasets import normalize_fixed_degree_batch
from src.geometry import project_normalized_net, normalize_net, has_spacing_violation
from src.metrics import relative_regret


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
    sobol_seed = int(np.random.SeedSequence([20261011,training_seed,degree,block]).generate_state(1)[0])
    base = qmc.Sobol(d=degree * 2 + 2, scramble=True, seed=sobol_seed).random_base2(int(np.ceil(np.log2(k))))[:k]
    raw = np.empty((count, k, degree, 2), dtype=np.float64)
    for i, (center, net_id) in enumerate(zip(centers, batch.net_ids, strict=True)):
        shift_rng = np.random.default_rng(np.random.SeedSequence([20261011,training_seed,update,i,block]))
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
        seen = [canonical_points(pins[i, 0])]
        for j in range(1, k + 1):
            key = canonical_points(pins[i, j])
            for attempt in range(16):
                if key not in seen:
                    break
                candidate = pins[i, j].astype(np.float64)
                order = np.lexsort((candidate[:, 1], candidate[:, 0]))
                candidate[order[(j + attempt) % degree], attempt % 2] += (attempt + 1) * 1e-4
                candidate = project_normalized_net(candidate, minimum_l1_spacing=2e-6)
                pins[i, j] = normalize_net(candidate, expected_degree=degree)[0]
                key = canonical_points(pins[i, j])
                repaired += 1
            if key in seen:
                raise RuntimeError("failed to produce distinct legal candidate support")
            seen.append(key)
    return CandidateSupport(pins, "not_computed", count*k, repaired, block)



def canonical_points(points):
 order=np.lexsort((points[:,1],points[:,0]))
 return tuple(map(tuple,points[order].tolist()))

def materialize_training_batch(*,protocol,degree,training_seed,update,batch_size):
 from types import SimpleNamespace
 from src.datasets import generate_terminals
 accepted=[];attempt=0
 while sum(len(a) for a in accepted)<batch_size:
  seed=np.random.SeedSequence([20261011,training_seed,update,degree,attempt])
  raw=generate_terminals(batch_size,degree,seed,'uniform',{'nonuniform':0.0})
  pins,_,_,valid,_=normalize_fixed_degree_batch(raw)
  accepted.append(pins[valid]);attempt+=1
  if attempt>100:raise RuntimeError('unable to generate legal training data')
 pins=np.concatenate(accepted)[:batch_size].astype(np.float32)
 ids=np.asarray([f'bank20261011:s{training_seed}:u{update}:n{i}'.encode() for i in range(batch_size)])
 return SimpleNamespace(pins=pins,net_ids=ids)
