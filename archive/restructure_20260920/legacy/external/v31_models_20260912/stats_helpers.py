#!/usr/bin/env python3
"""Frozen-model external statistics with paired design-family cluster CIs.

All metrics are computed separately for each training seed before averaging
the three seed metrics. Bootstrap draws resample design families, keeping
every net, design variant, training seed and paired method in a family
together. Point estimates retain the declared net/design weighting; families
are the resampling units, not substituted point-estimate weights.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import sys
import time

import numpy as np

import evaluate as ev

BASE = Path(__file__).resolve().parent
METHODS = ('erm', 'mixed_erm', 'hard_mining', 'group_dro', 'wdro_rest')
SEEDS = (101, 102, 103)
DEGREES = (10, 20, 30, 40, 50)
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 20260912


def string_vector(values):
    values = np.asarray(values)
    return np.char.decode(values, 'utf-8') if values.dtype.kind == 'S' else values.astype(str)


def metrics(values):
    values = np.asarray(values, dtype=np.float64)
    ev.require(values.ndim == 1 and np.isfinite(values).all(), 'invalid regret vector')
    if not len(values):
        return dict(count=0, mean=None, p95=None, minimum=None, maximum=None)
    return dict(count=len(values), mean=float(values.mean()), p95=float(np.quantile(values, .95)),
                minimum=float(values.min()), maximum=float(values.max()))


def pooled_metrics(values, designs, families):
    row = metrics(values)
    distinct = sorted(set(designs.tolist()))
    means = {design: float(values[designs == design].mean()) for design in distinct}
    worst = max(means, key=means.get) if means else None
    return dict(count=row['count'], net_weighted_mean=row['mean'], p95=row['p95'],
        design_count=len(distinct), family_count=len(set(families.tolist())),
        design_macro_mean=float(np.mean(list(means.values()))) if means else None,
        worst_design_mean=means[worst] if worst is not None else None, worst_design_id=worst)


def family_bootstrap(differences, designs, families):
    """Return per-seed and seed-mean paired CIs for two declared estimands.

    `differences` has shape (3, N); each column identifies the same net and
    design for all seeds. A draw consists of F families with replacement.
    If a family has several PDK variants, every variant gets its family's
    multiplicity. Mean-of-seed metrics is recomputed within each draw.
    """
    difference = np.asarray(differences, dtype=np.float64)
    designs, families = np.asarray(designs), np.asarray(families)
    ev.require(difference.shape == (len(SEEDS), len(designs)), 'paired difference shape mismatch')
    ev.require(families.shape == designs.shape and len(designs) > 0, 'empty/misaligned cluster labels')
    ev.require(np.isfinite(difference).all(), 'nonfinite paired differences')
    cluster_names, cluster = np.unique(families, return_inverse=True)
    count = len(cluster_names)
    net_count = np.bincount(cluster, minlength=count).astype(float)
    net_sums = np.array([np.bincount(cluster, weights=row, minlength=count) for row in difference])
    design_count = np.zeros(count)
    design_sums = np.zeros((len(SEEDS), count))
    for design in sorted(set(designs.tolist())):
        mask = designs == design
        family = np.unique(cluster[mask])
        ev.require(len(family) == 1, f'design spans several families: {design}')
        index = family[0]
        design_count[index] += 1
        design_sums[:, index] += difference[:, mask].mean(axis=1)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, count, size=(BOOTSTRAP_SAMPLES, count))
    multiplicity = np.zeros((BOOTSTRAP_SAMPLES, count), dtype=np.int64)
    np.add.at(multiplicity, (np.arange(BOOTSTRAP_SAMPLES)[:, None], indices), 1)
    rows = []
    for metric, numerators, denominators in (
        ('net_weighted_mean', net_sums, net_count),
        ('design_macro_mean', design_sums, design_count),
    ):
        point = numerators.sum(axis=1) / denominators.sum()
        resampled = (multiplicity @ numerators.T) / (multiplicity @ denominators)[:, None]
        points = [*point, float(point.mean())]
        bootstraps = [*(resampled[:, i] for i in range(len(SEEDS))), resampled.mean(axis=1)]
        for seed, estimate, replicates in zip((*SEEDS, 'mean'), points, bootstraps, strict=True):
            lo, hi = np.quantile(replicates, [.025, .975])
            rows.append(dict(metric=metric, training_seed=seed, mean_difference=float(estimate),
                ci_low=float(lo), ci_high=float(hi), confidence_level=.95,
                bootstrap_samples=BOOTSTRAP_SAMPLES, random_seed=BOOTSTRAP_SEED,
                count=len(designs), design_count=int(design_count.sum()), family_count=count,
                cluster_unit='design_family', interval_status='single_family_degenerate' if count == 1 else 'cluster_percentile',
                seed_aggregation='arithmetic_mean_of_three_seed_metrics' if seed == 'mean' else 'individual_training_seed'))
    return rows


def write_csv(path, rows):
    ev.require(bool(rows), f'empty table: {path}')
    fields = list(rows[0])
    ev.require(all(list(row) == fields for row in rows), f'inconsistent CSV columns: {path}')
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    with ev.atomic_target(path) as target:
        target.write(stream.getvalue().encode('utf-8'))

