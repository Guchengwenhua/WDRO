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


def load_verified_predictions(manifest, results):
    groups = sorted(manifest['groups'], key=lambda group: group['degree'])
    ev.require({g['degree'] for g in groups} == set(DEGREES), 'expected all five supported degrees')
    frozen_index = ev.checked(manifest['model_index'])
    frozen = ev.read_json(frozen_index)
    models = {(m['method'], m['degree'], m['seed']): m for m in frozen['models']}
    expected = {(method, degree, seed) for method in METHODS for degree in DEGREES for seed in SEEDS}
    ev.require(set(models) == expected and len(frozen['models']) == 75, 'frozen model set differs')
    metadata, values, bindings = [], {(m, s): [] for m in METHODS for s in SEEDS}, []
    for group in groups:
        data_path = ev.checked(group['input'])
        data = ev.load_npz(data_path)
        ev.validate_inputs(data, group['degree'])
        ev.require(len(data['pins']) == group['count'], 'input group count mismatch')
        metadata.append(dict(net_ids=string_vector(data['net_ids']), design_ids=string_vector(data['design_ids']),
            dataset_ids=string_vector(data['dataset_ids']), family_ids=string_vector(data['family_ids']),
            degree=np.full(group['count'], group['degree'], dtype=np.int64)))
        for method in METHODS:
            for seed in SEEDS:
                path = results / f'{method}_d{group["degree"]}_s{seed}' / 'result.json'
                result = ev.read_json(path)
                ev.require(result['status'] == 'complete' and result['completed_nets'] == group['count'], f'incomplete model result: {path}')
                binding = result['binding']
                ev.require(binding['model'] == models[(method, group['degree'], seed)], f'result uses different frozen model: {path}')
                ev.checked(binding['model']['best'])
                ev.require(ev.checked(binding['input']) == data_path, f'result input differs: {path}')
                ev.require(binding['model_selection'] is False and binding['train_or_retune'] is False, 'unexpected fitting on test data')
                prediction = ev.load_npz(ev.checked(result['predictions']))
                for key in ('net_ids', 'design_ids', 'dataset_ids', 'family_ids'):
                    ev.require(np.array_equal(prediction[key], data[key]), f'prediction identities changed: {key}, {path}')
                regret = prediction['regret']
                ev.require(regret.shape == (group['count'],) and np.isfinite(regret).all() and np.all(regret >= -1e-12), 'invalid regret vector')
                recomputed = (prediction['rest_lengths'] - prediction['lstar']) / prediction['lstar']
                ev.require(np.array_equal(regret, recomputed), 'stored regret formula mismatch')
                ev.require(result['summary'] == metrics(regret), 'result metrics differ from serialized predictions')
                ev.require(result['checks']['all_tree_lengths_independently_equal'] and result['checks']['references_provenance_verified'], 'evaluation/reference audit missing')
                values[(method, seed)].append(regret)
                bindings.append(dict(result=ev.artifact(path), predictions=result['predictions']))
    metadata = {key: np.concatenate([part[key] for part in metadata]) for key in metadata[0]}
    values = {key: np.concatenate(parts) for key, parts in values.items()}
    ev.require(len(metadata['net_ids']) == manifest['total_nets'] and len(np.unique(metadata['net_ids'])) == manifest['total_nets'], 'global net count or uniqueness mismatch')
    ev.require(sum(len(v) for v in values.values()) == manifest['expected_predictions'], 'prediction coverage mismatch')
    return metadata, values, bindings


def build_tables(manifest, metadata, values):
    design_plan = {r['design_id']: r for r in manifest['designs']}
    design_ids, datasets, families, degree_vector = (metadata[key] for key in ('design_ids', 'dataset_ids', 'family_ids', 'degree'))
    ev.require(set(design_ids) <= design_plan.keys(), 'input contains an undeclared design')
    dataset_names = sorted({r['dataset'] for r in design_plan.values()})
    scopes = [('overall', np.ones(len(design_ids), dtype=bool))] + [(f'dataset:{dataset}', datasets == dataset) for dataset in dataset_names]
    per_design_degree, per_design, per_degree, per_dataset, summary, paired = [], [], [], [], [], []
    for method in METHODS:
        for seed in SEEDS:
            regrets = values[(method, seed)]
            for design in sorted(design_plan):
                record = design_plan[design]
                design_mask = design_ids == design
                family = record['binding']['design'].get('design_family', design)
                if np.any(design_mask):
                    ev.require(set(datasets[design_mask]) == {record['dataset']} and set(families[design_mask]) == {family}, f'design metadata mismatch: {design}')
                identity = dict(dataset_id=record['dataset'], design_id=design, family_id=family, method=method, training_seed=seed)
                per_design.append(dict(**identity, **metrics(regrets[design_mask])))
                for degree in DEGREES:
                    mask = design_mask & (degree_vector == degree)
                    source = next(g for g in record['groups'] if g['degree'] == degree)
                    ev.require(int(mask.sum()) == source['valid_count'], f'design-degree coverage differs: {design}/{degree}')
                    per_design_degree.append(dict(**identity, degree=degree, parsed_count=source['parsed_count'],
                        invalid_count=source['invalid_count'], status='valid' if np.any(mask) else 'no_valid_nets', **metrics(regrets[mask])))
            for scope, mask in scopes:
                per_dataset.append(dict(scope=scope, method=method, training_seed=seed,
                    **pooled_metrics(regrets[mask], design_ids[mask], families[mask])))
                for degree in DEGREES:
                    degree_mask = mask & (degree_vector == degree)
                    per_degree.append(dict(scope=scope, degree=degree, method=method, training_seed=seed,
                        **pooled_metrics(regrets[degree_mask], design_ids[degree_mask], families[degree_mask])))
        for scope, mask in scopes:
            seeds = [r for r in per_dataset if r['scope'] == scope and r['method'] == method]
            ev.require(len(seeds) == 3, 'missing seed metrics')
            row = dict(scope=scope, method=method, training_seeds='101,102,103', count=seeds[0]['count'],
                design_count=seeds[0]['design_count'], family_count=seeds[0]['family_count'])
            for metric in ('net_weighted_mean', 'p95', 'design_macro_mean', 'worst_design_mean'):
                row[metric] = float(np.mean([r[metric] for r in seeds])) if seeds[0]['count'] else None
            summary.append(row)
    for scope, mask in scopes:
        if not np.any(mask):
            continue
        for baseline in METHODS[:-1]:
            difference = np.array([values[('wdro_rest', seed)][mask] - values[(baseline, seed)][mask] for seed in SEEDS])
            for row in family_bootstrap(difference, design_ids[mask], families[mask]):
                paired.append(dict(scope=scope, candidate='wdro_rest', baseline=baseline, **row))
    return dict(per_design_degree_seed=per_design_degree, per_design_seed=per_design,
        per_degree_seed=per_degree, per_dataset_seed=per_dataset, summary=summary, paired_design_cluster_ci=paired)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-manifest', default=str(BASE / 'cache/input_manifest.json'))
    parser.add_argument('--results', default=str(BASE / 'results'))
    parser.add_argument('--tables', default=str(BASE / 'tables'))
    parser.add_argument('--summary', default=str(BASE / 'summary.json'))
    args = parser.parse_args(argv)
    input_path, result_root, table_root, summary_path = map(ev.project_path,
        (args.input_manifest, args.results, args.tables, args.summary))
    ev.require(all(path.is_relative_to(BASE) for path in (input_path, result_root, table_root, summary_path)), 'all statistics artifacts must stay under extension')
    with ev.Lease(BASE / 'logs/statistics', -1) as lease:
        began = time.monotonic()
        manifest = ev.read_json(input_path)
        ev.require(manifest['status'] == 'frozen' and manifest['expected_model_tasks'] == 75, 'input freeze incomplete')
        metadata, values, predictions = load_verified_predictions(manifest, result_root)
        lease.update(len(predictions), 'loaded_and_verified')
        tables = build_tables(manifest, metadata, values)
        table_bindings = {}
        for name, rows in tables.items():
            path = table_root / f'{name}.csv'
            write_csv(path, rows)
            # Parse the actual serialized CSV and reconcile row and column counts.
            with path.open(newline='') as stream:
                parsed = list(csv.DictReader(stream))
            ev.require(len(parsed) == len(rows) and all(set(r) == set(rows[0]) for r in parsed), 'serialized table coverage mismatch')
            table_bindings[name] = dict(artifact=ev.artifact(path), rows=len(rows))
        result = dict(status='pass', at_utc=ev.now(), input_manifest=ev.artifact(input_path),
            source=ev.artifact(__file__), evaluator_source=ev.artifact(ev.__file__),
            model_tasks=len(predictions), net_count=manifest['total_nets'], predictions=manifest['expected_predictions'],
            design_count=manifest['design_count'], valid_design_count=manifest['valid_design_count'],
            design_family_count=len(set(metadata['family_ids'])), datasets=sorted(set(metadata['dataset_ids'])),
            tables=table_bindings, summary=tables['summary'], prediction_bindings=predictions,
            metric_scale='dimensionless relative regret; multiply by 100 for percent',
            aggregation='Each seed metric is computed from all eligible nets in its scope, then three seed metrics are equally averaged; P95 is not a quantile of seed-averaged per-net scores.',
            confidence_intervals=dict(samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED, confidence_level=.95,
                candidate='wdro_rest', baselines=list(METHODS[:-1]), cluster_unit='design_family',
                coupled_variants='All physical implementations of one logical family share their bootstrap multiplicity.',
                coupled_models='Same family draw for every method and seed; seed metrics averaged inside each replicate.',
                point_weighting='Net-weighted mean and equal-design macro mean retain their declared weights within resampled families.',
                training_seeds='Fixed fitted models; training seeds are not treated as independent design clusters.',
                single_family='A scope with one family produces a degenerate empirical interval, not informative distributional uncertainty.'),
            checks=dict(all_75_frozen_models_verified=True, all_predictions_covered=True,
                per_seed_aggregation_before_averaging=True, design_family_clusters_preserved=True,
                no_claim_support_flags=True), wall_seconds=time.monotonic()-began)
        ev.atomic_json(summary_path, result)
        lease.update(len(predictions), 'complete')
        print(json.dumps(dict(status='pass', summary=ev.artifact(summary_path), model_tasks=75,
            predictions=result['predictions'], tables=table_bindings)), flush=True)


if __name__ == '__main__':
    main()
