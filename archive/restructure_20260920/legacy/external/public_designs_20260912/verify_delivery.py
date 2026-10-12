#!/usr/bin/env python3
"""Independent, CPU-only delivery audit of the public external-test extension.

Run only after pipeline.py and its workers exit. No evaluator/statistics module
is imported. This verifier never repairs artifacts or reruns model inference.
"""
from __future__ import annotations

import argparse
import collections
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback

import numpy as np

BASE = Path(__file__).resolve().parent
REV = Path('/remote-home/Guilin/DRO/Codex/wdro_precision_v2')
METHODS = ('erm', 'mixed_erm', 'hard_mining', 'group_dro', 'wdro_rest')
DEGREES = (10, 20, 30, 40, 50)
SEEDS = (101, 102, 103)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def json_scalar(value):
    """Convert NumPy scalar leaves without accepting arbitrary objects/arrays."""
    if isinstance(value,np.generic):
        return value.item()
    raise TypeError(f'Unsupported JSON value: {type(value).__name__}')


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.partial')
    with tmp.open('w') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False,default=json_scalar)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def equal(a, b, label):
    require(np.array_equal(a, b), f'Array mismatch: {label}')


def strings(values):
    return np.char.decode(values, 'utf-8') if values.dtype.kind == 'S' else values.astype(str)


def basic(values):
    n = len(values)
    return dict(count=n, mean=float(np.mean(values)) if n else None,
        p95=float(np.percentile(values, 95)) if n else None,
        minimum=float(np.min(values)) if n else None,
        maximum=float(np.max(values)) if n else None)


def pooled(values, designs, families):
    unique = sorted(set(designs.tolist()))
    means = [float(np.mean(values[designs == d])) for d in unique]
    worst = int(np.argmax(means)) if means else None
    return dict(count=len(values), net_weighted_mean=float(np.mean(values)) if len(values) else None,
        p95=float(np.percentile(values, 95)) if len(values) else None,
        design_count=len(unique), family_count=len(set(families.tolist())),
        design_macro_mean=float(np.mean(means)) if means else None,
        worst_design_mean=means[worst] if means else None,
        worst_design_id=unique[worst] if means else None)


def compare_value(actual, expected, label):
    if expected is None:
        require(actual in ('', None), f'{label}: expected null, got {actual}')
    elif isinstance(expected, (int, float, np.number)) and not isinstance(expected, bool):
        require(actual not in ('', None), f'{label}: missing number')
        number = float(actual)
        require(math.isfinite(number) and math.isclose(number, float(expected), rel_tol=2e-12, abs_tol=2e-14),
                f'{label}: {number} != {expected}')
    else:
        require(str(actual) == str(expected), f'{label}: {actual!r} != {expected!r}')


class Audit:
    def __init__(self, base):
        self.base = Path(base).resolve()
        self.bindings = {}
        self.binding_references = 0
        self.checks = {}
        self.counts = collections.Counter()

    def check(self, binding):
        path = Path(binding.get('logical_path', binding.get('path', ''))).absolute()
        require(path.is_file(), f'Missing artifact: {path}')
        if 'resolved_path' in binding:
            require(str(path.resolve()) == binding['resolved_path'], f'Resolved path changed: {path}')
        key = str(path)
        stamp = (path.stat().st_size, path.stat().st_mtime_ns)
        if key not in self.bindings:
            self.bindings[key] = dict(logical_path=key, resolved_path=str(path.resolve()),
                                      sha256=digest(path), bytes=stamp[0], _stamp=stamp)
        record = self.bindings[key]
        require(record['_stamp'] == stamp, f'Artifact changed while verifying: {path}')
        require(record['sha256'] == binding['sha256'], f'SHA256 mismatch: {path}')
        if 'bytes' in binding:
            require(record['bytes'] == binding['bytes'], f'Byte count mismatch: {path}')
        self.binding_references += 1
        return path

    def bind(self, path):
        path = Path(path)
        return self.check(dict(path=str(path), sha256=digest(path)))

    def walk(self, obj):
        if isinstance(obj, dict):
            if 'sha256' in obj and ('logical_path' in obj or 'path' in obj):
                self.check(obj)
            for value in obj.values():
                self.walk(value)
        elif isinstance(obj, list):
            for value in obj:
                self.walk(value)

    def document(self, relative):
        p = self.bind(self.base / relative)
        d = read(p)
        self.walk(d)
        return d

    def stage(self, name, value=True):
        self.checks[name] = value
        print(json.dumps(dict(stage=name, status='pass', unique_artifact_bindings=len(self.bindings))), flush=True)

    def sources_and_inputs(self):
        plan = self.document('cache/input_manifest.json')
        require(plan['status'] == 'frozen' and plan['train_or_retune'] is False, 'Input freeze/fitting status')
        require(plan['expected_model_tasks'] == 75 and plan['models_verified'] == 75, 'Expected model count')
        require(plan['expected_predictions'] == 15 * plan['total_nets'], 'Prediction plan count')
        require(plan['total_source_nets'] == plan['total_nets'] + plan['invalid_nets'] + plan['unsupported_degree_nets'],
                'Source population accounting')
        source_designs = []
        for artifact in plan['source_manifests']:
            source = read(self.check(artifact))
            self.walk(source)
            source_designs.extend(source['designs'])
            for design in source['designs']:
                if 'archive_path' in design:
                    self.check(dict(path=design['archive_path'], sha256=design['archive_sha256'], bytes=design['archive_bytes']))
                    self.counts['contest_downloaded_archives'] += 1
            if 'download_manifest_path' in source:
                dp = self.check(dict(path=source['download_manifest_path'], sha256=source['download_manifest_sha256']))
                downloaded = read(dp)
                self.walk(downloaded)
                self.counts['author_downloaded_files'] += len(downloaded['files'])
        require(len(source_designs) == plan['design_count'], 'Source design count differs')
        require({d['design_id'] for d in source_designs} == {d['design_id'] for d in plan['designs']}, 'Source design identities differ')
        source_lookup = {d['design_id']: d for d in source_designs}
        for design in plan['designs']:
            require(design['status'] == 'pass' and design['model_calls'] == 0, 'Input design status or model use')
            require(design['binding']['design'] == source_lookup[design['design_id']], 'Frozen source design record differs')
            saved = self.document(f'cache/designs/{design["design_id"]}/manifest.json')
            require(saved == design, 'Per-design manifest differs from frozen plan')
            require(sum(design['degree_histogram'].values()) == design['total_source_nets'], 'Raw histogram accounting')
            for group in design['groups']:
                require(group['parsed_count'] == group['valid_count'] + group['invalid_count'], 'Degree exclusion accounting')
        frozen = read(self.check(plan['model_index']))
        self.walk(frozen)
        require(frozen['status'] == 'frozen' and frozen['selection'] == 'synthetic_validation_only' and not frozen['test_used_for_selection'], 'Model selection provenance')
        expected = {(m, d, s) for m in METHODS for d in DEGREES for s in SEEDS}
        models = {(m['method'], m['degree'], m['seed']): m for m in frozen['models']}
        require(len(frozen['models']) == 75 and set(models) == expected, 'Frozen 75-model Cartesian product')
        for relative, sha in frozen['code_files'].items():
            self.check(dict(path=str(REV / relative), sha256=sha))
        audit = self.document('cache/input_audit.json')
        require(audit['status'] == 'pass' and audit['models_verified'] == 75, 'Independent input audit failed')
        require(audit['unique_source_ids'] == plan['total_nets'], 'Input audit population mismatch')
        require(self.check(audit['input_manifest']) == self.base/'cache/input_manifest.json', 'Input audit plan mismatch')
        data, metadata = {}, []
        for group in sorted(plan['groups'], key=lambda g:g['degree']):
            degree, n = group['degree'], group['count']
            a = load(self.check(group['input']))
            require(a['pins'].shape == (n, degree, 2), 'Normalized pin shape')
            require(all(len(v) == n for v in a.values()), 'Input vector lengths')
            require(np.isfinite(a['pins']).all() and np.isfinite(a['scales']).all() and np.all(a['scales'] > 0), 'Input geometry/scales')
            data[degree] = a
            metadata.append({**{k:strings(a[k]) for k in ('net_ids','design_ids','dataset_ids','family_ids')}, 'degree':np.full(n, degree)})
        require(set(data) == set(DEGREES), 'Missing input degree')
        meta = {k:np.concatenate([g[k] for g in metadata]) for k in metadata[0]}
        require(len(meta['net_ids']) == plan['total_nets'] and len(np.unique(meta['net_ids'])) == plan['total_nets'], 'Global source identity coverage')
        for design in plan['designs']:
            mask = meta['design_ids'] == design['design_id']
            require(set(meta['dataset_ids'][mask]) <= {design['dataset']}, 'Dataset label mismatch')
            require(set(meta['family_ids'][mask]) <= {design['binding']['design']['design_family']}, 'Design family label mismatch')
            for g in design['groups']:
                require(np.count_nonzero(mask & (meta['degree'] == g['degree'])) == g['valid_count'], 'Per-design degree coverage')
        self.plan, self.models, self.inputs, self.metadata = plan, models, data, meta
        self.counts.update(source_designs=len(source_designs), raw_source_nets=plan['total_source_nets'], eligible_nets=plan['total_nets'])
        self.stage('source_downloads_inputs_and_frozen_models')

    def references(self):
        manifest = self.document('cache/reference_manifest.json')
        require(manifest['status'] == 'pass' and manifest['total_nets'] == self.plan['total_nets'], 'Reference status/count')
        require(manifest['calls'] == 2 * self.plan['total_nets'], 'Reference full dual-scale call count')
        require(all(manifest['checks'].values()), 'Reference checks failed')
        require(self.check(manifest['input_manifest']) == self.base/'cache/input_manifest.json', 'Reference plan binding')
        identity = manifest['reference_identity']
        require(identity['scale'] == 2**27 and identity['require_status'] == 'optimal' and identity['rounding'] == 'none' and identity['validate_embedding'], 'Reference identity requirements')
        references, calls, shards = {}, 0, 0
        for g in manifest['groups']:
            d, n = g['degree'], g['count']
            require(n == len(self.inputs[d]['pins']), 'Reference degree count')
            full = load(self.check(g['reference']))
            require(set(full) == {'net_ids','lstar','verification_lstar'}, 'Reference schema')
            equal(full['net_ids'], self.inputs[d]['net_ids'], 'reference net order')
            equal(full['lstar'], full['verification_lstar'], 'full dual scale equality')
            require(np.isfinite(full['lstar']).all() and np.all(full['lstar'] > 0), 'Invalid reference length')
            receipt = self.document(f'cache/references/degree_{d}.json')
            require(receipt['status'] == 'complete' and receipt['completed_nets'] == n and receipt['scales'] == [2**27,2**28], 'Reference receipt')
            require(receipt['reference_identity'] == identity, 'Reference receipt identity')
            require(self.check(receipt['input']) == self.base/f'cache/inputs/degree_{d}.npz', 'Reference receipt input binding')
            require(self.check(receipt['reference']) == self.check(g['reference']), 'Reference receipt output binding')
            for start in range(0, n, 256):
                end = min(start+256, n)
                stem = f'cache/references/shards/d{d}_{start:08d}_{end:08d}'
                row = self.document(stem+'.json')
                expected = dict(input_sha256=next(g['input']['sha256'] for g in self.plan['groups'] if g['degree']==d),
                    start=start,end=end,reference_identity=identity,wrapper_sha256=digest(self.base/'reference.py'),scales=[2**27,2**28])
                require(row['status'] == 'pass' and row['binding'] == expected and row['calls'] == 2*(end-start), 'Reference shard binding/calls')
                part = load(self.check(row['output']))
                for key in full:
                    equal(part[key], full[key][start:end], f'reference shard/full {d}/{start}/{key}')
                started = self.document(stem+'.started.json')
                require(started['binding'] == expected and started['planned_calls'] == row['calls'], 'Reference started ledger')
                calls += row['calls']; shards += 1
            references[d] = full
        require(set(references) == set(DEGREES) and calls == manifest['calls'], 'Reference coverage/call ledger')
        self.refmanifest, self.refs = manifest, references
        self.counts.update(reference_calls=calls, reference_shards=shards)
        self.stage('dual_scale_reference_shards_and_full_calls')

    def supplementary_audits(self):
        coordinate=self.document('notes/selected_pin_coordinate_audit.json')
        self.bind(self.base/'notes/selected_pin_coordinate_audit.log')
        require(coordinate['status']=='PASS','Independent selected-pin coordinate audit failed')
        totals=coordinate['totals'];plans={d['design_id']:d for d in self.plan['designs']}
        require(totals['mismatches_gt_1_dbu']==totals['reconstruction_failures']==0,'Coordinate audit contains failures')
        require(totals['designs']==len(coordinate['designs']),'Coordinate audit design count')
        for row in coordinate['designs']:
            require(row['status']=='PASS' and row['all_ir_pin_count']==row['all_reconstructed_pin_count'],'Coordinate reconstruction coverage')
            require(row['mismatches_gt_1_dbu']==row['reconstruction_failure_count']==row['selected_mismatches_gt_1_dbu']==0,'Coordinate audit mismatch')
            require(not row['selected_coverage_errors'],'Coordinate audit selected-net coverage')
            groups=plans[row['design_id']]['groups']
            require(row['selected_evaluated_net_count']==sum(g['valid_count'] for g in groups),'Coordinate audit selected net total')
            require(row['selected_expected_pin_count']==row['selected_observed_pin_count']==sum(g['valid_count']*g['degree'] for g in groups),'Coordinate audit selected pin total')
        require(totals['all_ir_pins']==sum(r['all_ir_pin_count'] for r in coordinate['designs']),'Coordinate audit all-pin total')
        require(totals['selected_evaluated_nets']==sum(r['selected_evaluated_net_count'] for r in coordinate['designs']),'Coordinate audit net count')
        require(totals['selected_evaluated_pins']==sum(r['selected_observed_pin_count'] for r in coordinate['designs']),'Coordinate audit pin count')
        storage=self.document('logs/storage_acceleration.json')
        require(len({r['task'] for r in storage['files']})==len(storage['files']),'Repeated storage redirect')
        for row in storage['files']:
            logical,target=Path(row['logical_path']),Path(row['target_logical_path'])
            require(logical==self.base/'results'/row['task'] and logical.is_symlink(),'Unexpected storage redirect location')
            require(str(logical.resolve())==str(target.resolve())==row['resolved_path'],'Storage redirect target changed')
        self.counts.update(coordinate_audit_designs=totals['designs'],coordinate_audit_all_pins=totals['all_ir_pins'],
            coordinate_audit_evaluated_nets=totals['selected_evaluated_nets'],coordinate_audit_evaluated_pins=totals['selected_evaluated_pins'],
            nvme_redirected_task_directories=len(storage['files']))
        self.stage('independent_selected_pin_coordinate_audit_and_storage_provenance')

    def predictions(self):
        values = {(m,s):[] for m in METHODS for s in SEEDS}
        all_calls = collections.Counter()
        for degree in DEGREES:
            data, ref = self.inputs[degree], self.refs[degree]
            n = len(data['pins'])
            for method in METHODS:
                for seed in SEEDS:
                    task = f'{method}_d{degree}_s{seed}'
                    root = self.base/'results'/task
                    result = self.document(f'results/{task}/result.json')
                    b = result['binding']
                    require(result['status'] == 'complete' and result['completed_nets'] == n and b['count'] == n and b['degree'] == degree, f'Incomplete result {task}')
                    require(b['model'] == self.models[method,degree,seed], f'Wrong frozen model {task}')
                    require(b['model_selection'] is False and b['train_or_retune'] is False and result['model_selection'] is False, 'Fitting on test data')
                    require(b['reference_identity'] == self.refmanifest['reference_identity'], 'Model/reference identity')
                    identity=b['reference_identity']
                    source_shas={Path(s['logical_path']).name:s['sha256'] for s in b['sources']}
                    require(source_shas['geosteiner.py']==identity['implementation_sha256'] and source_shas['libeval.so']==identity['library_sha256'], 'Reference implementation/library identity')
                    require(self.check(b['input']) == self.base/f'cache/inputs/degree_{degree}.npz', 'Prediction input binding')
                    require(self.check(b['reference']) == self.base/f'cache/references/degree_{degree}.npz', 'Prediction reference binding')
                    require(self.check(b['reference_manifest']) == self.base/f'cache/references/degree_{degree}.json', 'Prediction reference receipt')
                    require(self.check(b['model_index']) == self.check(self.plan['model_index']) and self.check(b['canonical_frozen_manifest']) == self.check(self.plan['model_index']), 'Model manifest identity')
                    require(all(result['checks'].values()), f'Evaluation audit checks failed: {task}')
                    require(b['inference']['T'] == 1 and b['inference']['mode'] == 'deterministic_greedy', 'Inference policy changed')
                    pred = load(self.check(result['predictions']))
                    require(all(v.ndim > 0 and len(v) == n for v in pred.values()), 'Prediction lengths')
                    for key, a in data.items(): equal(pred[key], a, f'{task} input {key}')
                    for key in ('lstar','verification_lstar'): equal(pred[key], ref[key], f'{task} {key}')
                    require(np.isfinite(pred['rest_lengths']).all() and np.isfinite(pred['actor_log_probability']).all(), 'Nonfinite prediction')
                    require(pred['topology'].shape == (n,2*(degree-1)) and np.issubdtype(pred['topology'].dtype,np.integer), 'Topology encoding shape/type')
                    require(np.all((pred['topology'] >= 0) & (pred['topology'] < degree)), 'Topology index bounds')
                    equal(pred['actions'], pred['topology'], 'Action/topology alias')
                    equal(pred['regret'], (pred['rest_lengths']-ref['lstar'])/ref['lstar'], 'Regret formula')
                    require(np.isfinite(pred['regret']).all() and np.all(pred['regret'] >= -1e-12), 'Regret validity')
                    equal(pred['rest_lengths_inverse'], pred['rest_lengths']/data['scales'], 'Inverse REST scale')
                    equal(pred['lstar_inverse'], ref['lstar']/data['scales'], 'Inverse reference scale')
                    require(result['summary'] == basic(pred['regret']), 'Serialized result metrics')
                    cursor = self.document(f'results/{task}/cursor.json')
                    require(cursor['binding'] == b and cursor['complete_nets'] == n, 'Cursor binding/coverage')
                    ledger_path = self.bind(root/'calls.jsonl')
                    events = [json.loads(line) for line in ledger_path.read_text().splitlines() if line]
                    starts, finished = {}, {}
                    for event in events:
                        target = starts if event['phase'] == 'started' else finished
                        require(event['phase'] in ('started','completed') and event['id'] not in target, 'Duplicate/unknown call event')
                        target[event['id']] = event
                    end, committed = 0, set()
                    for shard in cursor['shards']:
                        start, stop = shard['start'], shard['end']
                        require(start == end and stop > start and stop <= n, 'Shard continuity')
                        part = load(self.check(shard['artifact']))
                        require(part.keys() == pred.keys(), 'Shard schema differs')
                        for key in pred: equal(part[key], pred[key][start:stop], f'{task} shard/full {start}/{key}')
                        attempt = shard['attempt']; require(attempt not in committed and attempt in starts and attempt in finished, 'Committed shard attempt ledger')
                        event = finished[attempt]
                        require(event['start'] == starts[attempt]['start'] == start and event['end'] == starts[attempt]['end'] == stop, 'Committed attempt range')
                        for key in ('actor_calls','rest_calls','python_tree_audit_calls'):
                            require(event[key] == stop-start, 'Committed per-net call count')
                        committed.add(attempt); end=stop
                    require(end == n, 'Missing final shard coverage')
                    expected_calls = dict(actor_forward_nets=n,exact_rest_evaluator_nets=n,geosteiner_nets=0,independent_python_tree_nets=n)
                    require(result['call_ledger'] == expected_calls, 'Result committed call ledger')
                    all_calls.update(expected_calls)
                    self.counts['uncommitted_started_attempts'] += len(set(starts)-committed)
                    self.counts['completed_uncommitted_attempts'] += len(set(finished)-committed)
                    self.counts['completed_uncommitted_actor_nets'] += sum(finished[a]['actor_calls'] for a in set(finished)-committed)
                    self.counts['completed_uncommitted_rest_nets'] += sum(finished[a]['rest_calls'] for a in set(finished)-committed)
                    self.counts['started_without_completion_planned_nets'] += sum(starts[a]['end']-starts[a]['start'] for a in set(starts)-set(finished))
                    self.counts['evaluation_shards'] += len(cursor['shards'])
                    self.counts['model_tasks'] += 1
                    values[method,seed].append(pred['regret'].copy())
            print(json.dumps(dict(stage='prediction_degree_verified',degree=degree,models=15)),flush=True)
        self.values = {k:np.concatenate(v) for k,v in values.items()}
        require(self.counts['model_tasks'] == 75 and all_calls['actor_forward_nets'] == self.plan['expected_predictions'], 'Model prediction/call coverage')
        self.counts.update(all_calls)
        self.counts['total_geosteiner_calls'] = self.counts['reference_calls'] + all_calls['geosteiner_nets']
        self.stage('prediction_bindings_formulas_shards_and_committed_calls')

    def statistics(self):
        summary = self.document('summary.json')
        require(summary['status'] == 'pass' and summary['model_tasks'] == 75 and summary['predictions'] == self.plan['expected_predictions'], 'Summary status/count')
        meta, values = self.metadata, self.values
        plans = {d['design_id']:d for d in self.plan['designs']}
        scopes = ['overall'] + ['dataset:'+s for s in sorted({d['dataset'] for d in plans.values()})]
        masks = {s:np.ones(len(meta['degree']),dtype=bool) if s=='overall' else meta['dataset_ids']==s.split(':',1)[1] for s in scopes}
        expected_counts = dict(per_design_degree_seed=len(plans)*5*15,per_design_seed=len(plans)*15,
            per_degree_seed=len(scopes)*5*15,per_dataset_seed=len(scopes)*15,summary=len(scopes)*5,
            paired_design_cluster_ci=sum(bool(np.any(v)) for v in masks.values())*4*8)
        require(set(summary['tables']) == set(expected_counts), 'Statistics table set')
        csv_tables = {}
        for name, count in expected_counts.items():
            path = self.check(summary['tables'][name]['artifact'])
            with path.open(newline='') as f: rows=list(csv.DictReader(f))
            require(len(rows) == count == summary['tables'][name]['rows'], f'Table row count {name}')
            csv_tables[name] = rows
        def validate_rows(name, expected, keyfields):
            rows=csv_tables[name]
            lookup={tuple(str(row[k]) for k in keyfields):row for row in rows}
            expected_lookup={tuple(str(row[k]) for k in keyfields):row for row in expected}
            require(len(lookup)==len(rows) and lookup.keys()==expected_lookup.keys(), f'Table identities {name}')
            for key, want in expected_lookup.items():
                got=lookup[key]
                require(got.keys()==want.keys(), f'Table columns {name}')
                for field,value in want.items(): compare_value(got[field],value,f'{name}/{key}/{field}')
        by_design, by_dd, by_dataset, by_degree, overall = [],[],[],[],[]
        for method in METHODS:
            for seed in SEEDS:
                regret=values[method,seed]
                for design in sorted(plans):
                    p=plans[design];mask=meta['design_ids']==design
                    ident=dict(dataset_id=p['dataset'],design_id=design,family_id=p['binding']['design']['design_family'],method=method,training_seed=seed)
                    by_design.append(dict(**ident,**basic(regret[mask])))
                    for d in DEGREES:
                        selected=mask & (meta['degree']==d); g=next(x for x in p['groups'] if x['degree']==d)
                        by_dd.append(dict(**ident,degree=d,parsed_count=g['parsed_count'],invalid_count=g['invalid_count'],status='valid' if np.any(selected) else 'no_valid_nets',**basic(regret[selected])))
                for scope,mask in masks.items():
                    by_dataset.append(dict(scope=scope,method=method,training_seed=seed,**pooled(regret[mask],meta['design_ids'][mask],meta['family_ids'][mask])))
                    for d in DEGREES:
                        chosen=mask & (meta['degree']==d)
                        by_degree.append(dict(scope=scope,degree=d,method=method,training_seed=seed,**pooled(regret[chosen],meta['design_ids'][chosen],meta['family_ids'][chosen])))
            for scope in scopes:
                parts=[r for r in by_dataset if r['scope']==scope and r['method']==method]
                row=dict(scope=scope,method=method,training_seeds='101,102,103',**{k:parts[0][k] for k in ('count','design_count','family_count')})
                for key in ('net_weighted_mean','p95','design_macro_mean','worst_design_mean'):
                    row[key]=float(np.mean([r[key] for r in parts])) if parts[0]['count'] else None
                overall.append(row)
        validate_rows('per_design_seed',by_design,('design_id','method','training_seed'))
        validate_rows('per_design_degree_seed',by_dd,('design_id','degree','method','training_seed'))
        validate_rows('per_dataset_seed',by_dataset,('scope','method','training_seed'))
        validate_rows('per_degree_seed',by_degree,('scope','degree','method','training_seed'))
        validate_rows('summary',overall,('scope','method'))
        require(len(summary['summary'])==len(overall), 'JSON summary coverage')
        js={(r['scope'],r['method']):r for r in summary['summary']}
        for row in overall:
            actual=js[row['scope'],row['method']]
            require(actual.keys()==row.keys(),'JSON summary columns')
            for k,v in row.items():compare_value(actual[k],v,f'JSON summary {k}')
        ci=[]
        for scope,mask in masks.items():
            if not np.any(mask):continue
            names=sorted(set(meta['family_ids'][mask].tolist())); fcount=len(names)
            design=meta['design_ids'][mask]; family=meta['family_ids'][mask]
            draws=np.random.default_rng(20260912).integers(0,fcount,size=(2000,fcount))
            for baseline in METHODS[:-1]:
                delta=np.stack([values['wdro_rest',s][mask]-values[baseline,s][mask] for s in SEEDS])
                for metric in ('net_weighted_mean','design_macro_mean'):
                    numerators=[]; denominators=[]
                    for fam in names:
                        fm=family==fam
                        if metric=='net_weighted_mean':
                            numerators.append(np.sum(delta[:,fm],axis=1));denominators.append(int(fm.sum()))
                        else:
                            ds=sorted(set(design[fm].tolist()))
                            numerators.append(np.sum([np.mean(delta[:,design==d],axis=1) for d in ds],axis=0));denominators.append(len(ds))
                    numerator=np.asarray(numerators);denominator=np.asarray(denominators)
                    point=numerator.sum(axis=0)/denominator.sum()
                    replicates=numerator[draws].sum(axis=1)/denominator[draws].sum(axis=1)[:,None]
                    for i,seed in enumerate((*SEEDS,'mean')):
                        p=float(point[i]) if i<3 else float(point.mean())
                        sample=replicates[:,i] if i<3 else replicates.mean(axis=1)
                        lo,hi=np.percentile(sample,[2.5,97.5])
                        ci.append(dict(scope=scope,candidate='wdro_rest',baseline=baseline,metric=metric,training_seed=seed,
                            mean_difference=p,ci_low=float(lo),ci_high=float(hi),confidence_level=.95,bootstrap_samples=2000,random_seed=20260912,
                            count=len(design),design_count=len(set(design.tolist())),family_count=fcount,cluster_unit='design_family',
                            interval_status='single_family_degenerate' if fcount==1 else 'cluster_percentile',
                            seed_aggregation='arithmetic_mean_of_three_seed_metrics' if seed=='mean' else 'individual_training_seed'))
        validate_rows('paired_design_cluster_ci',ci,('scope','baseline','metric','training_seed'))
        self.counts['statistics_tables']=len(expected_counts);self.counts['statistics_rows']=sum(expected_counts.values())
        self.stage('independent_dataset_design_degree_seed_statistics_and_family_bootstrap')

    def processes_and_locks(self):
        pipeline=self.document('logs/pipeline/completion.json');queue=self.document('logs/queue/completion.json')
        require(pipeline['status']=='pass' and queue['status']=='pass' and not queue['failed'], 'Pipeline/queue completion')
        require(queue['completed_tasks']==queue['expected_tasks']==75, 'Queue model count')
        expected={f'{m}_d{d}_s{s}' for m,d,s in self.models}
        require({row['task']['id'] for row in queue['completed']}==expected, 'Queue task identity coverage')
        for name in ('prepare','reference','audit_inputs','run_queue','summarize'):
            row=self.document(f'logs/pipeline/{name}.json')
            require(row['status']=='pass' and row['exit_code']==0, f'Pipeline substep failed: {name}')
        reports=[]
        boot_seconds=int(next(line.split()[1] for line in Path('/proc/stat').read_text().splitlines() if line.startswith('btime ')))
        clock_ticks=os.sysconf('SC_CLK_TCK')
        paths=list((self.base/'results').glob('*/job.json'))+list((self.base/'logs').rglob('job.json'))
        for jobpath in paths:
            # Historical smoke attempts are still checked for exited ownership;
            # deliberately paused smoke jobs may have an exited/paused outcome.
            job=self.document(str(jobpath.relative_to(self.base)))
            require(job['status']=='exited' and job.get('lock_released') is True, f'Job not exited/released: {jobpath}')
            ident=job.get('process',dict(pid=job.get('pid')));pid=ident['pid']
            proc=Path(f'/proc/{pid}')
            state='absent';reused=False
            try:
                stat=(proc/'stat').read_text().rsplit(') ',1)[1].split();state=stat[0]
                cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
                reused='start_ticks' in ident and str(ident['start_ticks'])!=stat[19]
                if 'start_ticks' not in ident:
                    finish=job.get('finished',job.get('finished_at_utc'))
                    require(bool(finish),f'Job has no exit timestamp: {jobpath}')
                    finished_seconds=datetime.fromisoformat(finish).timestamp()
                    reused=boot_seconds+int(stat[19])/clock_ticks>finished_seconds+1
                matching=not reused
                require(not matching or state in ('Z','X'), f'Job process still executing: pid {pid}, {jobpath}')
            except FileNotFoundError:pass
            lock=jobpath.with_name('run.lock');require(lock.exists(),f'Missing lock artifact: {lock}')
            with lock.open('r+') as stream:
                fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
                fcntl.flock(stream,fcntl.LOCK_UN)
            reports.append(dict(job=str(jobpath),pid=pid,current_state=state,pid_reused=reused,lock_acquired_and_released=True))
        # Detect any unrecorded live pipeline/reference/evaluation worker.
        forbidden={'pipeline.py','prepare.py','reference.py','audit_inputs.py','run_queue.py','evaluate.py','summarize.py'}
        for proc in Path('/proc').iterdir():
            if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
            try:
                argv=(proc/'cmdline').read_bytes().split(b'\0')
                state=(proc/'stat').read_text().rsplit(') ',1)[1].split()[0]
                active=any(a.decode(errors='replace')==str(self.base/name) for a in argv for name in forbidden)
                require(not active or state in ('Z','X'),f'Unrecorded worker still active: {proc.name}')
            except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        self.process_report=reports
        self.counts['job_locks_verified']=len(reports)
        self.stage('all_task_processes_exited_and_locks_acquired_released')

    def run(self):
        self.sources_and_inputs();self.supplementary_audits();self.references();self.predictions();self.statistics();self.processes_and_locks()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,default=BASE)
    p.add_argument('--output',type=Path,default=BASE/'delivery_verification.json')
    p.add_argument('--bindings',type=Path,default=BASE/'delivery_artifact_bindings.json')
    args=p.parse_args(argv)
    require(args.base.resolve()==BASE,'Verifier is scoped to this extension')
    require(args.output.resolve().is_relative_to(BASE) and args.bindings.resolve().is_relative_to(BASE),'Output outside extension')
    audit=Audit(args.base);began=time.monotonic();failure=None
    try:
        audit.bind(__file__);audit.run()
    except Exception as error:
        failure=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
    bindings=[{k:v for k,v in row.items() if k!='_stamp'} for _,row in sorted(audit.bindings.items())]
    atomic_json(args.bindings,dict(status='pass' if failure is None else 'failed',artifacts=bindings,unique_artifacts=len(bindings),binding_references=audit.binding_references))
    result=dict(status='pass' if failure is None else 'failed',at_utc=datetime.now(timezone.utc).isoformat(),
        verifier=dict(path=str(Path(__file__).resolve()),sha256=digest(__file__)),checks=audit.checks,counts={k:int(v) for k,v in audit.counts.items()},
        artifact_bindings=dict(path=str(args.bindings.resolve()),sha256=digest(args.bindings)),
        process_audit=getattr(audit,'process_report',[]),failure=failure,wall_seconds=time.monotonic()-began,
        limitations=['No new GPU inference or GeoSteiner solve is performed.',
            'Tree connectivity and Python tree-length recomputation are evidenced by completed evaluator checks and exact committed-shard/full-prediction equality; this verifier independently checks formulas, scales, identities, topology bounds, provenance, and all statistics.',
            'Call totals distinguish committed per-net calls from started but uncommitted/replayed attempts; interrupted-call work is not asserted to be zero.'])
    atomic_json(args.output,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('process_audit',)},ensure_ascii=False,default=json_scalar),flush=True)
    return 0 if failure is None else 1


if __name__=='__main__':
    raise SystemExit(main())
