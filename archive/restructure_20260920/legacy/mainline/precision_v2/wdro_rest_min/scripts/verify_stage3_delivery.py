"""Independent readback of the completed stage-3 delivery and reference ledgers."""
import collections
import csv
import json
import os
import sys
from pathlib import Path

CODEX = Path('/remote-home/Guilin/DRO/Codex/wdro_precision_v2')
ROOT = CODEX / 'wdro_rest_min'
EXP = CODEX / 'experiments'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from src.io_utils import atomic_write_json
from src.job_control import utc_now
from src.protocol import load_protocol
from src.rest import file_sha256
from run_stage3_baseline_queue import build_tasks
from stage3_pipeline import verify_sources

verify_sources()
pipeline = json.loads((EXP / 'logs/stage3_pipeline/heartbeat.json').read_text())
assert pipeline['status'] == 'complete', 'Wait for audit and export to finish'
completion_path = ROOT / 'manifests/stage3_completion_v1.json'
completion = json.loads(completion_path.read_text())
assert completion['status'] == 'pass' and completion['model_count'] == 45
assert not completion['stage4_started'] and not completion['test_or_ispd_models_evaluated']
tasks = {t['run_id']: t for t in build_tasks()}
assert len(tasks) == 45 and {m['run_id'] for m in completion['models']} == set(tasks)
models = {m['run_id']: m for m in completion['models']}
audit_path = Path(completion['audit']['path'])
assert file_sha256(audit_path) == completion['audit']['sha256']
audit = json.loads(audit_path.read_text())
assert audit['status'] == 'pass' and audit['all_inputs_rebuilt']
assert not audit['test_ispd_models_evaluated']
assert len(audit['runs']) == 45 and {a['run_id'] for a in audit['runs']} == set(tasks)
assert audit['checks_passed'] == sum(len(a['checks']) for a in audit['runs'])
train_nets = candidate_nets = 0
for a in audit['runs']:
    assert a['status'] == 'pass' and all(a['checks'].values())
    cursor = json.loads(Path(a['stream_cursor']).read_text())
    assert cursor['status'] == 'pass' and cursor['completed_updates'] == 2000
    assert cursor['heldout_intersections'] == 0 and cursor['nets_checked'] == 512000
    expected_candidates = 16896000 if tasks[a['run_id']]['method'] == 'hard_mining' else 0
    assert cursor['candidate_nets_checked'] == expected_candidates
    train_nets += cursor['nets_checked']
    candidate_nets += cursor['candidate_nets_checked']

tables = {}
for name, item in completion['tables'].items():
    p = Path(item['logical_path'])
    assert p.is_relative_to(CODEX) and str(p.resolve()) == item['resolved_path']
    assert file_sha256(p) == item['sha256']
    with p.open(newline='') as handle:
        tables[name] = list(csv.DictReader(handle))
assert {name: len(rows) for name, rows in tables.items()} == {
    'runs': 45, 'best_validation': 180, 'all_validation': 3780, 'hard_mining_budgets': 15}
runs = {r['run_id']: r for r in tables['runs']}
assert set(runs) == set(tasks)
for run_id, row in runs.items():
    assert int(row['policy_updates']) == 2000 and int(row['batch_size']) == 256
    assert float(row['nominal_delta_vs_erm']) <= 0.002 + 1e-12
    result_path = EXP / 'logs/stage3' / run_id / 'result.json'
    a = next(a for a in audit['runs'] if a['run_id'] == run_id)
    assert file_sha256(result_path) == a['result_sha256']
    result = json.loads(result_path.read_text())
    for kind in ('latest', 'best'):
        assert file_sha256(result[kind + '_checkpoint']) == row[kind + '_checkpoint_sha256']
        assert models[run_id][kind + '_sha256'] == row[kind + '_checkpoint_sha256']
    model = models[run_id]
    assert model['best_logical_path'] == result['best_checkpoint']
    assert Path(model['best_logical_path']).is_relative_to(CODEX)
    assert model['best_resolved_path'] == str(Path(model['best_logical_path']).resolve())
    assert (model['method'], model['degree'], model['seed']) == (
        tasks[run_id]['method'], tasks[run_id]['degree'], tasks[run_id]['seed'])
groups = load_protocol()['data']['distributions']
all_keys = [(r['run_id'], int(r['update']), r['distribution']) for r in tables['all_validation']]
assert len(all_keys) == len(set(all_keys))
assert set(all_keys) == {(r, u, g) for r in tasks for u in range(0, 2001, 100) for g in groups}
best_keys = {(r['run_id'], int(r['update']), r['distribution']) for r in tables['best_validation']}
assert best_keys == {(r, int(runs[r]['best_update']), g) for r in tasks for g in groups}
assert {k for k, r in zip(all_keys, tables['all_validation']) if r['is_selected'] == 'True'} == best_keys
selected_rows = {k: r for k, r in zip(all_keys, tables['all_validation']) if r['is_selected'] == 'True'}
assert all(selected_rows[(r['run_id'], int(r['update']), r['distribution'])] == r
           for r in tables['best_validation'])

cache_path = EXP / 'logs/stage3_hard_mining_budget_precheck_20260905_all15/result.json'
cache = json.loads(cache_path.read_text())
assert cache['status'] == 'pass' and len(cache['reports']) == 15 and cache['checks_passed'] == 180
hard_ids = {r for r, t in tasks.items() if t['method'] == 'hard_mining'}
assert {r['run_id'] for r in cache['reports']} == hard_ids
assert {r['run_id'] for r in tables['hard_mining_budgets']} == hard_ids
actual_cache_calls = collections.Counter()
for report in cache['reports']:
    assert all(report['checks'].values())
    result_path = EXP / 'logs/stage3' / report['run_id'] / 'result.json'
    assert file_sha256(result_path) == report['result_sha256']
    actual_cache_calls.update(report['actual_cache_calls'])
report_path = CODEX / 'docs/阶段3_三基线与验收报告.md'
assert report_path.exists() and '45/45' in report_path.read_text()
verification = {'status': 'pass', 'at_utc': utc_now(), 'stage': 3,
    'completion_manifest': {'logical_path': str(completion_path),
        'resolved_path': str(completion_path.resolve()), 'sha256': file_sha256(completion_path)},
    'full_audit_checks_passed': audit['checks_passed'], 'training_nets_rebuilt': train_nets,
    'candidate_nets_rebuilt': candidate_nets, 'heldout_intersections': 0,
    'checkpoint_hashes_reverified': 90, 'table_row_counts': {k: len(v) for k, v in tables.items()},
    'reference_cache_checks_passed': 180,
    'reference_cache_audit': {'logical_path': str(cache_path),
        'resolved_path': str(cache_path.resolve()), 'sha256': file_sha256(cache_path)},
    'actual_reference_cache_calls': dict(actual_cache_calls),
    'maximum_selected_nominal_delta_vs_erm': max(float(r['nominal_delta_vs_erm']) for r in runs.values()),
    'stage4_started': False, 'test_or_ispd_models_evaluated': False}
path = ROOT / 'manifests/stage3_delivery_verification_v1.json'
atomic_write_json(path, verification, mode=0o444)
print(json.dumps({'status': 'pass', 'verification': str(path), **{
    k: verification[k] for k in ('full_audit_checks_passed', 'training_nets_rebuilt',
                                  'candidate_nets_rebuilt', 'reference_cache_checks_passed')}}), flush=True)
