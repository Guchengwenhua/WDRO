"""Early structural check; full candidate reconstruction remains mandatory."""
import json
import os
import shlex
import sqlite3
import sys
from pathlib import Path

import numpy as np
import torch

CODEX = Path('/remote-home/Guilin/DRO/Codex')
ROOT = CODEX / 'wdro_rest_min'
EXP = CODEX / 'experiments'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from src.io_utils import atomic_write_json
from src.job_control import JobLease, utc_now
from src.rest import file_sha256
from src.protocol import load_protocol
from src.stage2_audit_checks import all_finite
from audit_stage3 import effective_events
from audit_stage3_completed import stop_training
from run_stage3_baseline_queue import build_tasks, task_completed, result_path
from stage3_pipeline import verify_sources, snapshot_cpu

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.nice(10)
verify_sources()
runtime = snapshot_cpu('stage3_completed_hard_mining_structural_budget_precheck')
tasks = [t for t in build_tasks() if t['method'] == 'hard_mining' and task_completed(t)]
protocol = load_protocol()
job = EXP / 'logs/stage3_hard_mining_budget_precheck_20260905'
command = shlex.join(['bash', str(ROOT / 'scripts/stage3_env.sh'), sys.executable, str(Path(__file__))])
reports = []
with JobLease(job, run_id='stage3_hard_mining_budget_precheck_20260905', exact_command=command,
              resume_command=command, monitor_command=f'cat {job / "heartbeat.json"}',
              expected_updates=len(tasks), physical_gpu_index=None, host_runtime_manifest=str(runtime)) as lease:
    for task in tasks:
        result = json.loads(result_path(task).read_text())
        train, validation, raw = effective_events(result['event_log'])
        latest = torch.load(result['latest_checkpoint'], map_location='cpu', weights_only=False)
        best = torch.load(result['best_checkpoint'], map_location='cpu', weights_only=False)
        budget = result['method_state']['budget']
        n = 2000 * 256
        expected = {'actor_forward_nets': n * 5 + 21 * 8000 + 1,
                    'exact_rest_evaluator_nets': n * 5 + 21 * 8000 + 2,
                    'surrogate_evaluator_nets': n * 33,
                    'geosteiner_nets': 2 + sum(e['method_detail']['reference_cache_misses'] for e in train.values())}
        erm = json.loads((EXP / 'logs/stage2' / f"formal_erm_d{task['degree']}_s{task['seed']}_u2000" / 'result.json').read_text())
        limit = erm['best']['nominal_mean'] + protocol['training']['selection']['nominal_mean_regret_degradation_max_absolute']
        feasible = [e for _, e in sorted(validation.items()) if e['result']['selection']['nominal_mean'] <= limit]
        selected = min(feasible, key=lambda e: e['result']['selection']['worst_group_mean'])
        exact_argmax = True
        for e in train.values():
            detail = e['method_detail']
            indices = np.asarray(detail['exact_shortlist_indices'])
            scores = np.asarray(detail['exact_relative_regrets'])
            chosen = np.asarray(detail['selected_indices'])
            exact_argmax &= indices.shape == scores.shape == (256, 4) and np.isfinite(scores).all()
            exact_argmax &= np.array_equal(chosen, indices[np.arange(256), scores.argmax(axis=1)])
            exact_argmax &= np.all(indices[:, 0] == 0)
        dbpath = EXP / 'cache/stage3_references' / task['run_id'] / 'lstar.sqlite'
        with sqlite3.connect(dbpath.as_uri() + '?mode=ro', uri=True) as db:
            integrity = db.execute('PRAGMA quick_check').fetchall()
            calls = db.execute('SELECT COUNT(*),SUM(requested),SUM(misses_started),SUM(misses_completed) FROM calls').fetchone()
            references = db.execute('SELECT COUNT(*),MIN(length) FROM lstar').fetchone()
        checks = {
            'all_2000_updates_and_21_validation_points': sorted(train) == list(range(1, 2001)) and sorted(validation) == list(range(0, 2001, 100)),
            'policy_and_candidate_budgets': budget['centers'] == n and budget['perturbations'] == n * 32 and budget['surrogate_nets'] == n * 33 and budget['exact_rescore_nets'] == budget['reference_requests'] == n * 4,
            'effective_call_ledger': latest['call_ledger'] == result['call_ledger'] == expected,
            'all_logged_selections_maximize_true_shortlist_regret': bool(exact_argmax),
            'cache_hit_miss_balance': budget['reference_cache_hits'] + budget['reference_cache_misses'] == n * 4,
            'sqlite_integrity': integrity == [('ok',)],
            'actual_cache_calls_cover_effective_trajectory': calls[1] >= n * 4 and calls[2] >= calls[3] >= budget['reference_cache_misses'],
            'cache_committed_misses_match_stored_references': calls[3] == references[0] and references[1] > 0,
            'nominal_guard_and_earliest_feasible_best': result['nominal_limit'] == limit and best['update'] == result['best']['update'] == selected['update'] and result['best']['nominal_mean'] <= limit,
            'latest_and_best_hashes': file_sha256(result['latest_checkpoint']) == result['latest_checkpoint_sha256'] and file_sha256(result['best_checkpoint']) == result['best_checkpoint_sha256'],
            'finite_checkpoint_and_logged_regrets': all_finite(latest) and all_finite(best) and all_finite(list(train.values())),
            'only_validation_metrics_used': result['test_or_ispd_used'] is False and all(e['result']['split'] == 'validation' for e in validation.values()),
        }
        report = {'at_utc': utc_now(), 'run_id': task['run_id'], 'status': 'pass' if all(checks.values()) else 'fail',
                  'scope': 'structural_budget_precheck_only', 'full_candidate_reconstruction_complete': False,
                  'stage3_acceptance': False, 'checks': checks, 'budget': budget, 'call_ledger': expected,
                  'actual_cache_calls': dict(zip(['receipts', 'requested', 'misses_started', 'misses_completed'], calls)),
                  'cache_logical_path': str(dbpath), 'cache_resolved_path': str(dbpath.resolve()),
                  'result_sha256': file_sha256(result_path(task))}
        atomic_write_json(job / (task['run_id'] + '.json'), report)
        if report['status'] != 'pass':
            reason = f"Hard Mining structural audit failed: {task['run_id']}: {[k for k, v in checks.items() if not v]}"
            stop_training(reason)
            raise RuntimeError(reason)
        reports.append(report)
        lease.update(update=len(reports), status='running', detail={'completed_run': task['run_id']})
    summary = {'status': 'pass', 'scope': 'structural_budget_precheck_only', 'full_audit_required': True,
               'at_utc': utc_now(), 'reports': reports, 'checks_passed': sum(len(r['checks']) for r in reports)}
    atomic_write_json(job / 'result.json', summary)
    lease.finish(update=len(reports), status='complete', detail={'result': str(job / 'result.json')})
    print(json.dumps({'status': summary['status'], 'runs': len(reports), 'checks': summary['checks_passed'], 'full_audit_required': True}))
