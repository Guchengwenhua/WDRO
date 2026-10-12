"""Read-only recovery audit of the superseded v3 execution; no oracle/GPU use."""
from __future__ import annotations
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time

OLD = Path('/remote-home/Guilin/DRO/Codex/revisions/adversarial_v3_20260912/Codex')
NEW = Path('/remote-home/Guilin/DRO/Codex/revisions/adversarial_v3_1_20260912/Codex')
sys.path.insert(0, str(OLD))
import numpy as np
import torch
from adv_v3.report import verify_task, audit_arrays, load_npz, ledger_details


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def bind(path):
    return {'path': str(path), 'sha256': sha(path)}


def process(pid, start=None):
    p = Path(f'/proc/{pid}/stat')
    if not p.exists():
        return {'pid': pid, 'status': 'exited'}
    fields = p.read_text().rsplit(') ', 1)[1].split()
    same = start is None or fields[19] == str(start)
    return {'pid': pid, 'state': fields[0], 'start_ticks': fields[19],
            'same_identity': same, 'active': same and fields[0] != 'Z'}


def main():
    started = time.monotonic()
    report = {'audit_at_unix': time.time(), 'old_root': str(OLD), 'new_root': str(NEW),
              'scope': 'Recovery evidence only; superseded v3 results retained but ineligible for the final cross-method scientific study',
              'new_oracle_calls': 0, 'gpu_operations': 0, 'process_signals_sent': 0,
              'errors': [], 'tasks': []}
    parent = read(OLD / 'manifests/parent_snapshot.json')
    frozen = read(OLD / 'manifests/source_snapshot.json')
    original_errors = [r['source'] for r in parent['source_files']
                       if not Path(r['source']).is_file() or sha(r['source']) != r['sha256']]
    frozen_errors = [r['path'] for r in frozen['files']
                     if not Path(r['path']).is_file() or sha(r['path']) != r['sha256']]
    report['original_sources'] = {'verified': len(parent['source_files']), 'changed': original_errors}
    report['old_frozen_sources'] = {'verified': len(frozen['files']), 'changed': frozen_errors}
    report['errors'].extend(original_errors + frozen_errors)
    changes, absent = [], []
    for r in frozen['files']:
        new = Path(r['path'].replace(str(OLD.parent), str(NEW.parent), 1))
        if not new.is_file():
            absent.append(str(new))
        elif sha(new) != r['sha256']:
            changes.append({'old': r['path'], 'new': str(new),
                            'old_sha256': r['sha256'], 'new_sha256': sha(new)})
    report['new_scientific_source_delta_at_audit'] = {'changed': changes, 'missing': absent,
        'note': 'The new revision is still under integration; later operational/report/test edits may follow this timestamp.'}
    old_protocol, new_protocol = read(OLD/'protocol.json'), read(NEW/'protocol.json')
    report['protocol_top_level_delta'] = {k: {'old': old_protocol.get(k), 'new': new_protocol.get(k)}
        for k in sorted(set(old_protocol) | set(new_protocol)) if old_protocol.get(k) != new_protocol.get(k)}
    report['bindings'] = [bind(OLD/'manifests/parent_snapshot.json'), bind(OLD/'manifests/source_snapshot.json'),
                          bind(OLD/'protocol.json'), bind(NEW/'protocol.json'), bind(Path(__file__))]
    for task in sorted((OLD/'experiments/fixed').iterdir()):
        if not task.is_dir():
            continue
        row = {'task': task.name, 'complete_exists': (task/'complete.json').exists(),
               'failure_exists': (task/'failure.json').exists(),
               'partial_files': [str(p.relative_to(task)) for p in task.rglob('*.partial*')]}
        try:
            launch = read(task/'launch.json')
            row['process'] = process(launch['pid'], launch.get('process_start_ticks'))
            row['launch'] = launch
            row['heartbeat'] = read(task/'heartbeat.json') if (task/'heartbeat.json').exists() else None
            row['exit'] = read(task/'exit.json') if (task/'exit.json').exists() else None
            with (task/'task.lock').open('r') as f:
                try:
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    row['task_lock_released'] = True
                    fcntl.flock(f, fcntl.LOCK_UN)
                except BlockingIOError:
                    row['task_lock_released'] = False
            if not row['complete_exists']:
                row['local_status'] = 'incomplete'
                report['tasks'].append(row)
                continue
            done0 = read(task/'complete.json')
            done, listed = verify_task(task, (done0['parent'], done0['degree'], done0['seed']), 'fixed')
            row['completion_binding'] = bind(task/'complete.json')
            row['artifact_bindings_verified'] = len(listed)
            row['ledger'] = ledger_details(done['ledger'])
            persisted = read(task/'oracle/oracle_ledger.json')
            assert persisted == done['ledger'], 'persisted ledger differs from complete'
            expected_total = 1024 * (4 * 4 + 5 * 64) + 8192 + 1024 + 256 * 64 * 2 + 8 * 128 * 2
            for k in ['actor_requested', 'actor_completed', 'rest_requested', 'rest_completed', 'reference_requests']:
                assert row['ledger'][k] == expected_total, (k, row['ledger'][k], expected_total)
            assert not any(row['ledger'][k] for k in ['actor_uncertain_requests', 'rest_uncertain_requests', 'reference_uncertain_misses'])
            for parent_key in ['parent', 'price_parent']:
                meta = done['identity'][parent_key]
                assert sha(meta['logical_path']) == meta['sha256'], parent_key + ' checkpoint changed'
            chunks = nets = search_calls = 0
            mismatches = []
            for distribution in ['uniform', 'mix_gauss', 'edge_bias', 'grid_cluster']:
                centers = load_npz(task/'inputs'/f'{distribution}.npz')['pins']
                for method in done['methods']:
                    budget = 4 if method in done['methods'][:4] else 64
                    for i in range(0, 256, 64):
                        file = task/'evaluation'/f'{distribution}_{method}_{i:05d}.npz'
                        data = load_npz(file)
                        audit_arrays(data, centers[i:i+64], degree=done['degree'], budget=budget,
                                     price=done['price'], context=str(file))
                        baseline = load_npz(task/'evaluation'/f'{distribution}_random_top4_{i:05d}.npz')
                        differences = np.abs(data['original_regret'] - baseline['original_regret'])
                        for j in np.flatnonzero(differences > 2e-9):
                            mismatches.append({'distribution': distribution, 'method': method,
                                'center': int(i+j), 'absolute_difference': float(differences[j]),
                                'method_original_regret': float(data['original_regret'][j]),
                                'random_top4_original_regret': float(baseline['original_regret'][j])})
                        chunks += 1
                        nets += len(data['pins'])
                        search_calls += int(data['calls'].sum())
            row.update(npz_chunks_audited=chunks, selected_nets_audited=nets,
                       real_search_calls=search_calls, real_total_calls=expected_total,
                       cross_method_original_mismatches=mismatches,
                       local_status='complete_engineering_and_accounting_pass',
                       scientific_eligibility='excluded_superseded_batchshape_revision')
            checkpoints = {}
            for learner in ['critic', 'rl']:
                checkpoints[learner] = {}
                for kind in ['latest', 'best']:
                    path = task/learner/(kind+'.pt')
                    state = torch.load(path, map_location='cpu', weights_only=False)
                    checkpoints[learner][kind] = dict(bind(path),
                        epoch=state.get('epoch'), update=state.get('update'),
                        has_optimizer_state=any('optimizer' in k for k in state),
                        has_rng_state=any('rng' in k for k in state),
                        history_records=len(state.get('history', [])))
                    if kind == 'latest':
                        assert state.get('epoch' if learner == 'critic' else 'update') == (16 if learner == 'critic' else 256)
            row['checkpoints'] = checkpoints
        except Exception as exc:
            row['audit_error'] = repr(exc)
            report['errors'].append({'task': task.name, 'error': repr(exc)})
        report['tasks'].append(row)
        print(json.dumps({'task': task.name, 'status': row.get('local_status'),
                          'error': row.get('audit_error'), 'mismatches': len(row.get('cross_method_original_mismatches', []))}), flush=True)
    report['old_joint_task_count'] = len(list((OLD/'experiments/joint').glob('*/complete.json')))
    report['new_fixed_task_count_at_audit'] = len(list((NEW/'experiments/fixed').glob('*/complete.json')))
    report['new_joint_task_count_at_audit'] = len(list((NEW/'experiments/joint').glob('*/complete.json')))
    report['known_controller_pids_at_audit'] = [process(pid) for pid in [348424, 348440, 349383]]
    report['totals'] = {k: sum(t.get(k, 0) for t in report['tasks']) for k in
                        ['artifact_bindings_verified', 'npz_chunks_audited', 'selected_nets_audited',
                         'real_search_calls', 'real_total_calls']}
    report['totals']['cross_method_original_mismatch_rows'] = sum(len(t.get('cross_method_original_mismatches', [])) for t in report['tasks'])
    report['totals']['locally_complete_tasks'] = sum(t.get('local_status') == 'complete_engineering_and_accounting_pass' for t in report['tasks'])
    report['totals']['live_old_workers'] = sum(t.get('process', {}).get('active', False) for t in report['tasks'])
    report['status'] = 'recovery_audit_pass_old_study_ineligible' if not report['errors'] else 'recovery_audit_failed'
    report['wall_seconds'] = time.monotonic()-started
    target = NEW/'experiments/diagnostics/network_recovery_audit.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name + f'.partial.{os.getpid()}')
    temp.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    temp.replace(target)
    print(json.dumps({'path': str(target), 'status': report['status'], 'totals': report['totals'],
                      'errors': report['errors'], 'new_scientific_changes': changes}, indent=2), flush=True)


if __name__ == '__main__':
    main()
