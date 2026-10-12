#!/usr/bin/env python3
"""Review all finished training states and replay the last two validation models.

Full candidate audits remain independent mandatory inputs to the final export.
"""
import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODEX = ROOT.parent
EXP = CODEX / 'experiments'
JOB = EXP / 'logs/stage5_final_training_review'
sys.path.insert(0, str(ROOT))
from src.io_utils import atomic_write_json, append_jsonl
from src.job_control import JobLease, utc_now
from src.rest import file_sha256
from src.stage2_audit_checks import all_finite, optimizer_checks
from src.trainer import load_trainer_config
from stage5_pipeline import gate, observe


def record(path):
    path = Path(path)
    assert path.is_relative_to(CODEX)
    return {'logical_path': str(path), 'resolved_path': str(path.resolve()), 'sha256': file_sha256(path)}


def training_review():
    import torch
    tasks = json.loads((EXP / 'logs/stage5_pipeline/all_tasks.json').read_text())['tasks']
    assert len(tasks) == 9 and len({t['run_id'] for t in tasks}) == 9
    rows = []
    for task in tasks:
        rid = task['run_id']
        rp = EXP / 'logs/stage5' / rid / 'result.json'
        result = json.loads(rp.read_text())
        assert result['status'] == 'complete' and result['completed_updates'] == 2000
        assert result['degree'] == 30 and result['training_seed'] == 101 and result['batch_size'] == 256
        assert file_sha256(Path(task['config'])) == task['config_sha256']
        assert result['run_identity']['ablation_config'] == json.loads(Path(task['config']).read_text())
        assert not result['test_or_ispd_used'] and result['selection_feasible']
        assert result['best']['nominal_mean'] <= result['nominal_limit']
        checks = {'finished_frozen_training': True}
        artifacts = [record(rp), record(task['config'])]
        for kind in ('latest', 'best'):
            path = Path(result[kind + '_checkpoint'])
            assert file_sha256(path) == result[kind + '_checkpoint_sha256']
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            assert checkpoint['run_identity'] == result['run_identity']
            assert checkpoint['update'] == (2000 if kind == 'latest' else result['best']['update'])
            checks.update({kind + '_' + k: v for k, v in optimizer_checks(checkpoint, load_trainer_config()).items()})
            checks[kind + '_all_finite'] = all_finite(checkpoint)
            checks[kind + '_selected_best_matches'] = checkpoint['best'] == result['best']
            artifacts.append(record(path))
            if kind == 'latest':
                assert checkpoint['call_ledger'] == result['call_ledger']
                assert checkpoint['method_state'] == result['method_state']
                assert checkpoint['training_stream_hash'] == result['training_stream_hash']
            del checkpoint
        assert not (EXP / 'logs/jobs' / rid / 'run.lock').exists()
        assert sorted(p.name for p in Path(result['latest_checkpoint']).parent.iterdir()) == ['best.pt', 'latest.pt']
        search = result['run_identity']['ablation_config']['search_overrides']['searcher']
        budget = result['method_state']['budget']
        if search in ('random_64', 'cem_64'):
            assert budget['exact_rescore_nets'] == budget['reference_requests'] == 32768000
            assert budget['surrogate_nets'] == 0
        checks['checkpoint_identity_hashes_and_call_ledger'] = True
        checks['no_training_lock_or_partial'] = True
        checks['declared_equal_search_budget'] = True
        assert all(checks.values()), (rid, checks)
        rows.append({'run_id': rid, 'checks': checks, 'artifacts': artifacts,
            'best': result['best'], 'budget': budget, 'call_ledger': result['call_ledger']})
    output = {'status': 'pass', 'at_utc': utc_now(), 'scope': 'training completion, checkpoints and declared budget; full candidate audit still required',
        'runs': rows, 'run_count': 9, 'updates': 18000, 'check_count': sum(len(x['checks']) for x in rows),
        'stage5_complete': False, 'stage6_started': False}
    atomic_write_json(JOB / 'training_verification.json', output)
    return tasks


def verified(rid):
    path = EXP / 'logs/stage5_best_validation' / (rid + '.json')
    if not path.exists():
        return None
    v = json.loads(path.read_text())
    rp = EXP / 'logs/stage5' / rid / 'result.json'
    result = json.loads(rp.read_text())
    assert v['status'] == 'pass' and v['exact_metric_equality'] and v['validation_nets'] == 8000
    assert v['result_sha256'] == file_sha256(rp)
    assert v['checkpoint_sha256'] == result['best_checkpoint_sha256']
    assert v['validation_manifest_sha256'] == result['best']['validation_manifest_sha256']
    assert v['selected_update'] == result['best']['update'] and not v['test_ispd_used']
    assert v['call_ledger'] == {'actor_forward_nets': 8000, 'exact_rest_evaluator_nets': 8000,
                              'geosteiner_nets': 0, 'surrogate_evaluator_nets': 0}
    return dict(record(path), run_id=rid)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-new-runs', type=int)
    args = parser.parse_args()
    JOB.mkdir(parents=True, exist_ok=True)
    gate()
    contract = json.loads((JOB / 'operations_v1.json').read_text())
    for p, digest in contract['files'].items():
        assert file_sha256(Path(p)) == digest, p
    command = shlex.join(['bash', str(ROOT / 'scripts/stage3_env.sh'), sys.executable, str(Path(__file__)), *sys.argv[1:]])
    with JobLease(JOB, run_id='stage5_final_training_review', exact_command=command,
        resume_command=f'bash {ROOT}/scripts/stage3_env.sh {sys.executable} {Path(__file__)}',
        monitor_command=f'cat {JOB}/heartbeat.json', expected_updates=2, physical_gpu_index=None,
        host_runtime_manifest=os.environ.get('WDRO_HOST_RUNTIME_MANIFEST')) as lease:
        tasks = training_review()
        targets = [t['run_id'] for t in tasks if any(x in t['run_id'] for x in ('random_64', 'cem_64'))]
        assert len(targets) == 2
        records = [v for rid in targets if (v := verified(rid))]
        new = 0
        for rid in targets:
            if verified(rid):
                continue
            gate()
            cmd = [sys.executable, str(ROOT / 'scripts/capture_host_runtime.py'), '--purpose', 'stage5_final_best_replay_' + rid,
                   '--measured-peak-mib', '2654', '--safety-margin-mib', '2048']
            done = subprocess.run(cmd, capture_output=True, text=True)
            append_jsonl(JOB / 'preflights.jsonl', {'at_utc': utc_now(), 'command': cmd,
                'exit_code': done.returncode, 'stdout': done.stdout, 'stderr': done.stderr})
            assert done.returncode == 0, done.stdout + done.stderr
            runtime = json.loads(done.stdout.strip().splitlines()[-1])
            host = json.loads(Path(runtime['manifest']).read_text())
            gpu = max((x for x in host['gpu_decisions'] if x['eligible']), key=lambda x: x['minimum_free_memory_mib'])['index']
            cmd = ['bash', str(ROOT / 'scripts/stage3_env.sh'), sys.executable, str(ROOT / 'scripts/verify_stage5_best_validation.py'),
                   '--run-id', rid, '--gpu', str(gpu), '--preflight', runtime['manifest']]
            log = JOB / (rid + f'_{int(time.time())}.log')
            with log.open('a') as out:
                proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT)
            active = {'at_utc': utc_now(), 'run_id': rid, 'pid': proc.pid, 'gpu': gpu,
                      'command': cmd, 'exact_command': shlex.join(cmd), 'log': str(log), 'log_resolved': str(log.resolve()), 'preflight': runtime}
            append_jsonl(JOB / 'launches.jsonl', active)
            try:
                while proc.poll() is None:
                    snapshot = observe()
                    append_jsonl(JOB / 'resources.jsonl', dict(snapshot, run_id=rid, pid=proc.pid))
                    lease.update(update=len(records), status='running', detail=active)
                    assert next(x for x in snapshot['gpus'] if x['index'] == gpu)['memory_free_mib'] >= 2048, 'VRAM margin insufficient; retry this stateless replay'
                    assert snapshot['disk'][1]['free_gib'] >= 100
                    time.sleep(5)
                assert proc.returncode == 0, str(log)
                records.append(verified(rid))
            finally:
                if proc.poll() is None:
                    proc.terminate()
                    proc.wait(timeout=30)
            new += 1
            atomic_write_json(JOB / 'validation_result.json', {'status': 'pass' if len(records) == 2 else 'partial_pass',
                'at_utc': utc_now(), 'runs': records, 'validation_nets': 8000 * len(records),
                'candidate_audits_remain_mandatory': True, 'stage5_complete': False, 'stage6_started': False})
            print(json.dumps({'run_id': rid, 'validation_status': 'pass', 'completed': len(records)}), flush=True)
            if args.max_new_runs and new >= args.max_new_runs:
                break
        lease.finish(update=len(records), status='complete' if len(records) == 2 else 'paused_after_smoke')


if __name__ == '__main__':
    main()
