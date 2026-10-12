"""Verify stage 3.2 artifacts and reconcile a stale final queue summary."""
import collections
import datetime
import json
import os
import shlex
import sys
from pathlib import Path

CODEX = Path('/remote-home/Guilin/DRO/Codex')
ROOT = CODEX / 'wdro_rest_min'
EXP = CODEX / 'experiments'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from src.io_utils import atomic_write_json
from src.job_control import utc_now
from src.rest import file_sha256
from run_stage3_baseline_queue import build_tasks
from stage3_pipeline import verify_sources, snapshot_cpu

verify_sources()
runtime = snapshot_cpu('stage3_training_completion_artifact_verification')
queue_path = EXP / 'logs/stage3_queue/stage3_baselines_formal_v1/queue.json'
queue = json.loads(queue_path.read_text())
assert queue['status'] == 'complete'
assert {t['run_id'] for t in queue['tasks']} == {t['run_id'] for t in build_tasks()}
assert all(t['status'] == 'complete' for t in queue['tasks'])
models = []
for task in queue['tasks']:
    result_path = EXP / 'logs/stage3' / task['run_id'] / 'result.json'
    result = json.loads(result_path.read_text())
    assert result['status'] == 'complete' and result['completed_updates'] == 2000
    assert result['batch_size'] == 256 and result['test_or_ispd_used'] is False
    assert result['selection_feasible'] and result['best']['nominal_mean'] <= result['nominal_limit']
    assert (result['method'], result['degree'], result['training_seed']) == (task['method'], task['degree'], task['seed'])
    checkpoints = {}
    for kind in ('latest', 'best'):
        path = Path(result[kind + '_checkpoint'])
        assert path.is_relative_to(CODEX)
        digest = file_sha256(path)
        assert digest == result[kind + '_checkpoint_sha256']
        checkpoints[kind] = {'logical_path': str(path), 'resolved_path': str(path.resolve()), 'sha256': digest}
    parent = Path(result['latest_checkpoint']).parent
    assert sorted(p.name for p in parent.glob('*.pt')) == ['best.pt', 'latest.pt']
    assert not list(parent.glob('*.partial'))
    assert not (EXP / 'logs/jobs' / task['run_id'] / 'run.lock').exists()
    models.append({'run_id': task['run_id'], 'method': task['method'], 'degree': task['degree'],
                   'seed': task['seed'], 'updates': 2000, 'checkpoints': checkpoints,
                   'result_sha256': file_sha256(result_path), 'training_finished_at_utc': result['finished_at_utc']})
expected_counts = {'complete': 45, 'running': 0, 'pending': 0, 'failed': 0, 'total': 45}
reconciled = queue['counts'] != expected_counts
slug = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
archive = None
if reconciled:
    scheduler_job = json.loads(queue_path.with_name('job.json').read_text())
    command_line = Path(f"/proc/{scheduler_job['pid']}/cmdline")
    assert not command_line.exists() or b'run_stage3_baseline_queue.py' not in command_line.read_bytes()
    archive = queue_path.with_name('queue_before_count_reconciliation_' + slug + '.json')
    atomic_write_json(archive, queue)
    queue['counts'] = expected_counts
    queue['summary_reconciled_at_utc'] = utc_now()
    atomic_write_json(queue_path, queue)
report = {'status': 'pass', 'substep': '3.2_training_completion', 'at_utc': utc_now(),
          'model_count': len(models), 'policy_updates': sum(m['updates'] for m in models),
          'checkpoint_count': 2 * len(models), 'methods': dict(collections.Counter(m['method'] for m in models)),
          'all_90_checkpoint_hashes_verified': True, 'run_locks_released': True,
          'runtime_manifest': str(runtime), 'queue_counts_reconciled': reconciled,
          'previous_queue_archive': str(archive) if archive else None, 'models': models,
          'full_stage3_acceptance': False, 'remaining': '3.3_full_input_candidate_audit_and_export',
          'reproduce_command': shlex.join(['bash', str(ROOT / 'scripts/stage3_env.sh'), sys.executable, str(Path(__file__))])}
output = EXP / 'logs/stage3_pipeline' / ('training_completion_verification_' + slug + '.json')
atomic_write_json(output, report)
print(json.dumps({'status': 'pass', 'models': len(models), 'updates': report['policy_updates'],
                  'checkpoint_hashes': 90, 'queue_reconciled': reconciled, 'report': str(output)}), flush=True)
