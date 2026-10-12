import json
import os
import shlex
import sys
import time
from pathlib import Path

CODEX = Path('/remote-home/Guilin/DRO/Codex')
ROOT = CODEX / 'wdro_rest_min'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from src.io_utils import atomic_write_json
from src.job_control import JobLease, utc_now
from stage3_pipeline import snapshot_cpu, verify_sources
from audit_stage3 import audit_run

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.nice(10)
verify_sources()
runtime = snapshot_cpu('stage3_formal_audit_cursor_replay_check')
task = {'method': 'mixed_erm', 'degree': 40, 'seed': 101,
        'run_id': 'formal_mixed_erm_d40_s101_u2000'}
cursor = CODEX / 'experiments/logs/stage3_audit/cursors' / (task['run_id'] + '.json')
before = json.loads(cursor.read_text())
assert before['status'] == 'pass' and before['completed_updates'] == 2000
job = CODEX / 'experiments/logs/stage3_audit_cursor_replay_check'
command = shlex.join(['bash', str(ROOT / 'scripts/stage3_env.sh'), sys.executable, str(Path(__file__))])
with JobLease(job, run_id='stage3_audit_cursor_replay_check', exact_command=command,
              resume_command=command, monitor_command=f'cat {job / "heartbeat.json"}',
              expected_updates=1, physical_gpu_index=None, host_runtime_manifest=str(runtime)) as lease:
    start = time.monotonic()
    result = audit_run(task)
    after = json.loads(cursor.read_text())
    assert before['binding'] == after['binding'] and before['stream_hash'] == after['stream_hash']
    assert after['completed_updates'] == 2000 and result['status'] == 'pass'
    record = {'status': 'pass', 'at_utc': utc_now(), 'run_id': task['run_id'],
              'reused_cursor_updates': 2000, 'seconds': time.monotonic() - start,
              'checks_passed': len(result['checks']), 'binding_unchanged': True,
              'cursor_logical': str(cursor), 'cursor_resolved': str(cursor.resolve())}
    atomic_write_json(job / 'result.json', record)
    lease.finish(update=1, status='complete', detail=record)
    print(json.dumps(record))
