"""Record live sessions, evidence bindings, paths, and the Stage 6 gate."""
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ORIGINAL = Path('/remote-home/Guilin/DRO/Codex')
ROOT = ORIGINAL / 'revisions/reference_precision_v2/Codex/wdro_rest_min'
EXP = ROOT.parent / 'experiments'
JOB = Path(__file__).parent
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from src.io_utils import atomic_write_json
from src.job_control import utc_now
from src.rest import file_sha256
from precision_replay_pipeline import cpu_preflight, progress
import precision_stage5_capacity_pipeline as pipeline
import precision_queue_runtime as runtime


def artifact(path):
    path = Path(path)
    assert path.is_relative_to(ORIGINAL)
    return dict(logical_path=str(path), resolved_path=str(path.resolve()), sha256=file_sha256(path))


def read(path):
    return json.loads(path.read_text())


pipeline.gate()
preflight = cpu_preflight('stage6_1434_continuation_postlaunch')
sessions = []
environment_keys = ('TMPDIR', 'TMP', 'TEMP', 'XDG_CACHE_HOME', 'TORCH_HOME',
    'CUDA_CACHE_PATH', 'MPLCONFIGDIR', 'TRITON_CACHE_DIR', 'PYTHONPYCACHEPREFIX')
for name in ('wdro_replay', 'stage6_completion', 'stage5_random64_audit_wait_1434'):
    job = ORIGINAL / 'experiments/logs/precision_v2/jobs' / name
    launch = read(job / 'launch.json')
    child = read(job / 'child.json')
    heartbeat = read(job / 'heartbeat.json')
    assert runtime.process_is_alive(heartbeat['pid']) and runtime.process_is_alive(child['pid'])
    assert (datetime.now(timezone.utc) - datetime.fromisoformat(heartbeat['last_update_utc'].replace('Z', '+00:00'))).total_seconds() < 60
    assert subprocess.run(['tmux', '-S', launch['socket'], 'has-session', '-t', launch['session']], capture_output=True).returncode == 0
    rows = []
    for role, pid in [('wrapper', heartbeat['pid']), ('child', child['pid'])]:
        proc = Path('/proc') / str(pid)
        command = proc.joinpath('cmdline').read_bytes().decode().split('\0')[:-1]
        expected = str(ORIGINAL / 'wdro_rest_min/scripts/precision_job.py') if role == 'wrapper' else child['command'][3]
        assert expected in command, (pid, expected)
        env = dict(item.split('=', 1) for item in proc.joinpath('environ').read_bytes().decode().split('\0') if '=' in item)
        chosen = {key: env[key] for key in environment_keys if key in env}
        assert chosen and all(Path(value).is_relative_to(ORIGINAL) for value in chosen.values())
        rows.append(dict(role=role, pid=pid, command=command, cwd=str(proc.joinpath('cwd').resolve()),
            start_ticks=proc.joinpath('stat').read_text().split(') ', 1)[1].split()[19], runtime_directories=chosen))
    sessions.append(dict(name=name, launch=artifact(job / 'launch.json'), session=launch['session'],
        recovery_command=launch['recovery_command'], monitor_command=launch['monitor_command'],
        heartbeat=heartbeat, processes=rows))

checkpoint = max(JOB.glob('live_checkpoint_review_*.json'))
assert read(checkpoint)['status'] == 'pass' and read(checkpoint)['checks'] == 32
seven = max(JOB.glob('seven_completed_review_*.json'))
assert read(seven)['full_audit_checks'] == 308 and read(seven)['validation_nets'] == 56000
assert read(JOB / 'waiter_foreground_acceptance.json')['status'] == 'pass'
assert read(JOB / 'duplicate_launch_check.json')['status'] == 'pass'
status_command = ['bash', str(ORIGINAL / 'wdro_rest_min/scripts/stage3_env.sh'), sys.executable,
    str(ORIGINAL / 'wdro_rest_min/scripts/precision_status.py')]
status_run = subprocess.run(status_command, capture_output=True, text=True, check=True)
status = json.loads(status_run.stdout.splitlines()[0])
assert status['stage6'] == 'DOING' and status['corrected_test_cells_complete'] == 0
assert status['wdro_validation']['full_audits'] == status['wdro_validation']['best_validation_runs'] == 7
protocols = []
for name, digest in [('protocol_v1.json', '077505bccd33153117745f1e004943c262305234e39d8ebebb4a53b470a60850'),
                     ('data_contract_v1.json', '2121d1119bc20dcd493183fcab4ce8aaa986eee84523e487fb42c2cc29612055')]:
    item = artifact(ORIGINAL / 'wdro_rest_min/configs' / name)
    assert item['sha256'] == digest
    protocols.append(item)
checks = dict(three_sessions_live=True, child_identity_verified=True, heartbeat_fresh=True,
    runtime_directories_in_codex=True, original_protocol_hashes_unchanged=True, frozen_pipeline_gate_passed=True,
    seven_completed_runs_reverified=True, four_checkpoint_states_pass=True,
    foreground_smoke_pass=True, duplicate_launch_refused=True,
    fresh_postlaunch_preflight=True, corrected_test_not_started=True)
output = JOB / ('continuation_review_' + str(time.time_ns()) + '.json')
value = dict(status='pass', at_utc=utc_now(), checks=checks, source=artifact(Path(__file__)),
    preflight=artifact(preflight), status_snapshot=artifact(Path(status['manifest'])), live_status=status,
    sessions=sessions, evidence=[artifact(p) for p in (seven, checkpoint, JOB / 'waiter_foreground_acceptance.json',
        JOB / 'duplicate_launch_check.json', JOB / 'review_input_order_correction.json', JOB / 'checkpoint_schema_review_correction.json')],
    protocols=protocols, stage6_complete=False, stage7_started=False, research_protocol_override=False)
atomic_write_json(output, value, mode=0o444)
progress('续跑交接复核12项通过：7/9完整审计与best复算已验收；两项训练和Random-64训练后审计交接在tmux，Stage6门禁控制器持续等待纠错版validation冻结', str(output))
print(json.dumps(dict(status='pass', checks=len(checks), evidence=str(output),
    sessions=[dict(name=s['name'], pids=[p['pid'] for p in s['processes']]) for s in sessions])), flush=True)
