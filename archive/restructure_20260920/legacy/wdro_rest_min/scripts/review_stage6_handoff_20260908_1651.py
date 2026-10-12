"""Verify the R5 final-audit handoff without starting dependent experiments."""
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ORIGINAL = Path(__file__).resolve().parents[2]
CODEX = ORIGINAL / 'revisions/reference_precision_v2/Codex'
ROOT = CODEX / 'wdro_rest_min'
EXP = CODEX / 'experiments'
JOB = EXP / 'logs/stage6_remaining_20260908_1651'
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from src.io_utils import atomic_write_json
from src.job_control import utc_now
from src.rest import file_sha256
from precision_replay_pipeline import cpu_preflight
from precision_stage5_completed_batch import check_files, verified
from precision_stage5_capacity_pipeline import gate


def read(path):
    return json.loads(Path(path).read_text())


def artifact(path):
    path = Path(path)
    assert path.is_relative_to(ORIGINAL)
    return dict(logical_path=str(path), resolved_path=str(path.resolve()), sha256=file_sha256(path))


def process(pid, script):
    proc = Path('/proc') / str(pid)
    stat = proc.joinpath('stat').read_text().rsplit(')', 1)[1].split()
    assert stat[0] not in ('Z', 'X')
    argv = proc.joinpath('cmdline').read_bytes().decode().rstrip('\0').split('\0')
    cwd = proc.joinpath('cwd').resolve()
    assert any((Path(a) if Path(a).is_absolute() else cwd / a).resolve() == script.resolve()
               for a in argv if a.endswith('.py'))
    env = dict(item.split('=', 1) for item in proc.joinpath('environ').read_bytes().decode().split('\0') if '=' in item)
    names = ['TMPDIR', 'TMP', 'TEMP', 'XDG_CACHE_HOME', 'TORCH_HOME', 'CUDA_CACHE_PATH', 'MPLCONFIGDIR', 'TRITON_CACHE_DIR']
    controlled = {name: env[name] for name in names}
    assert all(Path(value).is_relative_to(ORIGINAL) for value in controlled.values())
    assert env['PYTHONDONTWRITEBYTECODE'] == '1'
    return dict(pid=pid, state=stat[0], start_ticks=int(stat[19]), cwd=str(cwd), argv=argv, controlled_environment=controlled)


def recent(path):
    value = read(path)
    when = datetime.fromisoformat(value['last_update_utc'].replace('Z', '+00:00'))
    age = (datetime.now(timezone.utc) - when).total_seconds()
    assert 0 <= age < 180, (str(path), age)
    return dict(path=str(path), captured_value=value, age_seconds=age)


def main():
    gate()
    preflight = cpu_preflight('stage6_1651_handoff_review')
    training_path = EXP / 'logs/stage5_final_training_review/training_verification.json'
    training = read(training_path)
    assert training['status'] == 'pass' and training['run_count'] == 9 and training['updates'] == 18000
    assert training['check_count'] == 162
    for row in training['runs']:
        assert all(row['checks'].values())
        check_files(row['artifacts'])
    batch = EXP / 'logs/stage5_completed_batches/stage6_20260908_1651'
    plan = read(batch / 'plan.json')
    check_files(plan['files'])
    records = [value for task in plan['tasks'] if (value := verified(task))]
    assert len(records) >= 8
    prior = read(JOB / 'eight_completed_acceptance.json')
    for old in prior['runs']:
        assert old == next(r for r in records if r['run_id'] == old['run_id'])
    assert not (batch / 'run.lock').exists()
    frozen_inputs = []
    for name in ('protocol_v1.json', 'data_contract_v1.json'):
        original = ORIGINAL / 'wdro_rest_min/configs' / name
        revision = ROOT / 'configs' / name
        assert file_sha256(original) == file_sha256(revision)
        assert original.stat().st_mode & 0o222 == revision.stat().st_mode & 0o222 == 0
        frozen_inputs += [artifact(original), artifact(revision)]
    for name in ('stage2_completion_20260905', 'stage3_delivery_verification_v1', 'stage4_delivery_verification_v1'):
        path = ROOT / 'manifests' / (name + '.json')
        assert read(path)['status'] == 'pass'
        frozen_inputs.append(artifact(path))
    controllers = []
    for name, script in [('wdro_replay', 'precision_wdro_replay.py'), ('stage6_completion', 'precision_stage6_completion.py')]:
        base = ORIGINAL / 'experiments/logs/precision_v2/jobs' / name
        launch = read(base / 'launch.json')
        heartbeat = recent(base / 'heartbeat.json')
        child = read(base / 'child.json')
        assert subprocess.run(['tmux', '-S', launch['socket'], 'has-session', '-t', launch['session']], capture_output=True).returncode == 0
        controllers.append(dict(name=name, launch=artifact(base / 'launch.json'), session=launch['session'], socket=launch['socket'],
            wrapper=process(heartbeat['captured_value']['pid'], ORIGINAL / 'wdro_rest_min/scripts/precision_job.py'),
            child=process(child['pid'], ROOT / 'scripts' / script), heartbeat=heartbeat,
            exact_command=shlex.join(launch['command']), log=launch['log'], log_resolved=launch['log_resolved'],
            monitor_command=launch['monitor_command'], recovery_command=launch['recovery_command']))
    pipeline_heartbeat = recent(EXP / 'logs/stage5_pipeline/heartbeat.json')
    pipeline = process(pipeline_heartbeat['captured_value']['pid'], ROOT / 'scripts/precision_stage5_capacity_pipeline.py')
    rid = 'stage5_cem_64_d30_s101_u2000_v1'
    cursor_path = EXP / 'logs/stage5_audit/cursors' / (rid + '.json')
    cursor = read(cursor_path)
    assert cursor['heldout_intersections'] == 0
    assert cursor['binding']['result_sha256'] == file_sha256(EXP / 'logs/stage5' / rid / 'result.json')
    assert cursor['binding']['auditor_sha256'] == file_sha256(ROOT / 'scripts/audit_stage5.py')
    audit_heartbeat = recent(EXP / 'logs/stage5_audit/jobs' / rid / 'heartbeat.json')
    auditor = process(audit_heartbeat['captured_value']['pid'], ROOT / 'scripts/audit_stage5.py')
    assert not (ROOT / 'configs/stage5_selected_wdro_v1.json').exists()
    assert not list((EXP / 'logs/stage6_test').glob('*/*/result.json'))
    assert not (ORIGINAL / 'wdro_rest_min/manifests/precision_v2_replay_completion.json').exists()
    review = dict(status='pass', at_utc=utc_now(), scope='R5 training completion and eight-run acceptance; remaining original audit and downstream gates stay mandatory',
        source=artifact(Path(__file__)), preflight=artifact(preflight), frozen_inputs=frozen_inputs, training_review=artifact(training_path),
        completed_acceptance=artifact(JOB / 'eight_completed_acceptance.json'), completed_runs=records,
        full_audit_checks=sum(r['full_audit_checks'] for r in records), validation_nets=sum(r['validation_nets'] for r in records),
        controllers=controllers, pipeline=dict(process=pipeline, heartbeat=pipeline_heartbeat),
        remaining_audit=dict(process=auditor, heartbeat=audit_heartbeat, cursor=cursor, cursor_logical_path=str(cursor_path), cursor_resolved_path=str(cursor_path.resolve())),
        corrected_test_cells=0, stage6_complete=False, stage7_started=False, research_protocol_override=False)
    path = JOB / ('continuation_review_' + str(time.time_ns()) + '.json')
    atomic_write_json(path, review, mode=0o444)
    index = ORIGINAL / 'wdro_rest_min/manifests/stage6_continuation_review_20260908_1651.json'
    atomic_write_json(index, dict(status='pass', at_utc=review['at_utc'], review=artifact(path), source=artifact(Path(__file__)),
        stage6='DOING', stage6_complete=False, stage7_started=False), mode=0o444)
    print(json.dumps(dict(status='pass', manifest=str(index), review=str(path), cem_audit_update=cursor['completed_updates'],
        full_audits=len(records), validation_nets=review['validation_nets']), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
