#!/usr/bin/env python3
"""Audit four completed ablations with the unchanged full auditor, one at a time."""
import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODEX = ROOT.parent
EXP = CODEX / 'experiments'
sys.path.insert(0, str(ROOT))
from src.io_utils import atomic_write_json, append_jsonl
from src.job_control import JobLease, utc_now, process_is_alive
from src.rest import file_sha256
from stage5_pipeline import gate, observe

JOB = EXP / 'logs/stage5_completed_audit'
PIPELINE = EXP / 'logs/stage5_pipeline'
ORDER = ('k1', 'k8', 'chamfer', 'lambda_fixed')


def preflight(purpose):
    cmd = [sys.executable, str(ROOT / 'scripts/capture_host_runtime.py'),
           '--purpose', purpose, '--measured-peak-mib', '2654', '--safety-margin-mib', '2048']
    done = subprocess.run(cmd, capture_output=True, text=True)
    append_jsonl(JOB / 'preflights.jsonl', {'at_utc': utc_now(), 'command': cmd,
        'exit_code': done.returncode, 'stdout': done.stdout, 'stderr': done.stderr})
    if done.returncode:
        raise RuntimeError('Fresh preflight failed: ' + done.stdout + done.stderr)
    return json.loads(done.stdout.strip().splitlines()[-1])


def valid_report(rid):
    path = EXP / 'logs/stage5_audit/runs' / (rid + '.json')
    if not path.exists():
        return None
    report = json.loads(path.read_text())
    assert report['status'] == 'pass' and all(report['checks'].values()), rid
    assert report['completed_updates'] == 2000
    assert report['binding']['result_sha256'] == file_sha256(EXP / 'logs/stage5' / rid / 'result.json')
    return {'run_id': rid, 'path': str(path), 'resolved_path': str(path.resolve()),
            'sha256': file_sha256(path), 'checks': len(report['checks'])}


def update_progress(records):
    path = CODEX / 'docs/01_WDRO-REST_项目进度.md'
    marker = '- 阶段5完成项提前审计：'
    line = (f'{marker}{len(records)}/4 个已结束消融通过原完整审计器（{utc_now()}）；'
            '证据 `experiments/logs/stage5_completed_audit/result.json` 与 `stage5_audit/runs/`。'
            '单进程 nice=19，原训练继续；最终仍须 9/9 训练/审计和 5.4 交付复核。')
    value = path.read_text()
    if marker in value:
        lines = value.splitlines()
        lines = [line if old.startswith(marker) else old for old in lines]
        value = '\n'.join(lines) + '\n'
    else:
        pos = value.index('\n', value.index('- `DOING` 5.3：'))
        value = value[:pos+1] + line + '\n' + value[pos+1:]
    partial = path.with_name(path.name + '.completed_audit.partial')
    partial.write_text(value)
    partial.replace(path)


def stop_on_failure(reason):
    # Stop only this project's controller and its own training children.
    stopped = []
    queue = json.loads((PIPELINE / 'queue.json').read_text())
    jobs = [(json.loads((PIPELINE / 'job.json').read_text())['pid'], 'stage5_pipeline.py')]
    jobs += [(a['pid'], 'train_stage5.py') for a in queue.get('active', [])]
    for pid, script in jobs:
        cmd = Path(f'/proc/{pid}/cmdline')
        if cmd.exists() and str(ROOT / 'scripts' / script).encode() in cmd.read_bytes():
            try:
                os.kill(pid, signal.SIGTERM)
                stopped.append({'pid': pid, 'script': script})
            except ProcessLookupError:
                pass
    atomic_write_json(JOB / 'failure.json', {'at_utc': utc_now(), 'reason': reason,
        'stop_requested': stopped, 'recovery': 'Resolve failure before launch_stage5.py; resume latest checkpoints.'})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--max-runs', type=int)
    p.add_argument('--smoke-pause-updates', type=int)
    args = p.parse_args()
    JOB.mkdir(parents=True, exist_ok=True)
    assert not (JOB / 'failure.json').exists(), 'Unresolved failure: inspect failure.json'
    gate()
    contract = json.loads((JOB / 'operations.json').read_text())
    for path, digest in contract['files'].items():
        assert file_sha256(Path(path)) == digest, 'completed-audit operation changed: ' + path
    tasks = json.loads((PIPELINE / 'all_tasks.json').read_text())['tasks']
    tasks = [next(t for t in tasks if t['run_id'] == f'stage5_{name}_d30_s101_u2000_v1') for name in ORDER]
    command = shlex.join(['bash', str(ROOT / 'scripts/stage3_env.sh'), sys.executable, str(Path(__file__)), *sys.argv[1:]])
    with JobLease(JOB, run_id='stage5_completed_audit', exact_command=command,
        resume_command=f'bash {ROOT}/scripts/stage3_env.sh {sys.executable} {ROOT}/scripts/launch_stage5_completed_audit.py',
        monitor_command=f'cat {JOB}/heartbeat.json', expected_updates=4,
        physical_gpu_index=None, host_runtime_manifest=os.environ.get('WDRO_HOST_RUNTIME_MANIFEST')) as lease:
        records = [r for t in tasks if (r := valid_report(t['run_id']))]
        newly = 0
        for t in tasks:
            rid = t['run_id']
            if valid_report(rid):
                continue
            gate()
            # The main queue will adopt any live audit and reuse its final report.
            if json.loads((PIPELINE / 'heartbeat.json').read_text())['status'] != '5.3_ablations':
                atomic_write_json(JOB / 'handoff.json', {'at_utc': utc_now(), 'status': 'main_pipeline_owns_remaining_audits'})
                break
            result = json.loads((EXP / 'logs/stage5' / rid / 'result.json').read_text())
            assert result['status'] == 'complete' and result['completed_updates'] == 2000
            assert not (EXP / 'logs/jobs' / rid / 'run.lock').exists()
            assert file_sha256(Path(t['config'])) == t['config_sha256']
            runtime = preflight('stage5_completed_audit_' + rid)
            audit_job = EXP / 'logs/stage5_audit/jobs' / rid
            if (audit_job / 'run.lock').exists():
                pid = json.loads((audit_job / 'run.lock').read_text())['pid']
                assert not process_is_alive(pid), f'Auditor already active: {pid}'
            cmd = ['bash', str(ROOT / 'scripts/stage3_env.sh'), 'nice', '-n', '19', sys.executable,
                   str(ROOT / 'scripts/audit_stage5.py'), '--run-id', rid, '--stability']
            log = JOB / (rid + '.log')
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='', WDRO_HOST_RUNTIME_MANIFEST=runtime['manifest'])
            with log.open('a') as output:
                proc = subprocess.Popen(cmd, stdout=output, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            append_jsonl(JOB / 'launches.jsonl', {'at_utc': utc_now(), 'run_id': rid, 'pid': proc.pid,
                'command': cmd, 'exact_command': shlex.join(cmd), 'log': str(log), 'log_resolved': str(log.resolve()),
                'preflight': runtime, 'gpu': None, 'nice': 19})
            append_jsonl(JOB / 'resources.jsonl', dict(observe(), run_id=rid, pid=proc.pid, event='postlaunch'))
            try:
                while proc.poll() is None:
                    gate()
                    cursor = EXP / 'logs/stage5_audit/cursors' / (rid + '.json')
                    saved = json.loads(cursor.read_text()) if cursor.exists() else None
                    lease.update(update=len(records), status='running', detail={'run_id': rid, 'pid': proc.pid,
                        'cursor': str(cursor), 'audit_updates': saved['completed_updates'] if saved else 0})
                    if args.smoke_pause_updates and saved and saved['completed_updates'] >= args.smoke_pause_updates:
                        proc.terminate()
                        proc.wait(timeout=30)
                        assert saved['binding']['result_sha256'] == file_sha256(EXP / 'logs/stage5' / rid / 'result.json')
                        assert not (EXP / 'logs/stage5_audit/runs' / (rid + '.json')).exists()
                        atomic_write_json(JOB / 'foreground_smoke.json', {'status': 'pass', 'at_utc': utc_now(),
                            'run_id': rid, 'cursor': str(cursor), 'completed_updates': saved['completed_updates'],
                            'cursor_sha256': file_sha256(cursor), 'log': str(log), 'pid': proc.pid,
                            'preflight': runtime, 'resume_required': True, 'final_audit_required': True})
                        lease.finish(update=0, status='paused_smoke')
                        return
                    time.sleep(15)
                assert proc.returncode == 0, f'Audit exited {proc.returncode}: {log}'
                report = valid_report(rid)
                assert report is not None
            except KeyboardInterrupt:
                if proc.poll() is None:
                    proc.terminate()
                    proc.wait(timeout=30)
                raise
            except Exception as error:
                if proc.poll() is None:
                    proc.terminate()
                    proc.wait(timeout=30)
                stop_on_failure(f'{rid}: {type(error).__name__}: {error}')
                raise
            records.append(report)
            newly += 1
            atomic_write_json(JOB / 'result.json', {'at_utc': utc_now(), 'status': 'pass' if len(records) == 4 else 'partial_pass',
                'runs': records, 'total': 4, 'stage5_complete': False, 'stage6_started': False})
            update_progress(records)
            print(json.dumps(report), flush=True)
            if args.max_runs and newly >= args.max_runs:
                break
        lease.finish(update=len(records), status='complete', detail={'result': str(JOB / 'result.json')})


if __name__ == '__main__':
    main()
