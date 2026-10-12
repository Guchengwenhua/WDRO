#!/usr/bin/env python3
"""Independently replay validation for seven fully audited, immutable runs."""
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
from src.job_control import JobLease, process_is_alive, utc_now
from src.rest import file_sha256
from stage5_pipeline import gate, observe

JOB = EXP / 'logs/stage5_validation_completed'
SCRIPT = Path(__file__)
SESSION = 'wdro_stage5_validation_completed_v1'
SOCKET = EXP / 'tmp/stage5.sock'


def check_input(task):
    rid = task['run_id']
    for item in task['inputs']:
        p = Path(item['logical_path'])
        assert p.is_relative_to(CODEX) and str(p.resolve()) == item['resolved_path']
        assert file_sha256(p) == item['sha256'], str(p)
    rp = EXP / 'logs/stage5' / rid / 'result.json'
    r = json.loads(rp.read_text())
    a = json.loads((EXP / 'logs/stage5_audit/runs' / (rid + '.json')).read_text())
    assert r['status'] == 'complete' and r['completed_updates'] == a['completed_updates'] == 2000
    assert a['status'] == 'pass' and all(a['checks'].values())
    assert a['binding']['result_sha256'] == file_sha256(rp)
    assert not r['test_or_ispd_used'] and not a['test_ispd_model_metrics_read']
    return r


def verified(task):
    r = check_input(task)
    p = EXP / 'logs/stage5_best_validation' / (task['run_id'] + '.json')
    if not p.exists():
        return None
    v = json.loads(p.read_text())
    assert v['status'] == 'pass' and v['exact_metric_equality'] and v['validation_nets'] == 8000
    assert v['result_sha256'] == file_sha256(EXP / 'logs/stage5' / task['run_id'] / 'result.json')
    assert v['checkpoint_sha256'] == r['best_checkpoint_sha256']
    assert v['validation_manifest_sha256'] == r['best']['validation_manifest_sha256']
    assert v['selected_update'] == r['best']['update'] and not v['test_ispd_used']
    return {'run_id': task['run_id'], 'logical_path': str(p), 'resolved_path': str(p.resolve()),
            'sha256': file_sha256(p), 'validation_nets': 8000}


def preflight(purpose):
    cmd = [sys.executable, str(ROOT / 'scripts/capture_host_runtime.py'), '--purpose', purpose,
           '--measured-peak-mib', '2654', '--safety-margin-mib', '2048']
    p = subprocess.run(cmd, capture_output=True, text=True)
    append_jsonl(JOB / 'preflights.jsonl', {'at_utc': utc_now(), 'command': cmd,
        'exit_code': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr})
    if p.returncode:
        raise RuntimeError('Fresh capacity preflight failed: ' + p.stdout + p.stderr)
    return json.loads(p.stdout.strip().splitlines()[-1])


def stop_stage(reason):
    pipeline = EXP / 'logs/stage5_pipeline'
    main = json.loads((pipeline / 'job.json').read_text())
    queue = json.loads((pipeline / 'queue.json').read_text())
    targets = [(main['pid'], 'stage5_pipeline.py')]
    targets += [(x['pid'], 'train_stage5.py') for x in queue.get('active', [])]
    stopped = []
    for pid, script in targets:
        p = Path(f'/proc/{pid}/cmdline')
        if p.exists() and str(ROOT / 'scripts' / script).encode() in p.read_bytes():
            try:
                os.kill(pid, signal.SIGTERM)
                stopped.append(pid)
            except ProcessLookupError:
                pass
    atomic_write_json(JOB / 'failure.json', {'at_utc': utc_now(), 'reason': reason, 'stopped_project_pids': stopped})


def publish(records, total):
    atomic_write_json(JOB / 'result.json', {'at_utc': utc_now(), 'status': 'pass' if len(records) == total else 'partial_pass',
        'runs': records, 'total': total, 'validation_nets': len(records) * 8000,
        'stage5_complete': False, 'stage6_started': False, 'final_export_requires_all_nine_runs': True})
    p = CODEX / '01_WDRO-REST_项目进度.md'
    marker = '- 阶段5完成项validation重算：'
    line = (f'{marker}{len(records)}/{total} 个已完整审计的best独立重算通过，共{len(records)*8000} validation nets（{utc_now()}）；'
            '证据 `experiments/logs/stage5_validation_completed/result.json`。这是各run验收的提前执行；5.4最终导出仍要求9/9完整训练/审计和全部best重算通过。')
    value = p.read_text()
    if marker in value:
        value = '\n'.join(line if old.startswith(marker) else old for old in value.splitlines()) + '\n'
    else:
        token = '### 阶段 5 最小 validation 搜索与消融（本次授权）\n'
        assert token in value
        value = value.replace(token, token + '\n' + line + '\n', 1)
    partial = p.with_name(p.name + '.validation_recheck.partial')
    partial.write_text(value)
    partial.replace(p)


def launch(tasks):
    # The first full replay is the foreground smoke for this exact evaluator.
    assert verified(tasks[0]), 'Foreground replay must pass first'
    lock = JOB / 'run.lock'
    assert not lock.exists() or not process_is_alive(json.loads(lock.read_text())['pid']), 'Validation controller already running'
    tmux = ['tmux', '-S', str(SOCKET)]
    assert subprocess.run([*tmux, 'has-session', '-t', SESSION], capture_output=True).returncode != 0, 'Session already exists'
    runtime = preflight('stage5_completed_validation_tmux_start_or_resume')
    args = ['env', f'WDRO_TMUX_SESSION={SESSION}', f'WDRO_HOST_RUNTIME_MANIFEST={runtime["manifest"]}',
            'bash', str(ROOT / 'scripts/stage3_env.sh'), sys.executable, str(SCRIPT)]
    log = JOB / 'controller.log'
    cmd = [*tmux, 'new-session', '-d', '-s', SESSION, '-c', str(CODEX),
           f'exec {shlex.join(args)} >> {shlex.quote(str(log))} 2>&1']
    record = {'at_utc': utc_now(), 'session': SESSION, 'socket': str(SOCKET), 'socket_resolved': str(SOCKET.resolve()),
        'exact_command': shlex.join(args), 'tmux_command': cmd, 'preflight': runtime,
        'log': str(log), 'log_resolved': str(log.resolve()), 'monitor_command': f'cat {JOB}/heartbeat.json',
        'recovery_command': shlex.join(['bash', str(ROOT / 'scripts/stage3_env.sh'), sys.executable, str(SCRIPT), '--launch'])}
    atomic_write_json(JOB / 'launch.json', record)
    subprocess.run(cmd, check=True)
    for _ in range(30):
        p = JOB / 'job.json'
        if p.exists():
            j = json.loads(p.read_text())
            if j.get('tmux_session') == SESSION and process_is_alive(j['pid']):
                record['pid'] = j['pid']
                break
        time.sleep(.5)
    assert 'pid' in record, str(log)
    record['postlaunch'] = observe()
    atomic_write_json(JOB / 'launch.json', record)
    print(json.dumps({'status': 'running', 'pid': record['pid'], 'session': SESSION}), flush=True)


def run(tasks, maximum):
    command = shlex.join(['bash', str(ROOT / 'scripts/stage3_env.sh'), sys.executable, str(SCRIPT), *sys.argv[1:]])
    with JobLease(JOB, run_id='stage5_completed_validation', exact_command=command,
        resume_command=shlex.join(['bash', str(ROOT / 'scripts/stage3_env.sh'), sys.executable, str(SCRIPT), '--launch']),
        monitor_command=f'cat {JOB}/heartbeat.json', expected_updates=len(tasks), physical_gpu_index=None,
        host_runtime_manifest=os.environ.get('WDRO_HOST_RUNTIME_MANIFEST')) as lease:
        records = [v for t in tasks if (v := verified(t))]
        newly = 0
        for t in tasks:
            if verified(t):
                continue
            rid = t['run_id']
            # A surviving validation child is adopted after a controller disconnect.
            active_path = JOB / 'active.json'
            old = json.loads(active_path.read_text()) if active_path.exists() else None
            proc = None
            if old and old['run_id'] == rid:
                cp = Path(f'/proc/{old["pid"]}/cmdline')
                if process_is_alive(old['pid']) and cp.exists() and rid.encode() in cp.read_bytes():
                    while process_is_alive(old['pid']):
                        if verified(t):
                            break
                        gate()
                        lease.update(update=len(records), status='adopting', detail=old)
                        time.sleep(5)
                    if verified(t):
                        records.append(verified(t))
                        publish(records, len(tasks))
                        continue
            for attempt in range(1, 4):
                gate()
                runtime = preflight('stage5_completed_validation_' + rid)
                host = json.loads(Path(runtime['manifest']).read_text())
                gpu = max((x for x in host['gpu_decisions'] if x['eligible']), key=lambda x: x['minimum_free_memory_mib'])['index']
                cmd = ['bash', str(ROOT / 'scripts/stage3_env.sh'), sys.executable,
                    str(ROOT / 'scripts/verify_stage5_best_validation.py'), '--run-id', rid, '--gpu', str(gpu), '--preflight', runtime['manifest']]
                log = JOB / (rid + f'_{int(time.time())}.log')
                with log.open('a') as out:
                    proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
                active = {'at_utc': utc_now(), 'run_id': rid, 'pid': proc.pid, 'gpu': gpu, 'command': cmd,
                    'exact_command': shlex.join(cmd), 'log': str(log), 'log_resolved': str(log.resolve()), 'preflight': runtime, 'attempt': attempt}
                atomic_write_json(active_path, active)
                append_jsonl(JOB / 'launches.jsonl', active)
                constrained = False
                saturated_samples = 0
                try:
                    while proc.poll() is None:
                        snap = observe()
                        append_jsonl(JOB / 'resources.jsonl', dict(snap, run_id=rid, pid=proc.pid))
                        lease.update(update=len(records), status='running', detail=active)
                        row = next(x for x in snap['gpus'] if x['index'] == gpu)
                        saturated_samples = saturated_samples + 1 if row['utilization_gpu_percent'] >= 95 else 0
                        if row['memory_free_mib'] < 2048 or snap['disk'][1]['free_gib'] < 100 or saturated_samples >= 6:
                            constrained = True
                            proc.terminate()
                            proc.wait(timeout=30)
                            break
                        time.sleep(5)
                    if proc.returncode == 0:
                        assert verified(t)
                        break
                    tail = log.read_text()[-20000:]
                    if not constrained and 'out of memory' not in tail.lower():
                        stop_stage(f'{rid}: validation replay failed; inspect {log}')
                        raise RuntimeError(str(log))
                    append_jsonl(JOB / 'recoveries.jsonl', dict(active, exit_code=proc.returncode, reason='capacity/OOM; stateless validation replay'))
                except BaseException:
                    if proc.poll() is None:
                        proc.terminate()
                        proc.wait(timeout=30)
                    raise
            else:
                raise RuntimeError('Three capacity retries exhausted; resume later')
            records.append(verified(t))
            newly += 1
            publish(records, len(tasks))
            print(json.dumps({'run_id': rid, 'status': 'pass', 'completed': len(records)}), flush=True)
            if maximum and newly >= maximum:
                break
        publish(records, len(tasks))
        lease.finish(update=len(records), status='complete' if len(records) == len(tasks) else 'paused_after_smoke')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--max-new-runs', type=int)
    args = parser.parse_args()
    gate()
    assert not (JOB / 'failure.json').exists(), 'Resolve validation failure first'
    contract = json.loads((JOB / 'operations_v1.json').read_text())
    for p, digest in contract['files'].items():
        assert file_sha256(Path(p)) == digest, p
    tasks = json.loads((JOB / 'tasks_v1.json').read_text())['tasks']
    for task in tasks:
        check_input(task)
    if args.launch:
        launch(tasks)
    else:
        run(tasks, args.max_new_runs)


if __name__ == '__main__':
    main()
