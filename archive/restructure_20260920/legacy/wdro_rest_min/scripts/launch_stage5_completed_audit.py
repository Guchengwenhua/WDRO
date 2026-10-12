#!/usr/bin/env python3
"""Persist the bounded completed-run audit after a foreground cursor smoke."""
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.io_utils import atomic_write_json
from src.job_control import process_is_alive, utc_now
from audit_stage5_completed import JOB, EXP, CODEX, preflight, gate, observe


def main():
    gate()
    assert json.loads((JOB / 'foreground_smoke.json').read_text())['status'] == 'pass'
    lock = JOB / 'run.lock'
    if lock.exists():
        assert not process_is_alive(json.loads(lock.read_text())['pid']), 'Completed audit is already running'
    socket = EXP / 'tmp/stage5.sock'
    session = 'wdro_stage5_completed_audit_v1'
    tmux = ['tmux', '-S', str(socket)]
    assert subprocess.run([*tmux, 'has-session', '-t', session], capture_output=True).returncode != 0, 'Session already exists'
    runtime = preflight('stage5_completed_audit_persistent_start_or_resume')
    cmd = ['env', f'WDRO_TMUX_SESSION={session}', f'WDRO_HOST_RUNTIME_MANIFEST={runtime["manifest"]}',
           'CUDA_VISIBLE_DEVICES=', 'bash', str(ROOT / 'scripts/stage3_env.sh'), sys.executable,
           str(ROOT / 'scripts/audit_stage5_completed.py')]
    log = JOB / 'controller.log'
    launch = [*tmux, 'new-session', '-d', '-s', session, '-c', str(CODEX),
              f'exec {shlex.join(cmd)} >> {shlex.quote(str(log))} 2>&1']
    record = {'at_utc': utc_now(), 'session': session, 'socket': str(socket), 'socket_resolved': str(socket.resolve()),
        'exact_command': shlex.join(cmd), 'tmux_command': launch, 'log': str(log), 'log_resolved': str(log.resolve()),
        'preflight': runtime, 'monitor_command': f'cat {JOB}/heartbeat.json',
        'recovery_command': shlex.join(['bash', str(ROOT / 'scripts/stage3_env.sh'), sys.executable, str(Path(__file__))]),
        'research_protocol_override': False}
    atomic_write_json(JOB / 'launch.json', record)
    subprocess.run(launch, check=True)
    for _ in range(30):
        path = JOB / 'job.json'
        if path.exists():
            state = json.loads(path.read_text())
            if state.get('tmux_session') == session and process_is_alive(state['pid']):
                record['pid'] = state['pid']
                break
        time.sleep(.5)
    assert 'pid' in record, f'No live audit controller: {log}'
    record['postlaunch'] = observe()
    atomic_write_json(JOB / 'launch.json', record)
    print(json.dumps({'status': 'running', 'pid': record['pid'], 'session': session, 'launch': str(JOB / 'launch.json')}))


if __name__ == '__main__':
    main()
