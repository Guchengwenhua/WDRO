"""Audited host scheduling helpers; no training or selection semantics."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODEX = ROOT.parent
EXP = CODEX / 'experiments'
sys.path.insert(0, str(ROOT))
from src.io_utils import atomic_write_json, append_jsonl
from src.job_control import utc_now
from src.rest import file_sha256
from precision_capacity_wait import classify_capture, check_operations

OPERATIONS = ROOT / 'manifests/precision_queue_recovery_operations_v1.json'


def gate():
    check_operations()
    value = json.loads(OPERATIONS.read_text())
    assert value['research_protocol_override'] is False
    for row in value['files']:
        assert file_sha256(Path(row['logical_path'])) == row['sha256'], row['logical_path']


def process_is_alive(pid):
    if not pid or pid <= 0:
        return False
    try:
        return Path(f'/proc/{pid}/stat').read_text().rsplit(') ', 1)[1].split()[0] not in ('Z', 'X')
    except FileNotFoundError:
        return False


def process_matches(pid, run_id, script):
    if not process_is_alive(pid):
        return False
    try:
        args = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        cwd = Path(f'/proc/{pid}/cwd').resolve()
        paths = [Path(os.fsdecode(a)) for a in args if a.endswith(b'.py')]
        return run_id.encode() in args and any(
            (p if p.is_absolute() else cwd / p).resolve() == Path(script).resolve() for p in paths)
    except FileNotFoundError:
        return False


def cleanup_dead_lock(root):
    path = Path(root) / 'run.lock'
    if path.exists():
        old = json.loads(path.read_text())
        if process_is_alive(old['pid']):
            raise RuntimeError(f'live lock prevents duplicate launch: {path}')
        target = path.with_name('run.lock.stale.' + str(time.time_ns()))
        path.rename(target)
        append_jsonl(EXP / 'logs/precision_queue_recovery/stale_locks.jsonl',
                     dict(at_utc=utc_now(), old=old, logical_path=str(target), resolved_path=str(target.resolve())))


def capture(purpose, peak, job):
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = ROOT / 'manifests' / f'host_runtime_{stamp}_p{os.getpid()}_{time.time_ns()}.json'
    command = [sys.executable, str(ROOT / 'scripts/capture_host_runtime.py'),
               '--purpose', purpose, '--measured-peak-mib', str(peak + 1024),
               '--safety-margin-mib', '2048', '--samples', '10',
               '--sample-interval-seconds', '1', '--output', str(output)]
    done = subprocess.run(command, capture_output=True, text=True)
    append_jsonl(Path(job) / 'preflights.jsonl', dict(at_utc=utc_now(), command=command,
        exit_code=done.returncode, stdout=done.stdout, stderr=done.stderr))
    ready, summary = classify_capture(done)
    return summary if ready else None


def choose_gpu(rows, active, history, blocked, need, now):
    candidates = []
    for row in rows:
        gpu = row['index']
        count = sum(a['gpu'] == gpu for a in active.values())
        recent = [g for h in history if h['time'] >= now - 600 for g in h['gpus'] if g['index'] == gpu]
        floor = min([row['minimum_free_memory_mib']] + [g['memory_free_mib'] for g in recent])
        # Bursty external jobs must pass both the new 10-sample capture and
        # observations retained by this queue, never an old GPU allocation.
        stable = (not recent or (sum(g['utilization_gpu_percent'] for g in recent) / len(recent) < 85
                  and sum(g['utilization_gpu_percent'] >= 95 for g in recent) / len(recent) < .5))
        if row['eligible'] and stable and floor > need and count < 2 and now > blocked.get(gpu, 0):
            candidates.append((-floor, count, gpu))
    return min(candidates)[2] if candidates else None


def capacity_exit(active, code, result, log_text):
    """Only verified resource exits can requeue without a failure budget."""
    if code == 0 or (code is None and active.get('adopted')):
        if result and result.get('status') == 'complete' and result.get('completed_updates') == 2000:
            return False
        if active.get('stopping') and result and result.get('status') == 'paused_for_resume':
            assert result['completed_updates'] >= active.get('start_update', 0)
            assert file_sha256(Path(result['latest_checkpoint'])) == result['latest_checkpoint_sha256']
            assert file_sha256(Path(result['best_checkpoint'])) == result['best_checkpoint_sha256']
            return True
        raise RuntimeError('unexplained incomplete training exit; inspect the saved segment')
    if code is not None and code != 0 and ('torch.OutOfMemoryError: CUDA out of memory' in log_text
                                         or 'RuntimeError: CUDA out of memory' in log_text):
        return True
    raise RuntimeError(f'non-resource training exit {code}; inspect the saved segment')


def stop_children(active, script, job):
    pending = []
    for rid, a in active.items():
        if process_matches(a['pid'], rid, script):
            os.kill(a['pid'], signal.SIGTERM)
            pending.append((rid, a))
    # Wait for the trainer's atomic checkpoint instead of letting tmux exit
    # terminate a slow CEM update. Children also own separate process sessions.
    while pending:
        pending = [(rid, a) for rid, a in pending if
                   (a['proc'].poll() is None if 'proc' in a else process_matches(a['pid'], rid, script))]
        atomic_write_json(Path(job) / 'shutdown.json', dict(at_utc=utc_now(),
            status='waiting_for_atomic_checkpoints' if pending else 'children_stopped',
            children=[dict(run_id=rid, pid=a['pid']) for rid, a in pending]))
        if pending:
            time.sleep(5)


def task_priority(task):
    config = json.loads(Path(task['config']).read_text())
    return (0 if config['search_overrides']['searcher'] in ('random_64', 'cem_64') else 1,
            -task.get('degree', 30), task['run_id'])
