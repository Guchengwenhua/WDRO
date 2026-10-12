"""Artifact, resource and job utilities for Stage 7 only."""
from __future__ import annotations

import contextlib
import fcntl
import json
import os
import shlex
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.io_utils import atomic_write_json, append_jsonl
from src.job_control import utc_now
from src.rest import file_sha256
from stage6.evaluation import atomic_npz, artifact

ROOT = Path(__file__).resolve().parents[1]
CODEX = ROOT.parent
TOP = Path('/remote-home/Guilin/DRO/Codex')
EXP = CODEX / 'experiments'
LOG = EXP / 'logs/stage7'
CACHE = EXP / 'cache/stage7'
PLAN = ROOT / 'manifests/stage7_plan_v1.json'
SCRIPT = ROOT / 'scripts/stage7.py'
ENV = ROOT / 'scripts/stage3_env.sh'


def read(path):
    return json.loads(Path(path).read_text())


def checked(row):
    path = Path(row['logical_path'])
    assert path.is_absolute() and path.is_relative_to(TOP), path
    assert str(path.resolve()) == row['resolved_path'], path
    assert file_sha256(path) == row['sha256'], path
    return path


def load_npz(path):
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def immutable_json(path, value):
    if Path(path).exists():
        assert read(path) == value, f'frozen output changed: {path}'
    else:
        atomic_write_json(path, value, mode=0o444)


def stamp():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + f'_{os.getpid()}_{time.time_ns()}'


def command(*args):
    return ['bash', str(ENV), sys.executable, str(SCRIPT), *map(str, args)]


def pid_identity(pid):
    try:
        tail = Path(f'/proc/{pid}/stat').read_text().rsplit(') ', 1)[1].split()
        return dict(pid=pid, state=tail[0], start_ticks=tail[19],
                    cwd=str(Path(f'/proc/{pid}/cwd').resolve()))
    except FileNotFoundError:
        return None


class Lease:
    """flock survives process crashes without treating stale PID files as live."""
    def __init__(self, directory, name, preflight=None):
        self.directory = Path(directory)
        self.name = name
        self.preflight = preflight
        self.count = 0

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = (self.directory / 'run.lock').open('a+')
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock.close()
            raise RuntimeError(f'duplicate job rejected: {self.name}')
        self.job = dict(name=self.name, status='running', started_at_utc=utc_now(),
            process=pid_identity(os.getpid()), session=os.environ.get('WDRO_TMUX_SESSION'),
            exact_command=shlex.join(['bash', str(ENV), sys.executable, *sys.argv]),
            recovery_command=shlex.join(command('launch')),
            monitor_command=f'cat {LOG}/pipeline/heartbeat.json', preflight=self.preflight,
            directory=dict(logical_path=str(self.directory), resolved_path=str(self.directory.resolve())),
            runtime_environment={k: os.environ.get(k) for k in ['TMPDIR', 'TMP', 'TEMP',
                'XDG_CACHE_HOME', 'TORCH_HOME', 'CUDA_CACHE_PATH', 'MPLCONFIGDIR',
                'TRITON_CACHE_DIR', 'PYTHONDONTWRITEBYTECODE', 'CUDA_VISIBLE_DEVICES']})
        atomic_write_json(self.directory / 'job.json', self.job)
        atomic_write_json(self.directory / 'pid.json', self.job['process'])
        append_jsonl(self.directory / 'attempts.jsonl', self.job)
        self.update(0, 'running')
        return self

    def update(self, count, status='running', **detail):
        self.count = count
        atomic_write_json(self.directory / 'heartbeat.json', dict(name=self.name,
            pid=os.getpid(), completed_items=count, status=status,
            updated_at_utc=utc_now(), detail=detail))

    def __exit__(self, kind, error, tb):
        self.job.update(status='failed' if error else 'exited', finished_at_utc=utc_now(),
                        completed_items=self.count, error=str(error) if error else None,
                        lock_released=True)
        atomic_write_json(self.directory / 'job.json', self.job)
        append_jsonl(self.directory / 'attempts.jsonl', self.job)
        if error:
            self.update(self.count, 'failed', error=str(error))
        fcntl.flock(self.lock, fcntl.LOCK_UN)
        self.lock.close()


def capture(purpose, peak=3602, samples=5):
    output = LOG / 'resources' / f'host_runtime_{stamp()}.json'
    cmd = [sys.executable, str(ROOT / 'scripts/capture_host_runtime.py'),
           '--purpose', purpose, '--measured-peak-mib', str(int(np.ceil(peak)) + 1024),
           '--safety-margin-mib', '2048', '--samples', str(samples),
           '--sample-interval-seconds', '1', '--output', str(output)]
    run = subprocess.run(cmd, text=True, capture_output=True)
    append_jsonl(LOG / 'resources/commands.jsonl', dict(at_utc=utc_now(), command=cmd,
        exit_code=run.returncode, stdout=run.stdout, stderr=run.stderr))
    assert run.returncode in (0, 2) and output.exists(), run.stderr
    data = read(output)
    assert data['disk']['experiments_artifact_root']['free_gib'] >= 100
    assert data['disk']['codex_logical_root']['free_gib'] > 1
    return output, data


def progress(substep, status, evidence, next_action):
    path = TOP / '01_WDRO-REST_项目进度.md'
    LOG.mkdir(parents=True, exist_ok=True)
    with (LOG / 'progress.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        content = path.read_text()
        backup = LOG / 'progress_before_stage7.md'
        if not backup.exists():
            backup.write_text(content)
        now = utc_now()
        updates = {'更新时间': now, '总体阶段': '阶段7 `DOING`（ISPD协议A外部测试）',
                   '实验状态': f'{substep}：{status}', '批量实验阻塞': '以阶段7子步骤门禁与心跳为准',
                   '下一动作': next_action}
        lines = content.splitlines()
        for i, line in enumerate(lines):
            for key, value in updates.items():
                if line.startswith(f'| {key} |'):
                    lines[i] = f'| {key} | {value} |'
            if line.startswith('| 7. ISPD 外部测试 |'):
                lines[i] = f'| 7. ISPD 外部测试 | `DOING` | {substep}；证据 {evidence} |'
        content = '\n'.join(lines) + '\n'
        entry = f'### {now} 阶段7\n\n- `{status}` {substep}。证据 `{evidence}`。下一步：{next_action}。阶段8–9未启动。\n\n'
        content = content.replace('## 8. 变更记录\n\n', '## 8. 变更记录\n\n' + entry, 1)
        tmp = path.with_suffix('.md.partial')
        tmp.write_text(content)
        os.replace(tmp, path)


def gate():
    p = read(PLAN)
    assert p['stage'] == 7 and p['status'] == 'frozen'
    for row in p['dependencies']:
        checked(row)
    if (ROOT / 'manifests/stage7_operations_v1.json').exists():
        for row in read(ROOT / 'manifests/stage7_operations_v1.json')['files']:
            checked(row)
    return p
