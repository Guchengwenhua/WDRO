"""Local, atomic artifact utilities for the authorized public-data extension."""
import contextlib
import fcntl
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
TOP = Path('/remote-home/Guilin/DRO/Codex')
REV = Path('/remote-home/Guilin/DRO/Codex/wdro_precision_v2')
PACKAGE = REV / 'wdro_rest_min'
PYTHON = '/remote-home/Guilin/miniconda3/envs/py311/bin/python'
ENV = PACKAGE / 'scripts/stage3_env.sh'
DEGREES = (10, 20, 30, 40, 50)

def now():
    return datetime.now(timezone.utc).isoformat()

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def artifact(path):
    p = Path(path).absolute()
    assert p.is_relative_to(TOP)
    return dict(logical_path=str(p), resolved_path=str(p.resolve()), sha256=sha(p), bytes=p.stat().st_size)

def read(path):
    return json.loads(Path(path).read_text())

def write(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_name(p.name + f'.{os.getpid()}.partial')
    with q.open('w') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(q, p)

def npz(path, arrays):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_name(p.name + f'.{os.getpid()}.partial')
    with q.open('wb') as f:
        np.savez_compressed(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    os.replace(q, p)

def load(path):
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}

@contextlib.contextmanager
def lease(name):
    p = BASE / 'logs' / name / 'run.lock'
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('a+') as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write(p.with_name('job.json'), dict(pid=os.getpid(), status='running', started=now()))
        try:
            yield
        except BaseException as e:
            write(p.with_name('job.json'), dict(pid=os.getpid(), status='failed', finished=now(), error=repr(e), lock_released=True))
            raise
        else:
            write(p.with_name('job.json'), dict(pid=os.getpid(), status='exited', finished=now(), lock_released=True))
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)

def gpu_snapshot():
    fields = 'index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu'
    result = subprocess.run(['nvidia-smi', f'--query-gpu={fields}', '--format=csv,noheader,nounits'], check=True, text=True, capture_output=True)
    rows = []
    for line in result.stdout.strip().splitlines():
        v = [x.strip() for x in line.split(',')]
        rows.append(dict(index=int(v[0]), uuid=v[1], name=v[2], total_mib=int(v[3]), used_mib=int(v[4]), free_mib=int(v[5]), utilization=int(v[6])))
    return rows

def preflight(purpose):
    snapshots = []
    for _ in range(3):
        snapshots.append(dict(at=now(), gpus=gpu_snapshot()))
        time.sleep(1)
    disk = os.statvfs(BASE)
    value = dict(purpose=purpose, at=now(), gpus=snapshots, free_disk_bytes=disk.f_bavail*disk.f_frsize,
                 logical_path=str(BASE), resolved_path=str(BASE.resolve()),
                 gpu_processes=subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_memory', '--format=csv'], capture_output=True, text=True).stdout)
    p = BASE / 'logs/resources' / f'{time.time_ns()}.json'
    write(p, value)
    assert value['free_disk_bytes'] > 50*1024**3, 'insufficient disk reserve'
    return p, value
