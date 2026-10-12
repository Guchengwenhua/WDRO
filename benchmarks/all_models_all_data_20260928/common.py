"""Utilities for the 2026-09-28 cross-dataset evaluation (no training)."""
import json
import os
from pathlib import Path
import time

PROJECT = Path('/remote-home/Guilin/DRO/Codex')
ROOT = PROJECT / 'benchmarks/all_models_all_data_20260928'
ART = ROOT / 'artifacts'
PYTHON = '/remote-home/Guilin/miniconda3/envs/py311/bin/python'
DISTS = ('uniform', 'mix_gauss', 'edge_bias', 'grid_cluster')

def read(path):
    return json.loads(Path(path).read_text())

def atomic(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.partial')
    with tmp.open('w') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def npz(path, **arrays):
    import numpy as np
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.partial')
    with tmp.open('wb') as f:
        np.savez_compressed(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def identity():
    return dict(pid=os.getpid(), start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19], time=time.time())

def event(path, value):
    with Path(path).open('a') as f:
        f.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')

def extract_state(cp, model):
    state = cp
    for key in model['state_key'].split('/'):
        state = state[key]
    if model.get('strip_prefix'):
        prefix = model['strip_prefix']
        state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    return state
