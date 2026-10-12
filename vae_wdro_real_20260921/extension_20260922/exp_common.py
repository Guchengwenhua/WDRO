"""Local experiment utilities; frozen implementations are imported read-only."""
import csv
import datetime as dt
import json
import os
from pathlib import Path
import random
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT / 'config.json').read_text())
ART = Path(CFG['artifact_storage'])
sys.path.insert(0, CFG['source_package'])

def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()

def json_default(x):
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, Path): return str(x)
    raise TypeError(type(x).__name__)

def atomic_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.partial')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=json_default, allow_nan=False) + '\n')
    os.replace(tmp, path)

def atomic_npz(path, **arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.partial')
    with tmp.open('wb') as f: np.savez_compressed(f, **arrays)
    os.replace(tmp, path)

def atomic_torch(path, obj):
    import torch
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.partial')
    torch.save(obj, tmp); os.replace(tmp, path)

def write_csv(path, rows, columns=None):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None: columns = list(dict.fromkeys(k for row in rows for k in row))
    tmp = path.with_name(path.name + f'.{os.getpid()}.partial')
    with tmp.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns); writer.writeheader(); writer.writerows(rows)
    os.replace(tmp, path)

def append_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        f.write(json.dumps(obj, ensure_ascii=False, default=json_default, allow_nan=False) + '\n'); f.flush()

def read_json(path): return json.loads(Path(path).read_text())

def canonical(x):
    x = np.asarray(x, dtype=np.float32)
    return np.ascontiguousarray(x[np.lexsort((x[:, 1], x[:, 0]))])

def check_pins(x):
    x = np.asarray(x)
    if x.shape[-2:] != (CFG['degree'], 2) or not np.isfinite(x).all():
        raise ValueError('invalid degree, shape or coordinates')
    if x.min() < .1 - 1e-7 or x.max() > .9 + 1e-7:
        raise ValueError('coordinates outside normalized domain')
    d = np.abs(x[..., :, None, :] - x[..., None, :, :]).sum(-1)
    d[..., np.arange(CFG['degree']), np.arange(CFG['degree'])] = np.inf
    if np.min(d) < 1e-6 - 1e-9: raise ValueError('minimum spacing violation')

def seed_all(seed):
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

def rng_state():
    import torch
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())

def set_rng(state):
    import torch
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'].cpu())
    torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])

def load_data():
    with np.load(ART / 'data.npz', allow_pickle=False) as f: return {k: f[k] for k in f.files}

def progress(name, **kw):
    atomic_json(ART / 'progress' / (name + '.json'), dict(time=utc(), pid=os.getpid(), **kw))

def mark_done(name, **kw):
    atomic_json(ART / 'done' / (name + '.json'), {'time': utc(), 'pid': os.getpid(), **kw})
