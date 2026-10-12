"""Frozen experiment helpers; no solver or external weight loading."""
import hashlib
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
import geometry as G

ROOT = Path(__file__).resolve().parent
GROUPS = tuple(G.PARAMETERS)

def now():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())

def read(path):
    return json.loads(Path(path).read_text())

def event(path, obj):
    with Path(path).open('a') as f:
        f.write(json.dumps(obj, sort_keys=True, allow_nan=False)+'\n')
        f.flush()
        os.fsync(f.fileno())

def identity():
    fields=Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')',1)[1].split()
    return dict(pid=os.getpid(),start_ticks=fields[19],boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())

def digest(model):
    h=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])

def set_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'].cpu())
    if torch.cuda.is_available() and state['cuda']:
        torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda']])

def training_batch(arm,seed,step,batch=64):
    if arm=='U':
        return G.make_data(batch,30,'uniform',f'train/{step}',seed),np.zeros(batch,dtype=np.int64)
    arrays=[G.make_data(batch//4,30,g,f'budget-v1/train/{step}/{g}',seed,independent_layouts=True) for g in GROUPS]
    return np.concatenate(arrays),np.repeat(np.arange(4),batch//4)

def binding(arm,seed):
    return dict(arm=arm,seed=seed,protocol_sha256=G.sha256(ROOT/'protocol.json'),
                code_sha256={str(ROOT/n):G.sha256(ROOT/n) for n in ('geometry.py','common.py','worker.py')},
                dependencies=G.dependencies(),data_manifest_sha256=G.sha256(ROOT/'artifacts/data/manifest.json'))

def check_binding(b):
    if b['protocol_sha256']!=G.sha256(ROOT/'protocol.json'): raise RuntimeError('protocol changed')
    if b['data_manifest_sha256']!=G.sha256(ROOT/'artifacts/data/manifest.json'): raise RuntimeError('data manifest changed')
    for field in ('code_sha256','dependencies'):
        for p,h in b[field].items():
            if G.sha256(p)!=h: raise RuntimeError('Source changed: '+p)

def configure(gpu,threads):
    if gpu is not None:
        os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES']=str(gpu) if gpu>=0 else ''
    torch.set_num_threads(threads); torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(.125)
        return torch.device('cuda:0')
    return torch.device('cpu')

def load_data(split):
    manifest=read(ROOT/'artifacts/data/manifest.json')
    p=ROOT/'artifacts/data'/f'{split}.npz'
    if G.sha256(p)!=manifest['artifacts'][p.name]: raise RuntimeError('Data hash mismatch')
    with np.load(p) as z: return {k:z[k] for k in z.files}

def summary(arrays):
    groups={g:dict(mean=float(arrays[g].mean()),p95=float(np.quantile(arrays[g],.95))) for g in GROUPS}
    return dict(groups=groups,mean=float(np.mean([v['mean'] for v in groups.values()])),
                worst=max(v['mean'] for v in groups.values()),p95=float(np.quantile(np.concatenate(list(arrays.values())),.95)))

def verify_complete(path):
    c=read(path)
    if c['status']!='complete': raise RuntimeError('Incomplete '+str(path))
    check_binding(c['binding'])
    for name,h in c['artifacts'].items():
        if G.sha256(Path(path).parent/name)!=h: raise RuntimeError('Artifact mismatch: '+name)
    return c

def repair_tail(path):
    if not path.exists(): return
    raw=path.read_bytes()
    if raw and not raw.endswith(b'\n'):
        end=raw.rfind(b'\n')+1
        path.with_name(path.name+f'.incomplete.{time.time_ns()}').write_bytes(raw[end:])
        with path.open('r+b') as f: f.truncate(end)
