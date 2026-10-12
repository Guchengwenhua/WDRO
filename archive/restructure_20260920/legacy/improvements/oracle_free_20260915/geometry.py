"""Label-free geometry and REST length evaluation. No solver binary is loaded."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
PARENT = PROJECT / 'revisions/adversarial_v3_1_20260912'
CODE = PARENT / 'Codex/wdro_rest_min'
LEGACY = PARENT / 'DRO/WDRO-REST'
sys.path.insert(0, str(CODE))
from src.datasets import generate_terminals, normalize_fixed_degree_batch
from src.geometry import has_spacing_violation

PARAMETERS = {
    'uniform': {},
    'mix_gauss': dict(nonuniform=.8, clusters=6, sigma=.04),
    'edge_bias': dict(nonuniform=.8, clusters=4, sigma=.06),
    'grid_cluster': dict(nonuniform=.8, clusters=8, sigma=.03),
}
_GUARD = False

def install_oracle_guard():
    """Fail closed for accidental known solver/label/legacy weight dependencies."""
    global _GUARD
    if _GUARD:
        return
    def guard(event, args):
        if event == 'import' and 'geosteiner' in str(args[0]).lower():
            raise RuntimeError('Oracle-free guard: forbidden solver import')
        if event in ('ctypes.dlopen', 'subprocess.Popen', 'os.system', 'os.exec', 'os.spawn'):
            if any(x in str(args).lower() for x in ('geosteiner', 'libeval', 'gst_rsmt')):
                raise RuntimeError('Oracle-free guard: forbidden solver/library call')
        if event == 'open' and isinstance(args[0], (str, bytes)):
            p = str(args[0]).lower()
            if any(x in p for x in ('l_star', 'lstar', 'reference.sqlite')):
                raise RuntimeError('Oracle-free guard: forbidden oracle label file')
            if p.endswith(('.pt', '.pth', '.ckpt')):
                if not Path(str(args[0])).resolve().is_relative_to(ROOT):
                    raise RuntimeError('Oracle-free guard: pretrained external weights forbidden')
    sys.addaudithook(guard)
    _GUARD = True

def build_actor(degree, device):
    sys.path.insert(0, str(LEGACY))
    from models.actor_critic import Actor
    return Actor(degree, device)

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def dependencies():
    paths = list((LEGACY/'models').glob('*.py'))
    paths += [CODE/'src/datasets.py', CODE/'src/geometry.py']
    return {str(p): sha256(p) for p in sorted(paths)}

def derived_seed(*parts):
    return int.from_bytes(hashlib.sha256(('oracle-free-v1:' + ':'.join(map(str, parts))).encode()).digest()[:4], 'little')

def _default(v):
    if isinstance(v, np.ndarray): return v.tolist()
    if isinstance(v, np.generic): return v.item()
    if isinstance(v, Path): return str(v)
    raise TypeError(type(v).__name__)

def atomic_json(path, obj):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name+f'.partial.{os.getpid()}.{time.time_ns()}')
    with tmp.open('w') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_default, allow_nan=False)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, p)

def atomic_torch(path, obj):
    import torch
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name+'.partial')
    with tmp.open('wb') as f:
        torch.save(obj, f); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, p)

def atomic_npz(path, **arrays):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name+'.partial')
    with tmp.open('wb') as f:
        np.savez_compressed(f, **arrays); f.flush(); os.fsync(f.fileno())
    with np.load(tmp, allow_pickle=False) as z:
        for k,v in arrays.items():
            if not np.array_equal(z[k], v): raise RuntimeError('NPZ validation failed')
    os.replace(tmp, p)

def make_data(count, degree, distribution, namespace, seed=0):
    accepted=[]; n=0
    for attempt in range(100):
        raw=generate_terminals(count, degree, derived_seed(namespace,seed,distribution,degree,attempt), distribution, PARAMETERS[distribution])
        x,_,_,valid,_=normalize_fixed_degree_batch(raw)
        x=x[valid].astype(np.float32)
        x=np.asarray([p for p in x if not has_spacing_violation(p)], dtype=np.float32)
        if len(x): accepted.append(x); n+=len(x)
        if n >= count: return np.concatenate(accepted)[:count]
    raise RuntimeError('Data generation exhausted')

def legalize(centers, deltas):
    x=np.asarray(centers, dtype=np.float32); d=np.asarray(deltas)
    raw=np.clip((x[:,None] if d.ndim==4 else x)+d,.1,.9)
    shape=raw.shape; flat=raw.reshape(-1,x.shape[1],2)
    z,_,_,valid,_=normalize_fixed_degree_batch(flat)
    z=z.astype(np.float32)
    for i,p in enumerate(z):
        if not valid[i] or has_spacing_violation(p):
            z[i]=x[i//shape[1] if d.ndim==4 else i]
    return z.reshape(shape)

def costs(centers, candidates):
    x=np.asarray(centers,dtype=np.float64); z=np.asarray(candidates,dtype=np.float64)
    paired=z.ndim==3
    if paired: z=z[:,None]
    out=np.empty(z.shape[:2],dtype=np.float64)
    for i in range(len(x)):
        for j in range(z.shape[1]):
            d=np.abs(x[i,:,None]-z[i,j,None]).sum(-1)
            rows,cols=linear_sum_assignment(d)
            out[i,j]=d[rows,cols].mean()
    return out[:,0] if paired else out

def bounds(pins):
    """B=max(HPWL,M/2)<=L*<=M; batched complete L1 metric Prim in O(B*n^2)."""
    x=np.asarray(pins,dtype=np.float64)
    if x.ndim!=3 or x.shape[-1]!=2 or x.shape[1]<2 or not np.isfinite(x).all():
        raise ValueError('invalid net batch')
    b,n,_=x.shape
    hpwl=np.ptp(x,axis=1).sum(-1)
    if np.any(hpwl<=0): raise ValueError('degenerate net')
    visited=np.zeros((b,n),bool); visited[:,0]=True
    best=np.abs(x-x[:,:1]).sum(-1); best[:,0]=np.inf
    mst=np.zeros(b); rows=np.arange(b)
    for _ in range(n-1):
        j=np.argmin(best,axis=1)
        mst+=best[rows,j]; visited[rows,j]=True
        distance=np.abs(x-x[rows,j,None]).sum(-1)
        best=np.where(visited,np.inf,np.minimum(best,distance))
    return np.maximum(hpwl,mst/2), mst, hpwl

def lengths(pins, topology):
    """Same terminal-segment span cost as REST eval.c, with topology connectivity checks.

    Aligned pins may make distinct terminal spans overlap: this matches historical
    evaluator accounting, potentially overcounting physical union, never undercounting.
    """
    x=np.asarray(pins,dtype=np.float64); t=np.asarray(topology)
    if x.ndim!=3 or x.shape[-1]!=2 or not np.isfinite(x).all(): raise ValueError('invalid pins')
    b,n,_=x.shape
    if t.shape!=(b,2*(n-1)) or not np.issubdtype(t.dtype,np.integer): raise ValueError('invalid topology')
    if np.any(t<0) or np.any(t>=n): raise ValueError('topology index out of range')
    # A connected n-1-edge terminal graph guarantees connected L-shaped routes.
    for edges in t.reshape(b,n-1,2):
        parent=list(range(n))
        def find(a):
            while parent[a]!=a:
                parent[a]=parent[parent[a]]; a=parent[a]
            return a
        for u,v in edges:
            ru,rv=find(int(u)),find(int(v))
            if ru==rv: raise ValueError('REST topology contains a cycle/self edge')
            parent[ru]=rv
    xl=x[:,:,0].copy(); xr=xl.copy(); yl=x[:,:,1].copy(); yh=yl.copy()
    rows=np.repeat(np.arange(b),n-1)
    u=t[:,::2].reshape(-1); v=t[:,1::2].reshape(-1)
    np.minimum.at(yl,(rows,u),x[rows,v,1]); np.maximum.at(yh,(rows,u),x[rows,v,1])
    np.minimum.at(xl,(rows,v),x[rows,u,0]); np.maximum.at(xr,(rows,v),x[rows,u,0])
    value=(yh-yl+xr-xl).sum(-1)
    if np.any(value<=0) or not np.isfinite(value).all(): raise RuntimeError('invalid tree lengths')
    return value
