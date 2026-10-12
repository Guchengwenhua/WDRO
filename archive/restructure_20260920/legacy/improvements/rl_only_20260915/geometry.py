"""RL environment: coordinates, action feasibility and own-action wirelength only.
No reference construction, optimizer, MST, bound or assignment solver is used.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
LEGACY=PROJECT/'revisions/adversarial_v3_1_20260912/DRO/WDRO-REST'
PARAMETERS={
 'uniform':{},
 'mix_gauss':dict(nonuniform=.8,clusters=6,sigma=.04),
 'edge_bias':dict(nonuniform=.8,clusters=4,sigma=.06),
 'grid_cluster':dict(nonuniform=.8,clusters=8,sigma=.03),
}
_GUARD=False

def install_oracle_guard():
    global _GUARD
    if _GUARD: return
    def guard(event,args):
        if event=='import':
            name=str(args[0]).lower()
            forbidden=('geosteiner','src.geosteiner','scipy.optimize','scipy.sparse.csgraph','networkx','ortools')
            if any(name==key or name.startswith(key+'.') for key in forbidden):
                raise RuntimeError('RL-only guard: external solver import forbidden')
        if event in ('ctypes.dlopen','subprocess.Popen','os.system','os.exec','os.spawn'):
            if any(key in str(args).lower() for key in ('geosteiner','libeval','gst_rsmt','gurobi','cplex','flute')):
                raise RuntimeError('RL-only guard: external reference solver forbidden')
        if event=='open' and isinstance(args[0],(str,bytes)):
            p=str(args[0]).lower()
            if any(key in p for key in ('l_star','lstar','reference.sqlite','mst_cache','optimal_labels')):
                raise RuntimeError('RL-only guard: external reference labels forbidden')
            if p.endswith(('.pt','.pth','.ckpt')) and not Path(str(args[0])).resolve().is_relative_to(ROOT):
                raise RuntimeError('RL-only guard: external pretrained weights forbidden')
    sys.addaudithook(guard); _GUARD=True

def dependencies():
    return {str(p):sha256(p) for p in sorted((LEGACY/'models').glob('*.py'))}

def derived_seed(*parts):
    return int.from_bytes(hashlib.sha256(('rl-only-v1:'+':'.join(map(str,parts))).encode()).digest()[:4],'little')

def has_spacing_violation(pins):
    p=np.asarray(pins,dtype=np.float64)
    if not np.isfinite(p).all(): return True
    d=np.abs(p[:,None]-p[None]).sum(-1)
    np.fill_diagonal(d,np.inf)
    return bool(np.any(d<1e-6))

def make_data(count,degree,distribution,namespace,seed=0):
    """New fixed-window coordinate data; no per-net span/length normalization."""
    accepted=[]; total=0
    param=PARAMETERS[distribution]
    for attempt in range(100):
        rng=np.random.default_rng(derived_seed(namespace,seed,distribution,degree,attempt))
        v=rng.random((count,degree,2))
        if distribution=='mix_gauss':
            centers=rng.random((param['clusters'],2))
            mask=rng.random((count,degree))<param['nonuniform']
            ids=rng.integers(param['clusters'],size=(count,degree))
            gaussian=centers[ids]+rng.normal(0,param['sigma'],size=v.shape)
            v[mask]=gaussian[mask]
        elif distribution=='edge_bias':
            v **= 1+param['nonuniform']*3
            flip=rng.random(v.shape)>.5
            v[flip]=1-v[flip]
        elif distribution=='grid_cluster':
            aligned=rng.random((count,degree))<param['nonuniform']
            rows=np.linspace(.1,.9,param['clusters'])
            ids=rng.integers(param['clusters'],size=(count,degree))
            ax=rng.random((count,degree)); ay=rows[ids]+rng.normal(0,param['sigma']*.3,size=(count,degree))
            v[:,:,0][aligned]=ax[aligned]; v[:,:,1][aligned]=ay[aligned]
        elif distribution!='uniform': raise ValueError(distribution)
        v=(.1+.8*np.clip(v,0,1)).astype(np.float32)
        good=np.asarray([p for p in v if not has_spacing_violation(p)],dtype=np.float32)
        if len(good): accepted.append(good); total+=len(good)
        if total>=count: return np.concatenate(accepted)[:count]
    raise RuntimeError('Cannot generate sufficient legal coordinates')

def legalize(centers,deltas):
    x=np.asarray(centers,dtype=np.float32); d=np.asarray(deltas)
    raw=np.clip((x[:,None] if d.ndim==4 else x)+d,.1,.9).astype(np.float32)
    shape=raw.shape; flat=raw.reshape(-1,x.shape[1],2)
    for i,p in enumerate(flat):
        if has_spacing_violation(p): flat[i]=x[i//shape[1] if d.ndim==4 else i]
    return flat.reshape(shape)

def costs(centers,candidates):
    """Explicit labeled-coordinate transport; no assignment optimization."""
    x=np.asarray(centers,dtype=np.float64); z=np.asarray(candidates,dtype=np.float64)
    if z.ndim==4: x=x[:,None]
    return np.abs(x-z).sum(-1).mean(-1)


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

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

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

def build_actor(degree, device):
    sys.path.insert(0, str(LEGACY))
    from models.actor_critic import Actor
    return Actor(degree, device)
