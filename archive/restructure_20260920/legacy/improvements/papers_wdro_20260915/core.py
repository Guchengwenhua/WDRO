"""Shared, isolated experiment contracts; no learned adversarial search."""
from __future__ import annotations
import hashlib, json, os, random, sys, time
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent
MAIN = ROOT.parents[1] / 'revisions/reference_precision_v2/Codex/wdro_rest_min'
if str(MAIN) not in sys.path:
    sys.path.insert(0, str(MAIN))
from src.datasets import build_synthetic_dataset, array_bundle_sha256
from src.candidates import generate_support, shortlist_support
from src.training_data import TrainingBatch

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for x in iter(lambda: f.read(1048576), b''): h.update(x)
    return h.hexdigest()

def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def atomic_json(path, obj):
    p=Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    q=p.with_name(p.name+f'.partial.{os.getpid()}')
    with q.open('w') as f:
        json.dump(obj, f, indent=2, sort_keys=True, allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(q,p)

def append(path,obj):
    p=Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('a') as f: f.write(json.dumps(obj,sort_keys=True,allow_nan=False)+'\n'); f.flush()

def atomic_npz(path, **arrays):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_name(p.name+f'.partial.{os.getpid()}')
    with q.open('wb') as f: np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
    with np.load(q,allow_pickle=False) as data:
        if set(data.files)!=set(arrays): raise ValueError('NPZ keys did not round trip')
    os.replace(q,p)

def read_protocol(): return json.loads((ROOT/'protocol.json').read_text())

ACTIVE_CUDA_DEVICE = None

def set_active_device(device):
    global ACTIVE_CUDA_DEVICE
    device=torch.device(device)
    ACTIVE_CUDA_DEVICE=device if device.type=='cuda' else None

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if ACTIVE_CUDA_DEVICE is not None: torch.cuda.manual_seed(seed)

def rng_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state(ACTIVE_CUDA_DEVICE) if ACTIVE_CUDA_DEVICE is not None else None}

def restore_rng(x):
    if (x['cuda'] is None)!=(ACTIVE_CUDA_DEVICE is None): raise ValueError('CPU/CUDA mode changed across checkpoint resume')
    random.setstate(x['python']);np.random.set_state(x['numpy']);torch.set_rng_state(x['torch'].cpu())
    if x['cuda'] is not None:
        if ACTIVE_CUDA_DEVICE is None: raise ValueError('CUDA checkpoint cannot silently resume as CPU')
        torch.cuda.set_rng_state(x['cuda'].cpu(),ACTIVE_CUDA_DEVICE)

def dataset(protocol, split, distribution, seed, count):
    arrays, meta=build_synthetic_dataset(protocol_id=protocol['experiment'],split=split,
        distribution=distribution,degree=protocol['degree'],base_seed=seed,count=count,
        parameters=protocol['distributions'][distribution])
    return TrainingBatch(arrays['pins'],arrays['net_ids'],arrays['offsets'],arrays['scales'],meta,array_bundle_sha256(arrays))

def costs(centers, candidates, rooted):
    """Rooted version forbids matching a source to a sink. Normalizer is n."""
    c=np.asarray(centers,dtype=np.float64);z=np.asarray(candidates,dtype=np.float64)
    if z.ndim==3: z=z[:,None]
    out=np.zeros(z.shape[:2],dtype=np.float64)
    for i in range(len(c)):
        for j in range(z.shape[1]):
            start=1 if rooted else 0
            d=np.abs(c[i,start:,None,:]-z[i,j,None,start:,:]).sum(-1)
            a,b=linear_sum_assignment(d)
            out[i,j]=(d[a,b].sum()+(np.abs(c[i,0]-z[i,j,0]).sum() if rooted else 0))/c.shape[1]
    return out

class Baseline(torch.nn.Module):
    """Permutation invariant sink aggregation with an explicit root embedding."""
    def __init__(self, device):
        super().__init__();self.device=device
        self.embed=torch.nn.Sequential(torch.nn.Linear(3,64),torch.nn.Tanh(),torch.nn.Linear(64,64),torch.nn.Tanh())
        self.head=torch.nn.Sequential(torch.nn.Linear(128,64),torch.nn.Tanh(),torch.nn.Linear(64,1));self.to(device)
    def forward(self,pins):
        x=torch.as_tensor(pins,dtype=torch.float32,device=self.device)
        flag=torch.zeros((*x.shape[:2],1),device=self.device);flag[:,0]=1
        h=self.embed(torch.cat([x,flag],-1))
        return self.head(torch.cat([h.mean(1),h[:,0]],-1)).squeeze(-1)

def make_adapter(family, degree, device, params):
    from importlib import import_module
    return import_module('adapter_'+family).Adapter(degree,device,params)

def verify_frozen():
    p=ROOT/'artifacts/frozen.json';f=json.loads(p.read_text())
    for name,value in f['files'].items():
        if sha(name)!=value: raise ValueError('Frozen source/data identity changed: '+name)
    return sha(p),f

def proc_identity(pid=None):
    pid=pid or os.getpid();p=Path(f'/proc/{pid}/stat')
    if not p.exists(): return None
    text=p.read_text();tail=text[text.rfind(')')+2:].split()
    return {'pid':pid,'start_ticks':int(tail[19]),'state':tail[0]}
