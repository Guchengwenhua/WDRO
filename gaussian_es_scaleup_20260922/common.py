"""Local artifacts; read-only imports from the accepted v3.1 implementation."""
from __future__ import annotations
import collections
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
ART = ROOT / 'artifacts'
PARENT = ROOT.parent / 'adversarial_v3_1_20260912'
sys.path.insert(0, str(PARENT))
from adv_v3.common import (atomic_json, atomic_npz, atomic_torch, append_json, sha256,
    make_data, legalize, costs, derived_seed, parent_price, load_parent, Oracle,
    build_actor, build_critic, RestTreeEvaluator, ParallelGeoSteinerEvaluator)
from adv_v3.joint import _configure, _rng_state, _restore_rng, _cpu_copy, sampled_update
from src.geometry import pin_set_sha256
from src.rest import LEGACY_ROOT
from src.geosteiner import reference_identity

PROTOCOL = json.loads((ROOT / 'protocol.json').read_text())
DISTS = PROTOCOL['distributions']

def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def array_hash(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()

def data(degree, count, distribution, split, seed=0):
    return make_data(degree, count, distribution, PROTOCOL['data_namespace'] + ':' + split, seed)

def proc_ticks(pid):
    try:
        content = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return None if content[0] == 'Z' else content[19]
    except (FileNotFoundError, ProcessLookupError):
        return None

def artifact(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha256(path)}

def checked(binding):
    p = Path(binding['path'])
    if sha256(p) != binding['sha256']: raise RuntimeError(f'artifact drift: {p}')
    return p

def source_bindings():
    paths = list(ROOT.glob('*.py')) + list(ROOT.glob('*.sh')) + [ROOT/'protocol.json', ROOT/'docs/理论与实验方案.md']
    paths += list((PARENT/'adv_v3').glob('*.py'))
    paths += list((PARENT/'wdro_rest_min/src').glob('*.py'))
    paths += list((PARENT/'wdro_rest_min/stage5').glob('*.py'))
    paths += list((PARENT/'wdro_rest_min/configs').glob('*.json'))
    paths += [PARENT/'manifests/parent_models_frozen.json']
    paths += list((LEGACY_ROOT/'models').glob('*.py'))
    paths += [LEGACY_ROOT/'algorithms/libeval.so']
    paths += [LEGACY_ROOT/f'save/DAC21/rsmt{d}b.pt' for d in PROTOCOL['degrees']]
    return {str(p): sha256(p) for p in sorted(set(paths))}

def verify_freeze():
    manifest = json.loads((ART/'manifests/source_freeze.json').read_text())
    for name, expected in manifest['sources'].items():
        if sha256(name) != expected: raise RuntimeError(f'frozen source changed: {name}')
    return digest(manifest)

class MemoryReferenceCache:
    """Bounded solver-only cache. Eviction affects cost, never the objective."""
    def __init__(self, owner, capacity=16384):
        self.owner, self.capacity = owner, capacity
        self.values = collections.OrderedDict()
    def lengths(self, pins, evaluator):
        keys = [pin_set_sha256(p) for p in pins]
        unique = dict(zip(keys, pins))
        missing = [k for k in unique if k not in self.values]
        self.owner.add('reference_requests', len(pins))
        self.owner.add('geosteiner_misses_started', len(missing)); self.owner.persist()
        found = {k: self.values[k] for k in unique if k in self.values}
        if missing:
            lengths = evaluator.lengths(np.asarray([unique[k] for k in missing]))
            if not np.isfinite(lengths).all() or np.any(lengths <= 0): raise RuntimeError('invalid reference')
            found.update(zip(missing, map(float, lengths)))
            self.owner.add('geosteiner_misses_completed', len(missing))
        for k, v in found.items():
            self.values[k] = v; self.values.move_to_end(k)
        while len(self.values) > self.capacity: self.values.popitem(last=False)
        self.owner.persist()
        return np.asarray([found[k] for k in keys]), len(missing)
    def close(self): self.values.clear()

class ExactOracle(Oracle):
    def __init__(self, actor, device, outdir, workers=2):
        self.actor, self.device = actor, torch.device(device)
        self.outdir = Path(outdir); self.outdir.mkdir(parents=True, exist_ok=True)
        self.ledger_path = self.outdir/'oracle_ledger.json'
        self.ledger = json.loads(self.ledger_path.read_text()) if self.ledger_path.exists() else {}
        self.rest = RestTreeEvaluator()
        self.reference_workers = workers
        self.geosteiner = ParallelGeoSteinerEvaluator(workers)
        self.cache = MemoryReferenceCache(self)
        self.microbatch = 256
    def persist(self): atomic_json(self.ledger_path, self.ledger)

def build_initial(degree, seed, device):
    _configure(seed)
    actor, critic = build_actor(degree, device), build_critic(degree, device)
    p = LEGACY_ROOT / f'save/DAC21/rsmt{degree}b.pt'
    cp = torch.load(p, map_location='cpu', weights_only=False)
    actor.load_state_dict(cp['actor_state_dict'])
    return actor, critic, artifact(p)

def dual_settings(degree):
    price, rho = parent_price(degree, 101)
    if price < 0 or rho <= 0: raise RuntimeError('invalid frozen dual settings')
    return float(price), float(rho)

def best_key(values, initial, pre=False):
    if pre: return values['uniform']['mean']
    if values['uniform']['mean'] > initial['uniform']['mean'] + PROTOCOL['nominal_tolerance']:
        return float('inf')
    return max(values[d]['mean'] for d in DISTS[1:])

def evaluate(oracle, degree, count, split, seed=0, save_dir=None):
    result = {}
    for dist in DISTS:
        pins = data(degree, count, dist, split, seed)
        pred = oracle.evaluate(pins)
        regret = pred['regret']
        result[dist] = {'mean': float(regret.mean()), 'p95': float(np.quantile(regret, .95)), 'n': len(pins)}
        if save_dir is not None:
            atomic_npz(Path(save_dir)/f'{dist}.npz', pins=pins, **pred)
    return result
