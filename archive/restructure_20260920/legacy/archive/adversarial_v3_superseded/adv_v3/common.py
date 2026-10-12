"""Shared, version-isolated data, geometry, scoring, and durable IO."""
from __future__ import annotations
import hashlib
import json
import os
import sqlite3
import sys
import time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "wdro_rest_min"
EXP = ROOT / "experiments"
sys.path.insert(0, str(CODE))
from src.datasets import generate_terminals, normalize_fixed_degree_batch
from src.geometry import normalize_net, project_normalized_net, has_spacing_violation, pin_set_sha256
from src.costs import hungarian_l1_mean
from src.rest import build_actor, build_critic, RestTreeEvaluator
from src.geosteiner import GeoSteinerEvaluator, ParallelGeoSteinerEvaluator
from src.candidates import ReferenceCache

DISTRIBUTIONS = json.loads((CODE / "configs/protocol_v1.json").read_text())["data"]["distributions"]
GEOMETRY_LEDGER = dict(proposals=0, repairs=0, fallbacks=0)

def derived_seed(*parts):
    return int.from_bytes(hashlib.sha256(
        ("adversarial_v3_20260912:" + ":".join(map(str, parts))).encode()
    ).digest()[:4], "little")

def json_default(value):
    if isinstance(value, Path): return str(value)
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    raise TypeError(type(value).__name__)

def atomic_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".partial.{os.getpid()}.{time.time_ns()}")
    with temp.open("w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=json_default, allow_nan=False)
        f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)

def atomic_torch(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".partial")
    torch.save(obj, temp); os.replace(temp, path)

def atomic_npz(path, **arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".partial")
    with temp.open("wb") as f:
        np.savez_compressed(f, **arrays); f.flush(); os.fsync(f.fileno())
    with np.load(temp, allow_pickle=False) as data:
        for k in arrays:
            if data[k].shape != np.asarray(arrays[k]).shape: raise RuntimeError("shape mismatch")
    os.replace(temp, path)

def append_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(obj, default=json_default, allow_nan=False) + "\n")
        f.flush()

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""): h.update(block)
    return h.hexdigest()

def make_data(degree, count, distribution, namespace, seed=0):
    # Each namespace is disjoint by construction; exact content overlap is audited later.
    output = []
    attempt = 0
    while sum(len(x) for x in output) < count:
        raw = generate_terminals(
            count, degree, derived_seed("data", namespace, seed, degree, distribution, attempt),
            distribution, DISTRIBUTIONS[distribution],
        )
        pins, _, _, valid, _ = normalize_fixed_degree_batch(raw)
        pins = pins[valid].astype(np.float32)
        pins = np.asarray([p for p in pins if not has_spacing_violation(p)])
        if len(pins): output.append(pins)
        attempt += 1
        if attempt > 20: raise RuntimeError("data generation exhausted")
    return np.concatenate(output)[:count].astype(np.float32)

def legalize(centers, deltas):
    x, delta = np.asarray(centers), np.asarray(deltas)
    if delta.ndim == 3:
        raw = np.clip(x + delta, .1, .9); shape = raw.shape
    elif delta.ndim == 4:
        raw = np.clip(x[:, None] + delta, .1, .9); shape = raw.shape
    else: raise ValueError("deltas require [B,n,2] or [B,K,n,2]")
    n = x.shape[1]; flat = raw.reshape(-1, n, 2)
    GEOMETRY_LEDGER["proposals"] += len(flat)
    normalized, _, _, valid, _ = normalize_fixed_degree_batch(flat)
    normalized = normalized.astype(np.float32)
    for i, p in enumerate(normalized):
        if valid[i] and not has_spacing_violation(p): continue
        GEOMETRY_LEDGER["repairs"] += 1
        center_i = i if delta.ndim == 3 else i // delta.shape[1]
        order = np.lexsort((x[center_i,:,1],x[center_i,:,0]))
        try:
            canonical = normalize_net(
                project_normalized_net(flat[i][order], minimum_l1_spacing=2e-6), expected_degree=n
            )[0].astype(np.float32)
            normalized[i,order] = canonical
            if has_spacing_violation(normalized[i]): raise ValueError("float32 spacing")
        except (ValueError, RuntimeError):
            GEOMETRY_LEDGER["fallbacks"] += 1
            normalized[i] = x[center_i]
    return normalized.reshape(shape)

def costs(centers, candidates):
    x, z = np.asarray(centers), np.asarray(candidates)
    if z.ndim != 4: raise ValueError("candidates require [B,K,n,2]")
    return np.asarray([[hungarian_l1_mean(a, p) for p in row] for a, row in zip(x, z)])

def models():
    return json.loads((ROOT/"manifests/parent_models_frozen.json").read_text())["models"]

def parent_model(method, degree, seed):
    return next(m for m in models() if (m["method"], m["degree"], m["seed"]) == (method,degree,seed))

def load_parent(method, degree, seed, device):
    meta = parent_model(method, degree, seed)
    path = meta["best"]["logical_path"]
    if sha256(path) != meta["best"]["sha256"]: raise RuntimeError("parent checkpoint hash mismatch")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    actor = build_actor(degree, device)
    critic = build_critic(degree, device)
    actor.load_state_dict(checkpoint["actor_state_dict"])
    critic.load_state_dict(checkpoint["critic_state_dict"])
    actor.eval(); critic.eval()
    return actor, critic, checkpoint, meta

def parent_price(degree, seed):
    meta = parent_model("wdro_rest", degree, seed)
    if sha256(meta["best"]["logical_path"]) != meta["best"]["sha256"]:
        raise RuntimeError("Price checkpoint hash mismatch")
    cp = torch.load(meta["best"]["logical_path"], map_location="cpu", weights_only=False)
    state = cp["method_state"]
    return float(state["lambda"]), float(state["rho"])

class Oracle:
    """Real actor/rest scores, exact cached references, monotonically durable call totals."""
    def __init__(self, actor, device, outdir, microbatch=256):
        self.actor, self.device = actor, torch.device(device)
        self.outdir = Path(outdir); self.outdir.mkdir(parents=True, exist_ok=True)
        self.ledger_path = self.outdir/"oracle_ledger.json"
        self.ledger = json.loads(self.ledger_path.read_text()) if self.ledger_path.exists() else {}
        self.rest = RestTreeEvaluator()
        self.reference_workers = int(os.environ.get("ADV_V3_REFERENCE_WORKERS","1"))
        self.geosteiner = ParallelGeoSteinerEvaluator(self.reference_workers)
        self.cache = ReferenceCache(self.outdir/"reference.sqlite")
        self.microbatch = microbatch
    def add(self, field, count):
        self.ledger[field] = self.ledger.get(field, 0) + int(count)
    def persist(self):
        rows = self.cache.db.execute(
            "SELECT COALESCE(SUM(requested),0),COALESCE(SUM(misses_started),0),COALESCE(SUM(misses_completed),0) FROM calls"
        ).fetchone()
        self.ledger.update(reference_requests=int(rows[0]), geosteiner_misses_started=int(rows[1]),
                           geosteiner_misses_completed=int(rows[2]))
        atomic_json(self.ledger_path,self.ledger)
    def evaluate(self, pins):
        points = np.asarray(pins, dtype=np.float32)
        if points.ndim != 3: raise ValueError("pins require [N,n,2]")
        self.add("actor_requested",len(points)); self.persist()
        was_training = self.actor.training; self.actor.eval()
        outputs=[]
        with torch.no_grad():
            for start in range(0,len(points),self.microbatch):
                out,_=self.actor(points[start:start+self.microbatch],True)
                outputs.append(out.cpu().numpy())
        self.actor.train(was_training)
        self.add("actor_completed",len(points)); self.add("rest_requested",len(points)); self.persist()
        length=self.rest.lengths(points,np.concatenate(outputs))
        self.add("rest_completed",len(points)); self.persist()
        reference,_=self.cache.lengths(points,self.geosteiner)
        self.persist()
        regret=length/reference-1
        if not np.isfinite(regret).all(): raise RuntimeError("nonfinite real scores")
        if np.any(regret < -1e-7): raise RuntimeError("REST below reference")
        return dict(regret=regret,lstar=reference,length=length)
    def close(self):
        self.persist(); self.cache.close(); self.geosteiner.close()
