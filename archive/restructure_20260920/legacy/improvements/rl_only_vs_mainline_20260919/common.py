"""Shared own-action evaluation utilities; no reference solver is imported."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
RL = PROJECT / 'improvements/rl_only_20260915'
MAIN = PROJECT / 'revisions/reference_precision_v2/Codex'
spec = importlib.util.spec_from_file_location('rl_geometry', RL / 'geometry.py')
G = importlib.util.module_from_spec(spec)
spec.loader.exec_module(G)
GROUPS = ['uniform', 'mix_gauss', 'edge_bias', 'grid_cluster']
SEEDS = [101, 102, 103]
read = lambda path: json.loads(Path(path).read_text())
sha = G.sha256
atomic = G.atomic_json


def checked(path, digest):
    path = Path(path)
    if sha(path) != digest:
        raise AssertionError(f'changed artifact: {path}')
    return path


def install_guard(models):
    """Allow only the explicitly frozen comparison weights; block solver labels."""
    allowed = {str(Path(m['checkpoint']).resolve()) for m in models}
    def guard(event, args):
        if event == 'import':
            name = str(args[0]).lower()
            if any(name == p or name.startswith(p + '.') for p in
                   ('geosteiner', 'src.geosteiner', 'scipy.optimize', 'scipy.sparse.csgraph', 'networkx', 'ortools')):
                raise RuntimeError('comparison forbids reference solver imports')
        if event in ('ctypes.dlopen', 'subprocess.Popen', 'os.system'):
            if any(x in str(args).lower() for x in ('geosteiner', 'libeval', 'gst_rsmt', 'gurobi', 'cplex', 'flute')):
                raise RuntimeError('comparison forbids reference solver execution')
        if event == 'open' and isinstance(args[0], (str, bytes)):
            name = str(args[0]); lower = name.lower()
            if any(x in lower for x in ('l_star', 'lstar', 'reference.sqlite', 'optimal_labels', 'mst_cache')):
                raise RuntimeError('comparison forbids reference labels')
            if lower.endswith(('.pt', '.pth', '.ckpt')) and str(Path(name).resolve()) not in allowed:
                raise RuntimeError('unlisted comparison checkpoint')
    sys.addaudithook(guard)


def independent_lengths(pins, topology):
    x = np.asarray(pins, np.float64); t = np.asarray(topology); out = []
    n = x.shape[1]
    for start in range(0, len(x), 256):
        p = x[start:start+256]; a = t[start:start+256]; b = len(p)
        links = np.broadcast_to(np.eye(n, dtype=bool), (b, n, n)).copy()
        links[np.arange(b)[:, None], a[:, ::2], a[:, 1::2]] = True
        yy = np.where(links, p[:, None, :, 1], np.nan)
        xx = np.where(links.transpose(0, 2, 1), p[:, None, :, 0], np.nan)
        out.append((np.nanmax(yy, 2)-np.nanmin(yy, 2)+np.nanmax(xx, 2)-np.nanmin(xx, 2)).sum(1))
    return np.concatenate(out)


def data_path(suite, group):
    return ROOT / 'artifacts/data' / f'{suite}__{group}.npz'


def check_frozen(manifest):
    for name, digest in manifest['bindings'].items():
        checked(name, digest)


def state_hash(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
