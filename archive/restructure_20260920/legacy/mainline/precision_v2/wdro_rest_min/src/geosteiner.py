"""GeoSteiner on an exact power-of-two scale, with status and geometry checks.

All contract float32 coordinates in [0.1, 0.9] become exact integers at 2**27.
No rounding, clipping, or change to the Actor's coordinates occurs. General
float64 diagnostic inputs retain their exact binary scaling but are not claimed
to have the integer-input guarantee. Formal cache construction checks that
guarantee separately. GeoSteiner's LP_SCALE remains disabled.
"""
from __future__ import annotations
import ctypes
import hashlib
import math
import multiprocessing as mp
from pathlib import Path
import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_LIBRARY = WORKSPACE_ROOT / 'DRO' / 'WDRO-REST' / 'algorithms' / 'libeval.so'
SCALE = 2**27

def reference_identity(library_path=DEFAULT_LIBRARY):
    digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    return dict(reference_id='geosteiner_exact_binary_scale_v2', scale=SCALE,
                rounding='none', require_status='optimal', validate_embedding=True,
                library_sha256=digest(library_path), implementation_sha256=digest(__file__))

class GeoSteinerEvaluator:
    def __init__(self, library_path=DEFAULT_LIBRARY):
        self.library_path = Path(library_path).resolve()
        self._library = ctypes.cdll.LoadLibrary(str(self.library_path))
        self._library.gst_open_geosteiner()
        self._function = self._library.gst_rsmt
        self._function.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.c_void_p)
        self._function.restype = ctypes.c_int

    def solve(self, pins, *, witness=False, require_integral=False):
        points = np.asarray(pins, dtype=np.float64, order='C')
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2 or not np.isfinite(points).all():
            raise ValueError('finite pins with shape (degree,2), degree>=2 required')
        scaled = np.ascontiguousarray(points * SCALE)
        if not np.isfinite(scaled).all() or not np.array_equal(scaled / SCALE, points):
            raise ValueError('binary scaling must be exactly reversible')
        if np.max(np.abs(scaled)) * len(points) * 4 >= 2**52:
            raise ValueError('coordinates exceed the exact-integer arithmetic budget')
        integral = bool(np.array_equal(scaled, np.rint(scaled)))
        if require_integral and not integral:
            raise ValueError('formal coordinates must be integral after exact scaling')
        n = len(points)
        length, ns, ne, status = ctypes.c_double(), ctypes.c_int(), ctypes.c_int(), ctypes.c_int(-99)
        steiners = np.empty((4*n, 2), dtype=np.float64)
        edges = np.empty((6*n, 2), dtype=np.int32)
        rc = self._function(n, scaled.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ctypes.byref(length),
            ctypes.byref(ns), steiners.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ctypes.byref(ne),
            edges.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), ctypes.byref(status), None)
        if rc != 0 or status.value != 0:
            raise RuntimeError(f'GeoSteiner failed: return={rc}, solution_status={status.value}')
        if not (0 <= ns.value <= 4*n and 0 <= ne.value <= 6*n):
            raise RuntimeError('invalid embedding sizes')
        nodes = np.concatenate((scaled, steiners[:ns.value]))
        topology = edges[:ne.value]
        if not np.isfinite(nodes).all() or np.any(topology<0) or np.any(topology>=len(nodes)):
            raise RuntimeError('invalid embedding')
        parent = list(range(len(nodes)))
        def root(k):
            while parent[k] != k:
                parent[k] = parent[parent[k]]
                k = parent[k]
            return k
        for a,b in topology:
            parent[root(int(a))] = root(int(b))
        if len({root(i) for i in range(len(nodes))}) != 1:
            raise RuntimeError('GeoSteiner embedding is disconnected')
        if integral:
            if not np.array_equal(nodes, np.rint(nodes)):
                raise RuntimeError('integer input produced noninteger rectilinear embedding')
            integer_nodes = nodes.astype(np.int64)
            embedded = int(np.abs(integer_nodes[topology[:,0]]-integer_nodes[topology[:,1]]).sum())
            if embedded != length.value:
                raise RuntimeError('integer embedding length differs from objective')
        else:
            embedded = float(np.abs(nodes[topology[:,0]]-nodes[topology[:,1]]).sum())
            if not math.isclose(embedded, length.value, rel_tol=1e-13, abs_tol=1e-6):
                raise RuntimeError('embedding length differs from objective')
        value = length.value / SCALE
        if not math.isfinite(value) or value <= 0:
            raise RuntimeError('invalid reference length')
        if witness:
            return dict(length=value, scaled_length=length.value, scale=SCALE, integral_input=integral,
                        return_code=rc, solution_status=status.value, nodes=nodes.tolist(), edges=topology.tolist())
        return value

    def length(self, pins):
        return self.solve(pins)

    def lengths(self, batch):
        nets = np.asarray(batch)
        if nets.ndim != 3 or nets.shape[-1] != 2:
            raise ValueError('batch must have shape (batch,degree,2)')
        return np.asarray([self.length(net) for net in nets], dtype=np.float64)

_WORKER_EVALUATOR = None

def _initialize_worker(library_path):
    global _WORKER_EVALUATOR
    _WORKER_EVALUATOR = GeoSteinerEvaluator(library_path)

def _worker_lengths(batch):
    return _WORKER_EVALUATOR.lengths(batch)

class ParallelGeoSteinerEvaluator:
    def __init__(self, workers, library_path=DEFAULT_LIBRARY):
        if workers<=0:
            raise ValueError('workers must be positive')
        self.workers=workers
        self.local=GeoSteinerEvaluator(library_path)
        self._pool=None
        if workers>1:
            self._pool=mp.get_context('spawn').Pool(workers,initializer=_initialize_worker,initargs=(str(Path(library_path).resolve()),))
    def length(self,pins):
        return self.local.length(pins)
    def lengths(self,batch):
        nets=np.asarray(batch)
        if self._pool is None or len(nets)<=1:
            return self.local.lengths(nets)
        chunks=[c for c in np.array_split(nets,min(self.workers,len(nets))) if len(c)]
        return np.concatenate(self._pool.map(_worker_lengths,chunks)).astype(np.float64,copy=False)
    def close(self):
        if self._pool is not None:
            self._pool.close();self._pool.join();self._pool=None
    def __enter__(self):
        return self
    def __exit__(self,*args):
        self.close()
