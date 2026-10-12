"""Direct, location-independent binding for the repository GeoSteiner evaluator."""

from __future__ import annotations

import ctypes
import math
import multiprocessing as mp
from pathlib import Path

import numpy as np


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LIBRARY = WORKSPACE_ROOT / "DRO" / "WDRO-REST" / "algorithms" / "libeval.so"


class GeoSteinerEvaluator:
    def __init__(self, library_path: str | Path = DEFAULT_LIBRARY) -> None:
        self.library_path = Path(library_path).resolve()
        if not self.library_path.is_file():
            raise FileNotFoundError(self.library_path)
        library = ctypes.cdll.LoadLibrary(str(self.library_path))
        library.gst_open_geosteiner()
        function = library.call_gst_rsmt
        function.argtypes = (
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        )
        function.restype = ctypes.c_int
        self._library = library
        self._function = function

    def length(self, pins: np.ndarray) -> float:
        points = np.asarray(pins, dtype=np.float64, order="C")
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
            raise ValueError("pins must have shape (degree,2) with degree >= 2")
        if not np.isfinite(points).all():
            raise ValueError("pins must be finite")
        degree = len(points)
        length = ctypes.c_double()
        number_steiners = ctypes.c_int()
        steiners = (ctypes.c_double * (degree * 2))()
        number_edges = ctypes.c_int()
        edges = (ctypes.c_int * (degree * 4))()
        status = self._function(
            degree,
            points.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.byref(length),
            ctypes.byref(number_steiners),
            steiners,
            ctypes.byref(number_edges),
            edges,
        )
        if status != 0:
            raise RuntimeError(f"GeoSteiner failed with status {status}")
        value = float(length.value)
        if not math.isfinite(value) or value <= 0:
            raise RuntimeError(f"GeoSteiner returned invalid length {value}")
        return value

    def lengths(self, batch: np.ndarray) -> np.ndarray:
        nets = np.asarray(batch)
        if nets.ndim != 3 or nets.shape[-1] != 2:
            raise ValueError("batch must have shape (batch,degree,2)")
        return np.asarray([self.length(net) for net in nets], dtype=np.float64)


_WORKER_EVALUATOR: GeoSteinerEvaluator | None = None


def _initialize_worker(library_path: str) -> None:
    global _WORKER_EVALUATOR
    _WORKER_EVALUATOR = GeoSteinerEvaluator(library_path)


def _worker_lengths(batch: np.ndarray) -> np.ndarray:
    if _WORKER_EVALUATOR is None:
        raise RuntimeError("GeoSteiner worker was not initialized")
    return _WORKER_EVALUATOR.lengths(batch)


class ParallelGeoSteinerEvaluator:
    """Order-preserving process pool with identical exact GeoSteiner outputs."""

    def __init__(self, workers: int, library_path: str | Path = DEFAULT_LIBRARY) -> None:
        if workers <= 0:
            raise ValueError("workers must be positive")
        self.workers = workers
        self.local = GeoSteinerEvaluator(library_path)
        self._pool = None
        if workers > 1:
            context = mp.get_context("spawn")
            self._pool = context.Pool(
                workers,
                initializer=_initialize_worker,
                initargs=(str(Path(library_path).resolve()),),
            )

    def length(self, pins: np.ndarray) -> float:
        return self.local.length(pins)

    def lengths(self, batch: np.ndarray) -> np.ndarray:
        nets = np.asarray(batch)
        if self._pool is None or len(nets) <= 1:
            return self.local.lengths(nets)
        chunks = [chunk for chunk in np.array_split(nets, min(self.workers, len(nets))) if len(chunk)]
        return np.concatenate(self._pool.map(_worker_lengths, chunks)).astype(np.float64, copy=False)

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None

    def __enter__(self) -> "ParallelGeoSteinerEvaluator":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()
