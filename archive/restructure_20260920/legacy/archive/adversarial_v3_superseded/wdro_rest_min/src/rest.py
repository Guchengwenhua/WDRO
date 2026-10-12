"""Location-independent access to the frozen legacy REST model and evaluator."""

from __future__ import annotations

import ctypes
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
LEGACY_ROOT = WORKSPACE_ROOT / "DRO" / "WDRO-REST"
DEFAULT_LIBRARY = LEGACY_ROOT / "algorithms" / "libeval.so"


def _legacy_models() -> tuple[type[torch.nn.Module], type[torch.nn.Module]]:
    """Import the tracked legacy model without copying or modifying it."""

    legacy = str(LEGACY_ROOT)
    if legacy not in sys.path:
        sys.path.insert(0, legacy)
    from models.actor_critic import Actor, Critic

    return Actor, Critic


def build_actor(degree: int, device: torch.device) -> torch.nn.Module:
    actor_class, _ = _legacy_models()
    return actor_class(degree, device)


def build_critic(degree: int, device: torch.device) -> torch.nn.Module:
    _, critic_class = _legacy_models()
    return critic_class(degree, device)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_trusted_legacy_actor(
    actor: torch.nn.Module,
    checkpoint_path: str | Path,
    *,
    map_location: torch.device | str,
) -> dict[str, object]:
    """Load only Actor weights from a trusted DAC21 checkpoint.

    PyTorch 2.6 defaults to ``weights_only=True``.  The tracked DAC21 files
    contain legacy NumPy metadata, so the frozen protocol explicitly permits
    ``weights_only=False`` for these trusted files.
    """

    source = Path(checkpoint_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    checkpoint = torch.load(source, map_location=map_location, weights_only=False)
    if "actor_state_dict" not in checkpoint:
        raise ValueError(f"legacy checkpoint has no actor_state_dict: {source}")
    actor.load_state_dict(checkpoint["actor_state_dict"], strict=True)
    return {
        "path": str(source),
        "sha256": file_sha256(source),
        "legacy_batch_idx": int(checkpoint.get("batch_idx", -1)),
        "legacy_best_eval": float(checkpoint.get("best_eval", float("nan"))),
        "weights_only": False,
        "loaded_components": ["actor_state_dict"],
        "ignored_components": sorted(key for key in checkpoint if key != "actor_state_dict"),
    }


class RestTreeEvaluator:
    """Absolute-path ctypes binding for the REST tree-length evaluator."""

    def __init__(self, library_path: str | Path = DEFAULT_LIBRARY) -> None:
        self.library_path = Path(library_path).resolve()
        if not self.library_path.is_file():
            raise FileNotFoundError(self.library_path)
        library = ctypes.cdll.LoadLibrary(str(self.library_path))
        function = library.eval
        function.argtypes = (
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
        )
        function.restype = ctypes.c_double
        self._library = library
        self._function = function

    def length(self, pins: np.ndarray, output: np.ndarray) -> float:
        points = np.ascontiguousarray(pins, dtype=np.float64)
        topology = np.ascontiguousarray(output, dtype=np.int32)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
            raise ValueError("pins must have shape (degree,2), degree >= 2")
        if topology.shape != (2 * (len(points) - 1),):
            raise ValueError("REST output has the wrong shape")
        if not np.isfinite(points).all():
            raise ValueError("pins must be finite")
        value = float(
            self._function(
                points.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                topology.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                len(points),
            )
        )
        if not np.isfinite(value) or value <= 0:
            raise RuntimeError(f"REST evaluator returned invalid length {value}")
        return value

    def lengths(self, pins: np.ndarray, outputs: np.ndarray) -> np.ndarray:
        nets = np.asarray(pins)
        topologies = np.asarray(outputs)
        if nets.ndim != 3 or nets.shape[-1] != 2:
            raise ValueError("pins must have shape (batch,degree,2)")
        expected = (len(nets), 2 * (nets.shape[1] - 1))
        if topologies.shape != expected:
            raise ValueError(f"outputs must have shape {expected}")
        return np.asarray(
            [self.length(net, topology) for net, topology in zip(nets, topologies, strict=True)],
            dtype=np.float64,
        )
