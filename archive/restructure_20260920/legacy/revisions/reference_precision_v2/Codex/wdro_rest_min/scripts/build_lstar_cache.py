#!/usr/bin/env python3
"""Compute immutable GeoSteiner L_star caches keyed by dataset content hash."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets import load_dataset, save_dataset
from src.geosteiner import DEFAULT_LIBRARY, GeoSteinerEvaluator


_EVALUATOR: GeoSteinerEvaluator | None = None


def _initialize_worker(library_path: str) -> None:
    global _EVALUATOR
    _EVALUATOR = GeoSteinerEvaluator(library_path)


def _evaluate_chunk(chunk: np.ndarray) -> np.ndarray:
    if _EVALUATOR is None:
        raise RuntimeError("worker evaluator was not initialized")
    return _EVALUATOR.lengths(chunk)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets-root",
        type=Path,
        default=WORKSPACE_ROOT / "Codex" / "experiments" / "cache" / "datasets" / "protocol_v1",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=WORKSPACE_ROOT / "Codex" / "experiments" / "cache" / "lstar" / "protocol_v1",
    )
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--chunk-size", type=int, default=128)
    args = parser.parse_args()

    if args.workers <= 0 or args.chunk_size <= 0:
        parser.error("workers and chunk-size must be positive")
    dataset_paths = sorted(args.datasets_root.glob("*/*.npz"))
    if not dataset_paths:
        raise FileNotFoundError(f"no frozen datasets under {args.datasets_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    records = []
    context = mp.get_context("spawn")
    with context.Pool(args.workers, initializer=_initialize_worker, initargs=(str(args.library),)) as pool:
        for dataset_path in dataset_paths:
            arrays, dataset_metadata = load_dataset(dataset_path, verify=True)
            content_hash = dataset_metadata["array_bundle_sha256"]
            output = args.output_root / f"{content_hash}.npz"
            started = time.monotonic()
            if output.exists():
                cache_arrays, cache_metadata = load_dataset(output, verify=True)
                if not np.array_equal(cache_arrays["net_ids"], arrays["net_ids"]):
                    raise ValueError(f"cached IDs do not match {dataset_path}")
                status = "verified_existing"
            else:
                pins = arrays["pins"]
                chunks = [pins[start : start + args.chunk_size] for start in range(0, len(pins), args.chunk_size)]
                results = pool.map(_evaluate_chunk, chunks)
                lengths = np.concatenate(results).astype(np.float64)
                if len(lengths) != len(pins) or not np.isfinite(lengths).all() or np.any(lengths <= 0):
                    raise RuntimeError(f"invalid GeoSteiner output for {dataset_path}")
                cache_arrays = {"lstar": lengths, "net_ids": arrays["net_ids"]}
                cache_metadata = {
                    "schema_version": 1,
                    "kind": "geosteiner_lstar",
                    "dataset_path": str(dataset_path.relative_to(WORKSPACE_ROOT)),
                    "dataset_array_bundle_sha256": content_hash,
                    "degree": dataset_metadata["degree"],
                    "count": len(lengths),
                    "evaluator_calls": len(lengths),
                    "library_path": str(args.library.relative_to(WORKSPACE_ROOT)),
                }
                cache_metadata = save_dataset(output, cache_arrays, cache_metadata)
                status = "created"
            record = {
                "dataset_path": str(dataset_path.relative_to(WORKSPACE_ROOT)),
                "dataset_array_bundle_sha256": content_hash,
                "cache_path": str(output.relative_to(WORKSPACE_ROOT)),
                "cache_npz_sha256": cache_metadata["npz_sha256"],
                "count": int(cache_metadata["count"]),
                "status": status,
                "wall_seconds": round(time.monotonic() - started, 6),
            }
            records.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)

    index = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "library": str(args.library.relative_to(WORKSPACE_ROOT)),
        "workers": args.workers,
        "dataset_count": len(records),
        "total_evaluator_calls": sum(record["count"] for record in records),
        "records": records,
    }
    index_path = args.output_root / "lstar_index.json"
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    index_path.chmod(0o444)
    print(json.dumps({"index": str(index_path), "total_evaluator_calls": index["total_evaluator_calls"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

