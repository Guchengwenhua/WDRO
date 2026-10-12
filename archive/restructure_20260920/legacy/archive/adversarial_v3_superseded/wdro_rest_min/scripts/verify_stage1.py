#!/usr/bin/env python3
"""End-to-end integrity, leakage, legality, cache, and ISPD audit for stage 1."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets import array_bundle_sha256, build_synthetic_dataset, file_sha256, load_dataset
from src.geosteiner import GeoSteinerEvaluator
from src.geometry import (
    MINIMUM_L1_SPACING,
    NORMALIZED_LOWER,
    NORMALIZED_UPPER,
    NetTransform,
    has_spacing_violation,
    inverse_net,
    normalize_net,
)
from src.protocol import load_protocol, protocol_sha256


def unordered_pin_fingerprint(pins: np.ndarray) -> bytes:
    points = np.asarray(pins, dtype="<f4")
    order = np.lexsort((points[:, 1], points[:, 0]))
    return hashlib.blake2b(points[order].tobytes(), digest_size=16).digest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets-root",
        type=Path,
        default=WORKSPACE_ROOT / "Codex" / "experiments" / "cache" / "datasets" / "protocol_v1",
    )
    parser.add_argument(
        "--lstar-root",
        type=Path,
        default=WORKSPACE_ROOT / "Codex" / "experiments" / "cache" / "lstar" / "protocol_v1",
    )
    parser.add_argument(
        "--ispd-manifest",
        type=Path,
        default=WORKSPACE_ROOT / "Codex" / "experiments" / "cache" / "ispd_protocol_a" / "ispd_protocol_a_manifest.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=WORKSPACE_ROOT / "Codex" / "experiments" / "logs" / "stage1_audit.json",
    )
    args = parser.parse_args()

    protocol = load_protocol()
    dataset_manifest_path = args.datasets_root / "datasets_manifest.json"
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    lstar_index_path = args.lstar_root / "lstar_index.json"
    lstar_index = json.loads(lstar_index_path.read_text(encoding="utf-8"))
    lstar_by_dataset_hash = {record["dataset_array_bundle_sha256"]: record for record in lstar_index["records"]}

    expected_grid = {
        (split, distribution, degree)
        for split in ("validation", "test")
        for distribution in protocol["data"]["distributions"]
        for degree in protocol["scope"]["degrees"]
    }
    actual_grid = set()
    split_ids: dict[str, set[bytes]] = {"validation": set(), "test": set()}
    split_pin_fingerprints: dict[str, set[bytes]] = {"validation": set(), "test": set()}
    evaluator = GeoSteinerEvaluator()
    dataset_records = []
    total_nets = 0
    sampled_lstar_rechecks = 0
    reproduced_datasets = 0

    for record in dataset_manifest["records"]:
        dataset_path = WORKSPACE_ROOT / record["path"]
        arrays, metadata = load_dataset(dataset_path, verify=True)
        key = (metadata["split"], metadata["distribution"], int(metadata["degree"]))
        if key in actual_grid:
            raise AssertionError(f"duplicate dataset grid cell {key}")
        actual_grid.add(key)
        expected_count = protocol["data"]["sizes_per_degree_distribution"][metadata["split"]]
        pins = arrays["pins"]
        ids = arrays["net_ids"]
        if pins.shape != (expected_count, metadata["degree"], 2):
            raise AssertionError(f"shape mismatch for {dataset_path}: {pins.shape}")
        if len(ids) != expected_count or len(np.unique(ids)) != expected_count:
            raise AssertionError(f"ID count/uniqueness mismatch for {dataset_path}")
        if not np.isfinite(pins).all():
            raise AssertionError(f"non-finite stored pins in {dataset_path}")
        tolerance = 2e-7
        if float(pins.min()) < NORMALIZED_LOWER - tolerance or float(pins.max()) > NORMALIZED_UPPER + tolerance:
            raise AssertionError(f"out-of-bounds stored pins in {dataset_path}")
        bbox_spans = pins.max(axis=1) - pins.min(axis=1)
        if not np.allclose(bbox_spans.max(axis=1), 0.8, atol=2e-7, rtol=0):
            raise AssertionError(f"longest bbox side is not 0.8 in {dataset_path}")

        rebuilt_arrays, _ = build_synthetic_dataset(
            protocol_id=protocol["protocol_id"],
            split=metadata["split"],
            distribution=metadata["distribution"],
            degree=metadata["degree"],
            base_seed=metadata["base_seed"],
            count=metadata["count"],
            parameters=metadata["parameters"],
        )
        if array_bundle_sha256(rebuilt_arrays) != metadata["array_bundle_sha256"]:
            raise AssertionError(f"deterministic rebuild mismatch for {dataset_path}")
        reproduced_datasets += 1

        for net in pins:
            if has_spacing_violation(net, MINIMUM_L1_SPACING):
                raise AssertionError(f"stored minimum-spacing violation in {dataset_path}")
            fingerprint = unordered_pin_fingerprint(net)
            if fingerprint in split_pin_fingerprints[metadata["split"]]:
                raise AssertionError(f"duplicate unordered pin set within {metadata['split']}")
            split_pin_fingerprints[metadata["split"]].add(fingerprint)
        for net_id in ids:
            if net_id in split_ids[metadata["split"]]:
                raise AssertionError(f"duplicate net ID within {metadata['split']}: {net_id!r}")
            split_ids[metadata["split"]].add(net_id)

        for sample_index in sorted({0, len(pins) - 1}):
            transform = NetTransform(
                offset_xy=arrays["offsets"][sample_index],
                scale=float(arrays["scales"][sample_index]),
                original_degree=metadata["degree"],
                unique_degree=metadata["degree"],
                duplicate_count=0,
            )
            restored = inverse_net(pins[sample_index], transform)
            renormalized, _ = normalize_net(restored, expected_degree=metadata["degree"])
            np.testing.assert_allclose(renormalized, pins[sample_index], atol=2e-7, rtol=0)

        dataset_hash = metadata["array_bundle_sha256"]
        if dataset_hash not in lstar_by_dataset_hash:
            raise AssertionError(f"missing L_star cache for {dataset_path}")
        cache_record = lstar_by_dataset_hash[dataset_hash]
        cache_path = WORKSPACE_ROOT / cache_record["cache_path"]
        cache_arrays, cache_metadata = load_dataset(cache_path, verify=True)
        if not np.array_equal(cache_arrays["net_ids"], ids):
            raise AssertionError(f"L_star net IDs mismatch for {dataset_path}")
        lstar = cache_arrays["lstar"]
        if len(lstar) != len(pins) or not np.isfinite(lstar).all() or np.any(lstar <= 0):
            raise AssertionError(f"invalid L_star vector for {dataset_path}")
        for sample_index in sorted({0, len(pins) - 1}):
            recomputed = evaluator.length(pins[sample_index])
            if not np.isclose(recomputed, lstar[sample_index], atol=1e-12, rtol=1e-12):
                raise AssertionError(f"L_star recheck mismatch for {dataset_path}:{sample_index}")
            sampled_lstar_rechecks += 1

        total_nets += len(pins)
        dataset_records.append({
            "path": record["path"],
            "array_bundle_sha256": dataset_hash,
            "npz_sha256": metadata["npz_sha256"],
            "lstar_cache_path": cache_record["cache_path"],
            "lstar_npz_sha256": cache_metadata["npz_sha256"],
            "count": len(pins),
        })

    if actual_grid != expected_grid:
        raise AssertionError(f"dataset grid mismatch; missing={expected_grid - actual_grid}, extra={actual_grid - expected_grid}")
    if split_ids["validation"] & split_ids["test"]:
        raise AssertionError("validation/test net IDs overlap")
    content_overlap = split_pin_fingerprints["validation"] & split_pin_fingerprints["test"]
    if content_overlap:
        raise AssertionError(f"validation/test unordered pin content overlaps: {len(content_overlap)}")
    if len(lstar_by_dataset_hash) != len(expected_grid):
        raise AssertionError("L_star index contains a missing or extra dataset cache")
    if lstar_index["total_evaluator_calls"] != total_nets:
        raise AssertionError("L_star evaluator call count does not equal dataset net count")

    ispd = json.loads(args.ispd_manifest.read_text(encoding="utf-8"))
    if ispd["base_design_count"] != 20 or ispd["contains_model_metrics"] or not ispd["model_metrics_locked"]:
        raise AssertionError("ISPD protocol-A lock/design count is invalid")
    if not all(item["duplicate_design_identity_confirmed"] for item in ispd["excluded_variants"]):
        raise AssertionError("an excluded metal5 variant does not match canonical net identity")
    ispd_dataset_count = 0
    for path in sorted(args.ispd_manifest.parent.glob("ispd*_test*/*.npz")):
        arrays, metadata = load_dataset(path, verify=True)
        if metadata.get("contains_model_metrics") is not False:
            raise AssertionError(f"ISPD artifact metric lock missing: {path}")
        if len(arrays["pins"]) != metadata["valid_count"]:
            raise AssertionError(f"ISPD valid count mismatch: {path}")
        ispd_dataset_count += 1

    report = {
        "schema_version": 1,
        "audited_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "pass",
        "protocol_canonical_sha256": protocol_sha256(protocol),
        "datasets": {
            "grid_cells": len(dataset_records),
            "total_nets": total_nets,
            "validation_nets": len(split_ids["validation"]),
            "test_nets": len(split_ids["test"]),
            "validation_test_id_overlap": 0,
            "validation_test_unordered_content_overlap": 0,
            "all_hashes_verified": True,
            "all_legal_after_float32_storage": True,
            "deterministically_rebuilt_grid_cells": reproduced_datasets,
        },
        "lstar": {
            "cache_count": len(lstar_by_dataset_hash),
            "total_evaluator_calls": lstar_index["total_evaluator_calls"],
            "sampled_exact_rechecks": sampled_lstar_rechecks,
            "all_ids_aligned": True,
            "all_lengths_finite_positive": True,
        },
        "ispd": {
            "base_design_count": ispd["base_design_count"],
            "dataset_count": ispd_dataset_count,
            "excluded_variant_count": ispd["excluded_variant_count"],
            "parsed_count": ispd["totals"]["parsed_count"],
            "valid_count": ispd["totals"]["valid_count"],
            "invalid_count": ispd["totals"]["invalid_count"],
            "model_metrics_locked": True,
        },
        "source_manifests": {
            "datasets": {
                "path": str(dataset_manifest_path.relative_to(WORKSPACE_ROOT)),
                "sha256": file_sha256(dataset_manifest_path),
            },
            "lstar": {
                "path": str(lstar_index_path.relative_to(WORKSPACE_ROOT)),
                "sha256": file_sha256(lstar_index_path),
            },
            "ispd": {
                "path": str(args.ispd_manifest.relative_to(WORKSPACE_ROOT)),
                "sha256": file_sha256(args.ispd_manifest),
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output.chmod(0o444)
    print(json.dumps({"output": str(args.output), "status": "pass", "total_nets": total_nets}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
