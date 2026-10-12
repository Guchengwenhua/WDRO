#!/usr/bin/env python3
"""Freeze design-aware ISPD inputs while keeping all model metrics locked."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets import load_dataset, save_dataset
from src.ispd import (
    discover_base_designs,
    discover_excluded_variants,
    file_sha256,
    normalize_ispd_degree_file,
    parse_degree_file,
)
from src.protocol import load_protocol, protocol_sha256


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=WORKSPACE_ROOT / "DRO" / "benchmark")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "experiments" / "cache" / "ispd_protocol_a",
    )
    args = parser.parse_args()

    protocol = load_protocol()
    degrees = protocol["scope"]["degrees"]
    designs = discover_base_designs(args.source_root)
    excluded = discover_excluded_variants(args.source_root)
    if len(designs) != 20:
        raise RuntimeError(f"expected 20 base designs, found {len(designs)}")

    records = []
    for design_path in designs:
        design_id = design_path.name.removesuffix(".input")
        for degree in degrees:
            source = design_path / f"degree_{degree}.txt"
            if not source.is_file():
                records.append({
                    "design_id": design_id,
                    "degree": degree,
                    "status": "source_degree_missing",
                    "parsed_count": 0,
                    "valid_count": 0,
                    "invalid_count": 0,
                })
                continue
            output = args.output_root / design_id / f"degree_{degree}.npz"
            if output.exists():
                _, metadata = load_dataset(output, verify=True)
                status = "verified_existing"
            else:
                arrays, metadata = normalize_ispd_degree_file(source, design_id, degree)
                metadata["protocol_canonical_sha256"] = protocol_sha256(protocol)
                metadata["source_path"] = str(source.relative_to(WORKSPACE_ROOT))
                metadata = save_dataset(output, arrays, metadata)
                status = "created"
            records.append({
                "design_id": design_id,
                "degree": degree,
                "status": status,
                "dataset_path": str(output.relative_to(WORKSPACE_ROOT)),
                "source_sha256": metadata["source_sha256"],
                "array_bundle_sha256": metadata["array_bundle_sha256"],
                "parsed_count": metadata["parsed_count"],
                "valid_count": metadata["valid_count"],
                "invalid_count": metadata["invalid_count"],
                "invalid_reasons": metadata["invalid_reasons"],
            })
            print(json.dumps(records[-1], sort_keys=True), flush=True)

    excluded_records = []
    for variant in excluded:
        base_name = variant.name.replace("_metal5.input", ".input")
        base = args.source_root / base_name
        comparisons = []
        for degree in degrees:
            variant_file = variant / f"degree_{degree}.txt"
            base_file = base / f"degree_{degree}.txt"
            if variant_file.is_file() and base_file.is_file():
                variant_ids = [net.net_name for net in parse_degree_file(variant_file)]
                base_ids = [net.net_name for net in parse_degree_file(base_file)]
                comparisons.append({
                    "degree": degree,
                    "byte_identical_sha256": file_sha256(variant_file) == file_sha256(base_file),
                    "same_net_id_sequence": variant_ids == base_ids,
                    "same_net_id_set": set(variant_ids) == set(base_ids),
                    "net_count": len(base_ids),
                })
        excluded_records.append({
            "variant": variant.name,
            "canonical_design": base_name,
            "reason": "metal5_duplicate_variant",
            "required_degree_files_compared": comparisons,
            "duplicate_design_identity_confirmed": bool(comparisons)
            and all(item["same_net_id_sequence"] for item in comparisons),
            "all_compared_files_byte_identical": bool(comparisons)
            and all(item["byte_identical_sha256"] for item in comparisons),
        })

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "protocol": "A",
        "protocol_canonical_sha256": protocol_sha256(protocol),
        "model_metrics_locked": True,
        "contains_model_metrics": False,
        "base_design_count": len(designs),
        "base_design_ids": [path.name.removesuffix(".input") for path in designs],
        "excluded_variant_count": len(excluded_records),
        "excluded_variants": excluded_records,
        "records": records,
        "totals": {
            "parsed_count": sum(record["parsed_count"] for record in records),
            "valid_count": sum(record["valid_count"] for record in records),
            "invalid_count": sum(record["invalid_count"] for record in records),
        },
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "ispd_protocol_a_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_path.chmod(0o444)
    print(json.dumps({"manifest": str(manifest_path), "totals": manifest["totals"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
