#!/usr/bin/env python3
"""Generate and freeze all protocol-v1 synthetic validation/test grids."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets import build_synthetic_dataset, load_dataset, save_dataset
from src.protocol import DEFAULT_PROTOCOL_PATH, load_protocol, protocol_sha256


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--data-contract", type=Path, default=PROJECT_ROOT / "configs" / "data_contract_v1.json")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=WORKSPACE_ROOT / "Codex" / "experiments" / "cache" / "datasets" / "protocol_v1",
    )
    parser.add_argument("--splits", nargs="+", choices=("validation", "test"), default=("validation", "test"))
    parser.add_argument("--degrees", nargs="+", type=int)
    parser.add_argument("--distributions", nargs="+")
    args = parser.parse_args()

    protocol = load_protocol(args.protocol)
    data_contract = json.loads(args.data_contract.read_text(encoding="utf-8"))
    if data_contract.get("status") != "frozen":
        raise ValueError("data contract must be frozen")
    degrees = args.degrees or protocol["scope"]["degrees"]
    distributions = args.distributions or list(protocol["data"]["distributions"])
    records = []
    for split in args.splits:
        base_seed = protocol["data"]["seeds"][split]
        count = protocol["data"]["sizes_per_degree_distribution"][split]
        for distribution in distributions:
            parameters = protocol["data"]["distributions"][distribution]
            for degree in degrees:
                output = args.output_root / split / f"{distribution}_d{degree}_s{base_seed}_n{count}.npz"
                if output.exists():
                    _, metadata = load_dataset(output, verify=True)
                    status = "verified_existing"
                else:
                    arrays, metadata = build_synthetic_dataset(
                        protocol_id=protocol["protocol_id"],
                        split=split,
                        distribution=distribution,
                        degree=degree,
                        base_seed=base_seed,
                        count=count,
                        parameters=parameters,
                    )
                    metadata["protocol_canonical_sha256"] = protocol_sha256(protocol)
                    metadata["data_contract_sha256"] = sha256_file(args.data_contract)
                    metadata = save_dataset(output, arrays, metadata)
                    status = "created"
                record = {
                    "path": str(output.relative_to(WORKSPACE_ROOT)),
                    "status": status,
                    "split": split,
                    "distribution": distribution,
                    "degree": degree,
                    "count": metadata["count"],
                    "array_bundle_sha256": metadata["array_bundle_sha256"],
                    "npz_sha256": metadata["npz_sha256"],
                    "rejections": metadata["rejections"],
                }
                records.append(record)
                print(json.dumps(record, sort_keys=True), flush=True)

    expected = len(args.splits) * len(distributions) * len(degrees)
    if len(records) != expected:
        raise RuntimeError(f"expected {expected} dataset records, got {len(records)}")
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "protocol_canonical_sha256": protocol_sha256(protocol),
        "data_contract_sha256": sha256_file(args.data_contract),
        "dataset_count": len(records),
        "records": records,
    }
    manifest_path = args.output_root / "datasets_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_path.chmod(0o444)
    print(json.dumps({"manifest": str(manifest_path), "dataset_count": len(records)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

