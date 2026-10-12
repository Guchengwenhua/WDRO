#!/usr/bin/env python3
"""Capture a reproducible environment/code manifest and enforce batch gates."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.protocol import DEFAULT_PROTOCOL_PATH, load_protocol, protocol_sha256


DEFAULT_RUNTIME_CONFIG_PATH = PROJECT_ROOT / "configs" / "runtime_migrated_server_v1.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tracked_project_files() -> Iterable[Path]:
    for relative_root in ("configs", "src", "scripts", "tests"):
        root = PROJECT_ROOT / relative_root
        for path in sorted(root.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                yield path


def combine_hash(file_hashes: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for path, file_hash in sorted(file_hashes.items()):
        digest.update(path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def gpu_snapshot() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    rows = []
    for line in completed.stdout.strip().splitlines():
        index, name, total, used, free, utilization = [part.strip() for part in line.split(",")]
        rows.append(
            {
                "index": int(index),
                "name": name,
                "memory_total_mib": int(total),
                "memory_used_mib": int(used),
                "memory_free_mib": int(free),
                "utilization_percent": int(utilization),
            }
        )
    return rows


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in ("torch", "numpy", "scipy", "matplotlib", "pytest", "PyYAML"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def run_smoke(degree: int = 10) -> dict[str, Any]:
    command = [sys.executable, str(PROJECT_ROOT / "scripts" / "smoke_rest_assets.py"), "--degree", str(degree)]
    completed = subprocess.run(command, cwd=WORKSPACE_ROOT, check=True, capture_output=True, text=True)
    return json.loads(completed.stdout.strip().splitlines()[-1])


def load_runtime_config(path: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    runtime = dict(protocol["runtime"])
    if not path.is_file():
        raise FileNotFoundError(f"runtime config not found: {path}")
    override = json.loads(path.read_text(encoding="utf-8"))
    if override.get("schema_version") != 1:
        raise ValueError("runtime config schema_version must be 1")
    if override.get("research_protocol_override") is not False:
        raise ValueError("runtime config must not override the research protocol")
    host_keys = {
        "python",
        "artifact_root",
        "allowed_gpu_indices",
        "forbidden_gpu_indices",
        "allowed_gpu_idle_memory_mib_max",
        "minimum_free_disk_gib_for_batch",
    }
    runtime.update({key: override[key] for key in host_keys})
    allowed = runtime["allowed_gpu_indices"]
    forbidden = runtime["forbidden_gpu_indices"]
    if not allowed or len(allowed) != len(set(allowed)):
        raise ValueError("runtime allowed_gpu_indices must be non-empty and unique")
    if set(allowed) & set(forbidden):
        raise ValueError("runtime allowed and forbidden GPU sets overlap")
    if runtime["minimum_free_disk_gib_for_batch"] < protocol["runtime"]["minimum_free_disk_gib_for_batch"]:
        raise ValueError("runtime config cannot relax the protocol disk floor")
    return runtime


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--runtime-config", type=Path, default=DEFAULT_RUNTIME_CONFIG_PATH)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-batch-ready", action="store_true")
    args = parser.parse_args()

    protocol = load_protocol(args.protocol)
    runtime = load_runtime_config(args.runtime_config, protocol)
    timestamp = datetime.now(timezone.utc)
    timestamp_slug = timestamp.strftime("%Y%m%dT%H%M%SZ")
    output = args.output or PROJECT_ROOT / "manifests" / f"preflight_{timestamp_slug}.json"

    project_hashes = {
        str(path.relative_to(WORKSPACE_ROOT)): sha256_file(path)
        for path in tracked_project_files()
    }
    legacy_hashes = {}
    for relative_path in protocol["tracked_legacy_assets"]:
        path = WORKSPACE_ROOT / relative_path
        legacy_hashes[relative_path] = sha256_file(path) if path.is_file() else None

    checkpoint_hashes = {}
    template = protocol["training"]["initial_actor_checkpoint_template"]
    for degree in protocol["scope"]["degrees"]:
        relative_path = template.format(degree=degree)
        path = WORKSPACE_ROOT / relative_path
        checkpoint_hashes[relative_path] = sha256_file(path) if path.is_file() else None

    gpus = gpu_snapshot()
    allowed_indices = set(runtime["allowed_gpu_indices"])
    idle_limit = runtime["allowed_gpu_idle_memory_mib_max"]
    allowed_rows = [row for row in gpus if row["index"] in allowed_indices]
    allowed_gpus_idle = len(allowed_rows) == len(allowed_indices) and all(
        row["memory_used_mib"] <= idle_limit for row in allowed_rows
    )

    artifact_root = WORKSPACE_ROOT / runtime["artifact_root"]
    disk = shutil.disk_usage(artifact_root.resolve())
    gib = 1024**3
    minimum_free_bytes = runtime["minimum_free_disk_gib_for_batch"] * gib
    python_matches = Path(sys.executable).resolve() == Path(runtime["python"]).resolve()
    assets_present = all(value is not None for value in legacy_hashes.values())
    checkpoints_present = all(value is not None for value in checkpoint_hashes.values())

    smoke: dict[str, Any]
    try:
        smoke = run_smoke()
        smoke_passed = smoke.get("status") == "pass"
    except Exception as error:
        smoke = {"status": "fail", "error": f"{type(error).__name__}: {error}"}
        smoke_passed = False

    checks = {
        "protocol_valid": True,
        "configured_python_active": python_matches,
        "legacy_assets_present": assets_present,
        "all_required_checkpoints_present": checkpoints_present,
        "allowed_gpus_idle_at_snapshot": allowed_gpus_idle,
        "rest_cpu_smoke_passed": smoke_passed,
        "free_disk_meets_batch_floor": disk.free >= minimum_free_bytes,
    }
    development_ready = all(value for key, value in checks.items() if key != "free_disk_meets_batch_floor")
    batch_ready = development_ready and checks["free_disk_meets_batch_floor"]

    manifest = {
        "schema_version": 1,
        "captured_at_utc": timestamp.isoformat().replace("+00:00", "Z"),
        "protocol": {
            "path": str(args.protocol.resolve().relative_to(WORKSPACE_ROOT)),
            "canonical_sha256": protocol_sha256(protocol),
            "status": protocol["status"],
        },
        "runtime_config": {
            "path": str(args.runtime_config.resolve().relative_to(WORKSPACE_ROOT)),
            "sha256": sha256_file(args.runtime_config),
            "runtime_id": json.loads(args.runtime_config.read_text(encoding="utf-8"))["runtime_id"],
            "effective": runtime,
        },
        "code": {
            "combined_sha256": combine_hash(project_hashes),
            "files": project_hashes,
            "legacy_assets": legacy_hashes,
        },
        "checkpoints": checkpoint_hashes,
        "environment": {
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "packages": package_versions(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "hardware": {
            "logical_cpu_count": os.cpu_count(),
            "gpus": gpus,
            "disk": {
                "path": str(artifact_root),
                "resolved_path": str(artifact_root.resolve()),
                "total_bytes": disk.total,
                "used_bytes": disk.used,
                "free_bytes": disk.free,
                "free_gib": round(disk.free / gib, 3),
                "minimum_free_gib_for_batch": runtime["minimum_free_disk_gib_for_batch"],
            },
        },
        "smoke": smoke,
        "checks": checks,
        "development_ready": development_ready,
        "batch_ready": batch_ready,
        "batch_blockers": [key for key, value in checks.items() if not value],
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sidecar = output.with_suffix(output.suffix + ".sha256")
    sidecar.write_text(f"{sha256_file(output)}  {output.name}\n", encoding="ascii")
    output.chmod(0o444)
    sidecar.chmod(0o444)

    print(json.dumps({
        "manifest": str(output),
        "manifest_sha256": sha256_file(output),
        "development_ready": development_ready,
        "batch_ready": batch_ready,
        "batch_blockers": manifest["batch_blockers"],
    }, sort_keys=True))
    if args.require_batch_ready and not batch_ready:
        return 2
    return 0 if development_ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
