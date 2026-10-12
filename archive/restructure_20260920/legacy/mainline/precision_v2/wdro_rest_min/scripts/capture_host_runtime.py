#!/usr/bin/env python3
"""Capture fresh multi-sample GPU/process/disk state for one launch decision."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
CODEX_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_ROOT = CODEX_ROOT / "experiments"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def query_csv(fields: str, *, compute: bool = False) -> list[list[str]]:
    selector = "--query-compute-apps" if compute else "--query-gpu"
    command = ["nvidia-smi", f"{selector}={fields}", "--format=csv,noheader,nounits"]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    if not completed.stdout.strip():
        return []
    return [[part.strip() for part in line.split(",")] for line in completed.stdout.splitlines()]


def gpu_sample() -> list[dict[str, Any]]:
    fields = (
        "index,uuid,name,memory.total,memory.used,memory.free,"
        "utilization.gpu,utilization.memory,temperature.gpu"
    )
    rows = []
    for values in query_csv(fields):
        index, uuid, name, total, used, free, util_gpu, util_memory, temperature = values
        rows.append(
            {
                "index": int(index),
                "uuid": uuid,
                "name": name,
                "memory_total_mib": int(total),
                "memory_used_mib": int(used),
                "memory_free_mib": int(free),
                "utilization_gpu_percent": int(util_gpu),
                "utilization_memory_percent": int(util_memory),
                "temperature_c": int(temperature),
            }
        )
    return rows


def compute_processes() -> list[dict[str, Any]]:
    rows = []
    for values in query_csv("gpu_uuid,pid,process_name,used_memory", compute=True):
        uuid, pid, name, memory = values
        rows.append(
            {"gpu_uuid": uuid, "pid": int(pid), "process_name": name, "used_memory_mib": int(memory)}
        )
    return rows


def disk_snapshot(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    usage = shutil.disk_usage(resolved)
    return {
        "logical_path": str(path),
        "resolved_path": str(resolved),
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
        "free_gib": round(usage.free / 1024**3, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--purpose", required=True)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--sample-interval-seconds", type=float, default=2.0)
    parser.add_argument("--measured-peak-mib", type=int)
    parser.add_argument("--safety-margin-mib", type=int)
    parser.add_argument("--minimum-free-disk-gib", type=float, default=100.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.samples < 2 or not 0 <= args.sample_interval_seconds <= 30:
        parser.error("use at least two samples and an interval in [0,30]")
    if args.measured_peak_mib is not None and args.measured_peak_mib <= 0:
        parser.error("measured peak must be positive")
    captured_at = utc_now()
    timestamp_slug = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    samples = []
    for sample_index in range(args.samples):
        samples.append(
            {
                "sample_index": sample_index,
                "captured_at_utc": utc_now(),
                "gpus": gpu_sample(),
                "compute_processes": compute_processes(),
            }
        )
        if sample_index + 1 < args.samples:
            time.sleep(args.sample_interval_seconds)

    peak = args.measured_peak_mib
    safety = args.safety_margin_mib
    if safety is None:
        safety = max(2048, int(math.ceil(peak * 0.25))) if peak is not None else 4096
    decisions = []
    indices = [row["index"] for row in samples[0]["gpus"]]
    for index in indices:
        rows = [next(row for row in sample["gpus"] if row["index"] == index) for sample in samples]
        minimum_free = min(row["memory_free_mib"] for row in rows)
        average_utilization = sum(row["utilization_gpu_percent"] for row in rows) / len(rows)
        maximum_utilization = max(row["utilization_gpu_percent"] for row in rows)
        if peak is None:
            memory_pass = max(row["memory_used_mib"] for row in rows) <= 1024
            memory_reason = "no measured training peak: conservative idle-GPU smoke only"
        else:
            memory_pass = minimum_free > peak + safety
            memory_reason = f"minimum free must exceed measured peak {peak} + margin {safety} MiB"
        compute_pass = average_utilization < 85.0 and not all(
            row["utilization_gpu_percent"] >= 90 for row in rows
        )
        reasons = []
        if not memory_pass:
            reasons.append(memory_reason)
        if not compute_pass:
            reasons.append("multi-sample compute utilization is persistently near saturation")
        decisions.append(
            {
                "index": index,
                "eligible": memory_pass and compute_pass,
                "minimum_free_memory_mib": minimum_free,
                "average_gpu_utilization_percent": round(average_utilization, 3),
                "maximum_gpu_utilization_percent": maximum_utilization,
                "memory_check_passed": memory_pass,
                "compute_check_passed": compute_pass,
                "reasons": reasons,
            }
        )

    codex_disk = disk_snapshot(CODEX_ROOT)
    experiments_disk = disk_snapshot(EXPERIMENTS_ROOT)
    disk_pass = experiments_disk["free_gib"] >= args.minimum_free_disk_gib
    eligible = [row["index"] for row in decisions if row["eligible"]]
    manifest = {
        "schema_version": 1,
        "kind": "host_runtime_preflight",
        "captured_at_utc": captured_at,
        "purpose": args.purpose,
        "research_protocol_override": False,
        "sampling": {
            "count": args.samples,
            "interval_seconds": args.sample_interval_seconds,
            "samples": samples,
        },
        "capacity_model": {
            "measured_peak_mib": peak,
            "safety_margin_mib": safety,
            "mode": "measured_peak_plus_margin" if peak is not None else "conservative_idle_smoke",
        },
        "gpu_decisions": decisions,
        "eligible_gpu_indices": eligible,
        "disk": {
            "codex_logical_root": codex_disk,
            "experiments_artifact_root": experiments_disk,
            "minimum_free_gib": args.minimum_free_disk_gib,
        },
        "checks": {
            "at_least_one_eligible_gpu": bool(eligible),
            "experiments_disk_meets_floor": disk_pass,
            "logical_paths_within_codex": True,
        },
        "preflight_passed": bool(eligible) and disk_pass,
    }
    output = args.output or PROJECT_ROOT / "manifests" / f"host_runtime_{timestamp_slug}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    sidecar = output.with_suffix(output.suffix + ".sha256")
    sidecar.write_text(f"{sha256_file(output)}  {output.name}\n", encoding="ascii")
    output.chmod(0o444)
    sidecar.chmod(0o444)
    print(
        json.dumps(
            {
                "manifest": str(output),
                "sha256": sha256_file(output),
                "eligible_gpu_indices": eligible,
                "experiments_free_gib": experiments_disk["free_gib"],
                "preflight_passed": manifest["preflight_passed"],
            },
            sort_keys=True,
        )
    )
    return 0 if manifest["preflight_passed"] else 2


if __name__ == "__main__":
    import math

    raise SystemExit(main())
