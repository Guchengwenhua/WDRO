#!/usr/bin/env python3
"""Stage-5 WDRO entry point with GPU isolation and resumable job metadata."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
CODEX_ROOT = Path(__file__).resolve().parents[2]


def _preparse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--physical-gpu-index", type=int)
    return parser.parse_known_args()[0]


early = _preparse()
os.environ["TMPDIR"] = str(CODEX_ROOT / "experiments" / "tmp")
os.environ["TMP"] = os.environ["TMPDIR"]
os.environ["TEMP"] = os.environ["TMPDIR"]
os.environ["XDG_CACHE_HOME"] = str(CODEX_ROOT / "experiments" / "cache" / "runtime")
os.environ["TORCH_HOME"] = str(CODEX_ROOT / "experiments" / "cache" / "runtime" / "torch")
os.environ["CUDA_CACHE_PATH"] = str(CODEX_ROOT / "experiments" / "cache" / "runtime" / "cuda")
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
for environment_path in ("TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME", "TORCH_HOME"):
    Path(os.environ[environment_path]).mkdir(parents=True, exist_ok=True)
if early.device == "cuda":
    if early.physical_gpu_index is None:
        raise SystemExit("--physical-gpu-index is required for CUDA runs")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(early.physical_gpu_index)

sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.job_control import JobLease
from src.protocol import DEFAULT_PROTOCOL_PATH, load_protocol
from src.trainer import DEFAULT_TRAINER_CONFIG, load_trainer_config
from stage5.trainer import train_wdro


def _load_host_runtime(path: Path, physical_gpu_index: int) -> dict[str, object]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("kind") != "host_runtime_preflight" or not manifest.get("preflight_passed"):
        raise ValueError("host-runtime manifest is not a passing preflight")
    if physical_gpu_index not in manifest["eligible_gpu_indices"]:
        raise ValueError(f"GPU {physical_gpu_index} was not eligible in {path}")
    captured = datetime.fromisoformat(manifest["captured_at_utc"].replace("Z", "+00:00"))
    age = (datetime.now(timezone.utc) - captured).total_seconds()
    if age > 900:
        raise ValueError(f"host-runtime manifest is stale ({age:.1f}s > 900s)")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rho-mode", choices=["q50", "q75", "zero"], default="q50")
    parser.add_argument("--wdro-config", type=Path, required=True)
    parser.add_argument("--baseline-config", type=Path, default=PROJECT_ROOT / "configs" / "stage3_baselines_v1.json")
    parser.add_argument("--degree", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--scope", choices=["stage5_smoke", "ablation"], required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--physical-gpu-index", type=int)
    parser.add_argument("--host-runtime-manifest", type=Path)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--trainer-config", type=Path, default=DEFAULT_TRAINER_CONFIG)
    parser.add_argument("--run-id")
    parser.add_argument("--geosteiner-workers", type=int, default=1)
    args = parser.parse_args()

    wdro_config = json.loads(args.wdro_config.read_text())
    baseline_config = json.loads(args.baseline_config.read_text())
    if args.scope == "ablation" and (wdro_config["status"] != "frozen" or baseline_config["status"] != "frozen"):
        raise ValueError("formal runs require frozen baseline implementation config")
    if args.run_id and ("/" in args.run_id or args.run_id in (".", "..")):
        raise ValueError("run_id must be a simple path component")
    protocol = load_protocol(args.protocol)
    trainer_config = load_trainer_config(args.trainer_config)
    expected_updates = 2000
    stop_after = args.stop_after if args.stop_after is not None else expected_updates
    if args.device == "cuda":
        if args.physical_gpu_index is None or args.host_runtime_manifest is None:
            parser.error("CUDA runs require --physical-gpu-index and --host-runtime-manifest")
        _load_host_runtime(args.host_runtime_manifest, args.physical_gpu_index)
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("CUDA isolation failed: expected exactly one visible GPU")
        device = torch.device("cuda:0")
    else:
        if args.physical_gpu_index is not None or args.host_runtime_manifest is not None:
            parser.error("CPU runs may not specify GPU runtime inputs")
        device = torch.device("cpu")

    run_id = args.run_id or f"{args.scope}_wdro_{args.rho_mode}_d{args.degree}_s{args.seed}_u{expected_updates}"
    checkpoint_dir = CODEX_ROOT / "experiments" / "checkpoints" / "stage5" / run_id
    log_dir = CODEX_ROOT / "experiments" / "logs" / "stage5" / run_id
    job_dir = CODEX_ROOT / "experiments" / "logs" / "jobs" / run_id
    exact_command = shlex.join([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]])
    resume_argv = list(sys.argv[1:])
    if "--resume" not in resume_argv:
        resume_argv.append("--resume")
    resume_command = shlex.join([sys.executable, str(Path(__file__).resolve()), *resume_argv])
    monitor_command = f"tail -n 20 {shlex.quote(str(log_dir / 'events.jsonl'))}"
    host_runtime_path = str(args.host_runtime_manifest.resolve()) if args.host_runtime_manifest else None
    with JobLease(
        job_dir,
        run_id=run_id,
        exact_command=exact_command,
        resume_command=resume_command,
        monitor_command=monitor_command,
        expected_updates=expected_updates,
        physical_gpu_index=args.physical_gpu_index,
        host_runtime_manifest=host_runtime_path,
    ) as lease:
        result = train_wdro(
            wdro_config=wdro_config,
            rho_mode=args.rho_mode,
            baseline_config=baseline_config,
            protocol=protocol,
            trainer_config=trainer_config,
            degree=args.degree,
            training_seed=args.seed,
            expected_updates=expected_updates,
            stop_after=stop_after,
            device=device,
            run_id=run_id,
            checkpoint_dir=checkpoint_dir,
            log_dir=log_dir,
            resume=args.resume,
            heartbeat_callback=lambda update, status, detail: lease.update(
                update=update, status=status, detail=detail
            ),
            physical_gpu_index=args.physical_gpu_index,
            host_runtime_manifest=host_runtime_path,
            geosteiner_workers=args.geosteiner_workers,
        )
        lease.finish(
            update=int(result["completed_updates"]),
            status=str(result["status"]),
            detail={
                "latest_checkpoint": result["latest_checkpoint"],
                "best_checkpoint": result["best_checkpoint"],
                "event_log": result.get("event_log"),
            },
        )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
