#!/usr/bin/env bash
set -euo pipefail
cd /remote-home/Guilin/DRO/Codex/geometry_uncertainty_20260930
mkdir -p artifacts/attack/cache
export CUDA_VISIBLE_DEVICES="${1:-4}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$PWD/artifacts/attack/cache" XDG_CACHE_HOME="$PWD/artifacts/attack/cache"
export CUDA_CACHE_PATH="$TMPDIR/cuda" TORCH_HOME="$TMPDIR/torch" TRITON_CACHE_DIR="$TMPDIR/triton" MPLCONFIGDIR="$TMPDIR/matplotlib"
exec /remote-home/Guilin/miniconda3/envs/py311/bin/python -u attack.py --gpu "$CUDA_VISIBLE_DEVICES"
