#!/usr/bin/env bash
set -euo pipefail
cd /remote-home/Guilin/DRO/Codex/geometry_uncertainty_20260930
mkdir -p artifacts/cache
export CUDA_VISIBLE_DEVICES="${1:-4}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export CUDA_CACHE_PATH="$PWD/artifacts/cache/cuda" TORCH_HOME="$PWD/artifacts/cache/torch" TRITON_CACHE_DIR="$PWD/artifacts/cache/triton" MPLCONFIGDIR="$PWD/artifacts/cache/matplotlib"
export TMPDIR="$PWD/artifacts/cache" XDG_CACHE_HOME="$PWD/artifacts/cache"
exec /remote-home/Guilin/miniconda3/envs/py311/bin/python -u research.py --gpu "$CUDA_VISIBLE_DEVICES"
