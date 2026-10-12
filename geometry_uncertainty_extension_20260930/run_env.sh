#!/usr/bin/env bash
set -euo pipefail
cd /remote-home/Guilin/DRO/Codex/geometry_uncertainty_extension_20260930
mkdir -p artifacts/cache
export CUDA_VISIBLE_DEVICES="${1:-7}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$PWD/artifacts/cache" XDG_CACHE_HOME="$PWD/artifacts/cache"
export CUDA_CACHE_PATH="$TMPDIR/cuda" TORCH_HOME="$TMPDIR/torch" TRITON_CACHE_DIR="$TMPDIR/triton" MPLCONFIGDIR="$TMPDIR/matplotlib"
exec /remote-home/Guilin/miniconda3/envs/py311/bin/python -u extension.py --gpu "$CUDA_VISIBLE_DEVICES"
