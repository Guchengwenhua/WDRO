#!/bin/bash
set -euo pipefail
ROOT=/remote-home/Guilin/DRO/Codex/rl_research_20260915
export TMPDIR="$ROOT/artifacts/tmp" TMP="$ROOT/artifacts/tmp" TEMP="$ROOT/artifacts/tmp"
export XDG_CACHE_HOME="$ROOT/artifacts/cache" TORCH_HOME="$ROOT/artifacts/cache/torch"
export CUDA_CACHE_PATH="$ROOT/artifacts/cache/cuda" MPLCONFIGDIR="$ROOT/artifacts/cache/matplotlib"
export TRITON_CACHE_DIR="$ROOT/artifacts/cache/triton" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p "$TMPDIR" "$TORCH_HOME" "$CUDA_CACHE_PATH" "$MPLCONFIGDIR" "$TRITON_CACHE_DIR"
exec /remote-home/Guilin/miniconda3/envs/py311/bin/python "$@"
