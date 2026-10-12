#!/usr/bin/env bash
set -euo pipefail
BRANCH_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export TMPDIR="$BRANCH_DIR/artifacts/tmp" TMP="$BRANCH_DIR/artifacts/tmp" TEMP="$BRANCH_DIR/artifacts/tmp"
export XDG_CACHE_HOME="$BRANCH_DIR/artifacts/cache" TORCH_HOME="$BRANCH_DIR/artifacts/cache/torch"
export CUDA_CACHE_PATH="$BRANCH_DIR/artifacts/cache/cuda" MPLCONFIGDIR="$BRANCH_DIR/artifacts/cache/matplotlib"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p "$TMPDIR" "$TORCH_HOME" "$CUDA_CACHE_PATH" "$MPLCONFIGDIR"
cd "$BRANCH_DIR"
exec /remote-home/Guilin/miniconda3/envs/py311/bin/python "$@"
