#!/usr/bin/env bash
set -euo pipefail
BRANCH_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export TMPDIR="$BRANCH_DIR/artifacts/tmp"
export TMP="$TMPDIR" TEMP="$TMPDIR"
export XDG_CACHE_HOME="$BRANCH_DIR/artifacts/cache"
export TORCH_HOME="$XDG_CACHE_HOME/torch"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${ORACLE_FREE_CPU_THREADS:-3}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p "$TMPDIR" "$TORCH_HOME" "$CUDA_CACHE_PATH" "$MPLCONFIGDIR" "$TRITON_CACHE_DIR"
exec /remote-home/Guilin/miniconda3/envs/py311/bin/python "$@"
