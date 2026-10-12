#!/bin/bash
set -euo pipefail
TASK_ROOT=/remote-home/Guilin/DRO/Codex/benchmarks/all_models_all_data_20260928
cd "$TASK_ROOT"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR="$TASK_ROOT/artifacts/tmp"
export TMP="$TMPDIR" TEMP="$TMPDIR"
export XDG_CACHE_HOME="$TASK_ROOT/artifacts/cache"
export TORCH_HOME="$XDG_CACHE_HOME/torch" CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib" TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p "$TMPDIR" "$TORCH_HOME" "$CUDA_CACHE_PATH" "$MPLCONFIGDIR" "$TRITON_CACHE_DIR" "$TASK_ROOT/artifacts/logs"
exec /remote-home/Guilin/miniconda3/envs/py311/bin/python -u "$@"
