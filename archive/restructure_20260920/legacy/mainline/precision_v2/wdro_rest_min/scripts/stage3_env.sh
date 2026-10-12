#!/bin/bash
set -euo pipefail
export TMPDIR=/remote-home/Guilin/DRO/Codex/wdro_precision_v2/experiments/tmp
export TMP="$TMPDIR" TEMP="$TMPDIR"
export XDG_CACHE_HOME=/remote-home/Guilin/DRO/Codex/wdro_precision_v2/experiments/cache/runtime
export TORCH_HOME="$XDG_CACHE_HOME/torch"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p "$TMPDIR" "$TORCH_HOME" "$CUDA_CACHE_PATH" "$MPLCONFIGDIR" "$TRITON_CACHE_DIR"
exec "$@"
