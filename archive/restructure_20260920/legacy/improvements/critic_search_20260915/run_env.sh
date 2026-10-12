#!/bin/bash
set -euo pipefail
task_root=/remote-home/Guilin/DRO/Codex/improvements/critic_search_20260915
export TMPDIR="$task_root/artifacts/tmp" TMP="$task_root/artifacts/tmp" TEMP="$task_root/artifacts/tmp"
export XDG_CACHE_HOME="$task_root/artifacts/runtime_cache"
export TORCH_HOME="$XDG_CACHE_HOME/torch" CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda" TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton" MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 ADV_V3_REFERENCE_WORKERS=3
mkdir -p "$TMPDIR" "$TORCH_HOME" "$CUDA_CACHE_PATH" "$TRITON_CACHE_DIR" "$MPLCONFIGDIR"
cd "$task_root"
exec /remote-home/Guilin/miniconda3/envs/py311/bin/python "$@"
