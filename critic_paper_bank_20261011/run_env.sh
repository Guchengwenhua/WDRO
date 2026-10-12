#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=/remote-home/Guilin/DRO/Codex/critic_paper_bank_20261011
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH=/remote-home/Guilin/DRO/Codex/papers_wdro_20260915/.deps${PYTHONPATH:+:$PYTHONPATH}
export TMPDIR="$TASK_ROOT/artifacts/tmp"
export XDG_CACHE_HOME="$TASK_ROOT/artifacts/cache"
export TORCH_HOME="$XDG_CACHE_HOME/torch" CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton" MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
mkdir -p "$TMPDIR" "$TORCH_HOME" "$CUDA_CACHE_PATH" "$TRITON_CACHE_DIR" "$MPLCONFIGDIR"
cd "$TASK_ROOT"
exec /remote-home/Guilin/miniconda3/envs/py311/bin/python -u -B "$@"
