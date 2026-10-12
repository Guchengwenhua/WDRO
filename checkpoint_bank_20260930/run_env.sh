#!/usr/bin/env bash
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export XDG_CACHE_HOME=/remote-home/Guilin/DRO/Codex/checkpoint_bank_20260930/artifacts/cache/xdg
export TORCH_HOME=/remote-home/Guilin/DRO/Codex/checkpoint_bank_20260930/artifacts/cache/torch
export CUDA_CACHE_PATH=/remote-home/Guilin/DRO/Codex/checkpoint_bank_20260930/artifacts/cache/cuda
export TRITON_CACHE_DIR=/remote-home/Guilin/DRO/Codex/checkpoint_bank_20260930/artifacts/cache/triton
export TMPDIR=/remote-home/Guilin/DRO/Codex/checkpoint_bank_20260930/artifacts/tmp
mkdir -p "$XDG_CACHE_HOME" "$TORCH_HOME" "$CUDA_CACHE_PATH" "$TRITON_CACHE_DIR" "$TMPDIR"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
