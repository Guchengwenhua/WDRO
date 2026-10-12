#!/usr/bin/env bash
set -euo pipefail
cd /remote-home/Guilin/DRO/Codex/checkpoint_bank_rolling_20260930
export CUDA_VISIBLE_DEVICES=5
source ./run_env.sh
exec /remote-home/Guilin/miniconda3/envs/py311/bin/python -u -B run_pipeline.py
