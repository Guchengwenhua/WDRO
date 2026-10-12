#!/usr/bin/env bash
set -euo pipefail
cd /remote-home/Guilin/DRO/Codex/checkpoint_bank_20260930
source ./run_env.sh
exec /remote-home/Guilin/miniconda3/envs/py311/bin/python -u run_pipeline.py
