#!/bin/bash
set -euo pipefail
TASK_ROOT=/remote-home/Guilin/DRO/Codex/papers_wdro_20260915
cd "$TASK_ROOT"
# GPU task commands set CUDA_VISIBLE_DEVICES themselves; preparation never initializes CUDA.
if [ ! -f "$TASK_ROOT/artifacts/data/manifest.json" ]; then
  CUDA_VISIBLE_DEVICES='' bash "$TASK_ROOT/run_env.sh" prepare.py --workers 4
fi
CUDA_VISIBLE_DEVICES='' bash "$TASK_ROOT/run_env.sh" freeze.py
exec /remote-home/Guilin/miniconda3/envs/py311/bin/python "$TASK_ROOT/scheduler.py" --manifest queue.json
