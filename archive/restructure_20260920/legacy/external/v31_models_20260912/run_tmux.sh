#!/usr/bin/env bash
set -euo pipefail
BASE=/remote-home/Guilin/DRO/Codex/benchmarks/v31_models_20260912
SOCKET=/remote-home/Guilin/DRO/Codex/shared/experiments/real_v31.sock
SESSION=real-v31-20260912
mkdir -p "$BASE/artifacts/logs"
if tmux -S "$SOCKET" has-session -t "$SESSION" 2>/dev/null; then
  echo "Existing session retained: $SESSION"
  exit 0
fi
export WDRO_TMUX_SESSION="$SESSION"
tmux -S "$SOCKET" new-session -d -s "$SESSION" -c "$BASE" \
  'exec bash /remote-home/Guilin/DRO/Codex/wdro_precision_v2/wdro_rest_min/scripts/stage3_env.sh /remote-home/Guilin/miniconda3/envs/py311/bin/python -u /remote-home/Guilin/DRO/Codex/benchmarks/v31_models_20260912/pipeline.py >> /remote-home/Guilin/DRO/Codex/benchmarks/v31_models_20260912/artifacts/logs/pipeline.log 2>&1'
echo "Started $SESSION; status: $BASE/artifacts/STATUS.json"
