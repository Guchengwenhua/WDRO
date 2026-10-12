#!/usr/bin/env bash
set -euo pipefail
BASE=/remote-home/Guilin/DRO/Codex/external_benchmarks/20260912_v31_models
SOCKET=/remote-home/Guilin/DRO/Codex/experiments/real_v31.sock
SESSION=real-v31-20260912
mkdir -p "$BASE/artifacts/logs"
if tmux -S "$SOCKET" has-session -t "$SESSION" 2>/dev/null; then
  echo "Existing session retained: $SESSION"
  exit 0
fi
export WDRO_TMUX_SESSION="$SESSION"
tmux -S "$SOCKET" new-session -d -s "$SESSION" -c "$BASE" \
  'exec bash /remote-home/Guilin/DRO/Codex/revisions/reference_precision_v2/Codex/wdro_rest_min/scripts/stage3_env.sh /remote-home/Guilin/miniconda3/envs/py311/bin/python -u /remote-home/Guilin/DRO/Codex/external_benchmarks/20260912_v31_models/pipeline.py >> /remote-home/Guilin/DRO/Codex/external_benchmarks/20260912_v31_models/artifacts/logs/pipeline.log 2>&1'
echo "Started $SESSION; status: $BASE/artifacts/STATUS.json"
