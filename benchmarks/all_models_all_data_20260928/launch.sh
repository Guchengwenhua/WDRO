#!/bin/bash
set -euo pipefail
TASK_ROOT=/remote-home/Guilin/DRO/Codex/benchmarks/all_models_all_data_20260928
TASK_SOCKET="$TASK_ROOT/artifacts/evaluation.sock"
mkdir -p "$TASK_ROOT/artifacts/logs"
if tmux -S "$TASK_SOCKET" has-session -t dro-all-data 2>/dev/null; then
  printf 'Existing tmux session: dro-all-data\n'
  exit 0
fi
tmux -S "$TASK_SOCKET" new-session -d -s dro-all-data -c "$TASK_ROOT" "bash '$TASK_ROOT/run_env.sh' '$TASK_ROOT/pipeline.py' >> '$TASK_ROOT/artifacts/logs/pipeline.log' 2>&1"
printf 'Started tmux session: dro-all-data\n'
