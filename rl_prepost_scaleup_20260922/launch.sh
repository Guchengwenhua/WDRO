#!/bin/bash
set -euo pipefail
ROOT=/remote-home/Guilin/DRO/Codex/rl_prepost_scaleup_20260922
SOCKET="$ROOT/artifacts/operations/prepost.sock"
if tmux -S "$SOCKET" has-session -t rl-prepost 2>/dev/null; then
  echo 'Existing rl-prepost session; refusing duplicate launch.'
  exit 1
fi
bash "$ROOT/run_env.sh" -c 'import sys,json; sys.path.insert(0,"/remote-home/Guilin/DRO/Codex/rl_prepost_scaleup_20260922"); import scheduler as s; import ops; ops.ensure_no_prior_workers(s.config()["tasks"],root=s.ROOT); assert ops.verify_source_freeze(s.ROOT), "Missing source freeze"'
tmux -S "$SOCKET" new-session -d -s rl-prepost -c "$ROOT" "bash $ROOT/run_env.sh $ROOT/bootstrap.py >> $ROOT/artifacts/operations/bootstrap.log 2>&1"
tmux -S "$SOCKET" list-sessions
