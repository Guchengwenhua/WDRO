#!/bin/bash
set -euo pipefail
ROOT=/remote-home/Guilin/DRO/Codex/rl_scaleup_20260921
SOCKET="$ROOT/artifacts/operations/scaleup.sock"
if tmux -S "$SOCKET" has-session -t rl-scaleup 2>/dev/null; then
  echo 'rl-scaleup is already running; refusing duplicate launch.'
  exit 1
fi
bash "$ROOT/run_env.sh" -c 'import sys; sys.path.insert(0,"/remote-home/Guilin/DRO/Codex/rl_scaleup_20260921"); import scheduler as s; import json; p=json.loads((s.ROOT/"protocol.json").read_text()); s.ensure_no_prior_workers(s.tasks(p)); assert s.verify_source_freeze(), "source freeze required"'
tmux -S "$SOCKET" new-session -d -s rl-scaleup -c "$ROOT" "bash $ROOT/run_session.sh"
tmux -S "$SOCKET" list-sessions
