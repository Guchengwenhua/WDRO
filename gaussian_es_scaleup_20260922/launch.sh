#!/usr/bin/env bash
set -euo pipefail
BRANCH_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOCKET="$BRANCH_DIR/artifacts/tmux.sock"
SESSION=gaussian_es_scaleup
if tmux -S "$SOCKET" has-session -t "$SESSION" 2>/dev/null; then
  echo "Existing session: tmux -S $SOCKET attach -t $SESSION"
  exit 0
fi
test -f "$BRANCH_DIR/artifacts/tests/gpu_gate.json"
test -f "$BRANCH_DIR/artifacts/manifests/source_freeze.json"
tmux -S "$SOCKET" new-session -d -s "$SESSION" -c "$BRANCH_DIR" "bash '$BRANCH_DIR/run_session.sh'"
echo "Started: tmux -S $SOCKET attach -t $SESSION"
