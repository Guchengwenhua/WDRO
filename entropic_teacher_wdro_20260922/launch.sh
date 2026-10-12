#!/bin/bash
set -euo pipefail
ROOT=/remote-home/Guilin/DRO/Codex/entropic_teacher_wdro_20260922
SOCKET="$ROOT/artifacts/operations/et.sock"
if tmux -S "$SOCKET" has-session -t et-wdro 2>/dev/null; then
  echo 'Existing et-wdro session; refusing duplicate launch.'
  exit 0
fi
test -f "$ROOT/artifacts/operations/start_gate.json"
bash "$ROOT/run_env.sh" -c 'import json; from engine import ART,binding; g=json.loads((ART/"operations/start_gate.json").read_text()); assert g["status"]=="PASS"; assert g["binding"]==binding()'
tmux -S "$SOCKET" new-session -d -s et-wdro "bash '$ROOT/run_env.sh' '$ROOT/pipeline.py' >> '$ROOT/artifacts/operations/controller.log' 2>&1"
echo "Started et-wdro at $SOCKET"
