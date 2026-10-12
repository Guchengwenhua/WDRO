#!/usr/bin/env bash
set -euo pipefail
cd /remote-home/Guilin/DRO/Codex/critic_paper_bank_20261011
if tmux -S "$PWD/artifacts/queue.sock" has-session -t critic-paper-bank-20261011 2>/dev/null; then
  echo 'Existing experiment session is still running.'
  exit 0
fi
tmux -S "$PWD/artifacts/queue.sock" new-session -d -s critic-paper-bank-20261011 "bash run_env.sh controller.py >> artifacts/controller.log 2>&1"
