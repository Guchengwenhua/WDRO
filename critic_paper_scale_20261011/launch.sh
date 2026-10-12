#!/usr/bin/env bash
set -euo pipefail
cd /remote-home/Guilin/DRO/Codex/critic_paper_scale_20261011
if tmux -S "$PWD/artifacts/queue.sock" has-session -t critic-paper-scale-20261011 2>/dev/null; then
  echo 'Existing scale experiment session is running.'
  exit 0
fi
tmux -S "$PWD/artifacts/queue.sock" new-session -d -s critic-paper-scale-20261011 "bash run_env.sh controller.py >> artifacts/controller.log 2>&1"
