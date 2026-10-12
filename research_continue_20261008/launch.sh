#!/usr/bin/env bash
set -euo pipefail
cd /remote-home/Guilin/DRO/Codex/research_continue_20261008
tmux -S "$PWD/artifacts/parallel.sock" new-session -d -s research-continue-20261008 "/remote-home/Guilin/miniconda3/envs/py311/bin/python -u -B controller.py >> artifacts/controller.log 2>&1"
tmux -S "$PWD/artifacts/parallel.sock" new-window -t research-continue-20261008 -n correlation "/remote-home/Guilin/miniconda3/envs/py311/bin/python -u -B monitor_geometry.py >> artifacts/geometry_monitor.log 2>&1"
