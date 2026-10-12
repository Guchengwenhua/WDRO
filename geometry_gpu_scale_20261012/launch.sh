#!/usr/bin/env bash
set -euo pipefail
cd /remote-home/Guilin/DRO/Codex/geometry_gpu_scale_20261012
tmux -S "$PWD/artifacts/parallel.sock" new-session -d -s geometry-gpu-scale-20261012 "/remote-home/Guilin/miniconda3/envs/py311/bin/python -u -B controller.py >> artifacts/controller.log 2>&1"
tmux -S "$PWD/artifacts/parallel.sock" new-window -t geometry-gpu-scale-20261012 -n correlation "/remote-home/Guilin/miniconda3/envs/py311/bin/python -u -B monitor_geometry.py >> artifacts/geometry_monitor.log 2>&1"
