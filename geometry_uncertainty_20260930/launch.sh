#!/usr/bin/env bash
set -euo pipefail
cd /remote-home/Guilin/DRO/Codex/geometry_uncertainty_20260930
mkdir -p artifacts
printf '%s\n' 'bash run_env.sh 4' > artifacts/launch_command.txt
tmux -S "$PWD/artifacts/geometry.sock" new-session -d -s geom-uncertainty-20260930 'bash run_env.sh 4 > artifacts/run.log 2>&1'
