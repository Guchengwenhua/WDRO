#!/usr/bin/env bash
set -euo pipefail
cd /remote-home/Guilin/DRO/Codex/geometry_uncertainty_20260930
gpu="${1:-4}"
if [[ ! "$gpu" =~ ^[0-9]+$ ]]; then
  printf '%s\n' 'GPU must be a non-negative integer' >&2
  exit 2
fi
mkdir -p artifacts/attack
printf 'bash run_attack.sh %s\n' "$gpu" > artifacts/attack/launch_command.txt
tmux -S "$PWD/artifacts/attack.sock" new-session -d -s geom-attack-20260930 "bash run_attack.sh $gpu > artifacts/attack/run.log 2>&1"
