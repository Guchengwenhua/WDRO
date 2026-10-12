#!/usr/bin/env bash
set -euo pipefail
cd /remote-home/Guilin/DRO/Codex/geometry_uncertainty_extension_20260930
gpu="${1:-7}"
if [[ ! "$gpu" =~ ^[0-9]+$ ]]; then exit 2; fi
mkdir -p artifacts
printf 'bash run_env.sh %s\n' "$gpu" > artifacts/launch_command.txt
tmux -S "$PWD/artifacts/extension.sock" new-session -d -s geometry-extension-20260930 "bash run_env.sh $gpu > artifacts/run.log 2>&1"
