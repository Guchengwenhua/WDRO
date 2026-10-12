#!/usr/bin/env bash
set -euo pipefail
cd /remote-home/Guilin/DRO/Codex/checkpoint_bank_scale_20260930
tmux -S "$PWD/artifacts/scale.sock" new-session -d -s checkpoint-bank-scale 'bash run_queue.sh >artifacts/queue.log 2>&1'
