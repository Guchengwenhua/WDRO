#!/usr/bin/env bash
set -euo pipefail
cd /remote-home/Guilin/DRO/Codex/checkpoint_bank_20260930
tmux -S "$PWD/artifacts/bank.sock" new-session -d -s checkpoint-bank-20260930 'bash run_queue.sh >artifacts/queue.log 2>&1'
