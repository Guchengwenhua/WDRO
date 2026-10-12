#!/usr/bin/env bash
set -euo pipefail
cd /remote-home/Guilin/DRO/Codex/holiday_research_20260930
mkdir -p artifacts
tmux -S "$PWD/artifacts/holiday.sock" new-session -d -s holiday-research-20260930 "/remote-home/Guilin/miniconda3/envs/py311/bin/python -u -B controller.py >> artifacts/controller.log 2>&1"
