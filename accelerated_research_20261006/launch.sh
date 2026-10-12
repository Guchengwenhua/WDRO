#!/usr/bin/env bash
set -euo pipefail
cd /remote-home/Guilin/DRO/Codex/accelerated_research_20261006
tmux -S "$PWD/artifacts/parallel.sock" new-session -d -s accelerated-research-20261006 "/remote-home/Guilin/miniconda3/envs/py311/bin/python -u -B controller.py >> artifacts/controller.log 2>&1"
