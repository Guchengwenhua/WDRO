#!/bin/bash
set -u
cd /remote-home/Guilin/DRO/Codex/revisions/reference_precision_v2/Codex
export WDRO_TMUX_SESSION=wdro_stage8_supporting_v1
bash /remote-home/Guilin/DRO/Codex/revisions/reference_precision_v2/Codex/wdro_rest_min/scripts/stage3_env.sh /remote-home/Guilin/miniconda3/envs/py311/bin/python /remote-home/Guilin/DRO/Codex/revisions/reference_precision_v2/Codex/wdro_rest_min/scripts/stage8.py supporting > /remote-home/Guilin/DRO/Codex/revisions/reference_precision_v2/Codex/experiments/logs/stage8/supporting_launch/20260909T103818172337Z_247705_1788950298172394952/output.log 2>&1
supporting_exit=$?
printf '{"exit_code":%s}\n' "$supporting_exit" > /remote-home/Guilin/DRO/Codex/revisions/reference_precision_v2/Codex/experiments/logs/stage8/supporting_launch/20260909T103818172337Z_247705_1788950298172394952/exit.json
exit "$supporting_exit"
