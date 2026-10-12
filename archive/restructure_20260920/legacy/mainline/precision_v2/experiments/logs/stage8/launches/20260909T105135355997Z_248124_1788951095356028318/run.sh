#!/bin/bash
set -u
cd /remote-home/Guilin/DRO/Codex/revisions/reference_precision_v2/Codex
export WDRO_TMUX_SESSION=wdro_stage8_cem256_v1
bash /remote-home/Guilin/DRO/Codex/revisions/reference_precision_v2/Codex/wdro_rest_min/scripts/stage3_env.sh /remote-home/Guilin/miniconda3/envs/py311/bin/python /remote-home/Guilin/DRO/Codex/revisions/reference_precision_v2/Codex/wdro_rest_min/scripts/stage8.py pipeline > /remote-home/Guilin/DRO/Codex/revisions/reference_precision_v2/Codex/experiments/logs/stage8/launches/20260909T105135355997Z_248124_1788951095356028318/pipeline.log 2>&1
stage8_exit=$?
printf '{"exit_code":%s}\n' "$stage8_exit" > /remote-home/Guilin/DRO/Codex/revisions/reference_precision_v2/Codex/experiments/logs/stage8/launches/20260909T105135355997Z_248124_1788951095356028318/exit.json.partial
mv /remote-home/Guilin/DRO/Codex/revisions/reference_precision_v2/Codex/experiments/logs/stage8/launches/20260909T105135355997Z_248124_1788951095356028318/exit.json.partial /remote-home/Guilin/DRO/Codex/revisions/reference_precision_v2/Codex/experiments/logs/stage8/launches/20260909T105135355997Z_248124_1788951095356028318/exit.json
exit "$stage8_exit"
