#!/bin/bash
set -u
cd /remote-home/Guilin/DRO/Codex/revisions/reference_precision_v2/Codex
export WDRO_TMUX_SESSION=wdro_stage7_evaluation_v1
bash /remote-home/Guilin/DRO/Codex/revisions/reference_precision_v2/Codex/wdro_rest_min/scripts/stage3_env.sh /remote-home/Guilin/miniconda3/envs/py311/bin/python /remote-home/Guilin/DRO/Codex/revisions/reference_precision_v2/Codex/wdro_rest_min/scripts/stage7.py pipeline > /remote-home/Guilin/DRO/Codex/revisions/reference_precision_v2/Codex/experiments/logs/stage7/launches/20260909T095205366995Z_245803_1788947525367023421/pipeline.log 2>&1
stage7_exit=$?
printf '{"exit_code":%s}\n' "$stage7_exit" > /remote-home/Guilin/DRO/Codex/revisions/reference_precision_v2/Codex/experiments/logs/stage7/launches/20260909T095205366995Z_245803_1788947525367023421/exit.json.partial
mv /remote-home/Guilin/DRO/Codex/revisions/reference_precision_v2/Codex/experiments/logs/stage7/launches/20260909T095205366995Z_245803_1788947525367023421/exit.json.partial /remote-home/Guilin/DRO/Codex/revisions/reference_precision_v2/Codex/experiments/logs/stage7/launches/20260909T095205366995Z_245803_1788947525367023421/exit.json
exit "$stage7_exit"
