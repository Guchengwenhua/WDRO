#!/bin/bash
set -u
cd /remote-home/Guilin/DRO/Codex/revisions/reference_precision_v2/Codex
export WDRO_TMUX_SESSION=wdro_stage8_finish_v1
bash /remote-home/Guilin/DRO/Codex/revisions/reference_precision_v2/Codex/wdro_rest_min/scripts/stage3_env.sh /remote-home/Guilin/miniconda3/envs/py311/bin/python /remote-home/Guilin/DRO/Codex/revisions/reference_precision_v2/Codex/wdro_rest_min/scripts/stage8_finish_v1.py run > /remote-home/Guilin/DRO/Codex/revisions/reference_precision_v2/Codex/experiments/logs/stage8/recovery_v1/launches/20260910T032742523691Z_302504_1789010862523725719/output.log 2>&1
finish_exit=$?
printf '{"exit_code":%s}\n' "$finish_exit" > /remote-home/Guilin/DRO/Codex/revisions/reference_precision_v2/Codex/experiments/logs/stage8/recovery_v1/launches/20260910T032742523691Z_302504_1789010862523725719/exit.json.partial
mv /remote-home/Guilin/DRO/Codex/revisions/reference_precision_v2/Codex/experiments/logs/stage8/recovery_v1/launches/20260910T032742523691Z_302504_1789010862523725719/exit.json.partial /remote-home/Guilin/DRO/Codex/revisions/reference_precision_v2/Codex/experiments/logs/stage8/recovery_v1/launches/20260910T032742523691Z_302504_1789010862523725719/exit.json
exit "$finish_exit"
