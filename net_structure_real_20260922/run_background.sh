#!/usr/bin/env bash
set -uo pipefail
cd /remote-home/Guilin/DRO/Codex/net_structure_real_20260922
for attempt in 1 2 3; do
    bash ./run_env.sh /remote-home/Guilin/miniconda3/envs/py311/bin/python supervise.py >> artifacts/logs/supervisor.log 2>&1
    rc=$?
    if [ "$rc" -eq 0 ]; then exit 0; fi
    sleep 15
done
exit "$rc"
