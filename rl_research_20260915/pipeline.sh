#!/bin/bash
set -u
cd /remote-home/Guilin/DRO/Codex/rl_research_20260915
bash run_env.sh eta.py --watch &
eta_pid=$!
bash run_env.sh scheduler.py
queue_exit=$?
bash run_env.sh closeout.py --queue-exit "$queue_exit"
closeout_exit=$?
wait "$eta_pid"
if [ "$queue_exit" -ne 0 ]; then exit "$queue_exit"; fi
exit "$closeout_exit"
