#!/bin/bash
set -uo pipefail
ROOT=/remote-home/Guilin/DRO/Codex/rl_scaleup_20260921
cd "$ROOT"
bash "$ROOT/run_env.sh" "$ROOT/eta.py" --watch > "$ROOT/artifacts/operations/eta.log" 2>&1 &
eta_pid=$!
bash "$ROOT/run_env.sh" "$ROOT/scheduler.py" >> "$ROOT/artifacts/operations/scheduler.log" 2>&1
queue_exit=$?
bash "$ROOT/run_env.sh" "$ROOT/closeout.py" --queue-exit "$queue_exit" > "$ROOT/artifacts/operations/closeout.log" 2>&1
closeout_exit=$?
kill "$eta_pid" 2>/dev/null || true
wait "$eta_pid" 2>/dev/null || true
bash "$ROOT/run_env.sh" "$ROOT/eta.py" >> "$ROOT/artifacts/operations/eta.log" 2>&1
exit "$closeout_exit"
