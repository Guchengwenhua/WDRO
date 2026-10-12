#!/bin/bash
set -uo pipefail
ACCEPT_ROOT=/remote-home/Guilin/DRO/Codex/rl_acceptance_20260926
EXP_ROOT=/remote-home/Guilin/DRO/Codex/rl_prepost_scaleup_20260922
exec 9>"$ACCEPT_ROOT/interim.lock"
flock -n 9 || exit 2
python3 - <<'PY'
import json,time
from pathlib import Path
r=Path('/remote-home/Guilin/DRO/Codex/rl_acceptance_20260926')
while True:
 d=json.loads((r/'storage_migration.json').read_text())
 if d['status']=='PASS':break
 if not Path('/proc/1197356').exists():raise RuntimeError('Migration process exited before PASS; inspect storage_migration.log')
 time.sleep(10)
PY
if [ "$?" -ne 0 ]; then exit 3; fi
nice -n 10 bash "$EXP_ROOT/run_env.sh" "$ACCEPT_ROOT/review.py" old > "$ACCEPT_ROOT/old_review.log" 2>&1 &
old_review_pid=$!
nice -n 10 bash "$EXP_ROOT/run_env.sh" "$ACCEPT_ROOT/review.py" new > "$ACCEPT_ROOT/new_review.log" 2>&1 &
new_review_pid=$!
wait "$old_review_pid"
old_review_exit=$?
wait "$new_review_pid"
new_review_exit=$?
printf 'old_exit=%s new_exit=%s\n' "$old_review_exit" "$new_review_exit" > "$ACCEPT_ROOT/interim_exit.txt"
bash "$EXP_ROOT/run_env.sh" "$ACCEPT_ROOT/summarize.py"
