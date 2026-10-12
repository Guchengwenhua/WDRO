#!/usr/bin/env bash
set -uo pipefail
BRANCH_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
for attempt in 1 2 3; do
  bash "$BRANCH_DIR/run_env.sh" "$BRANCH_DIR/pipeline.py" >> "$BRANCH_DIR/artifacts/logs/pipeline.log" 2>&1
  code=$?
  if [ "$code" -eq 0 ]; then exit 0; fi
  echo "Controller exited $code, recovery attempt $attempt" >> "$BRANCH_DIR/artifacts/logs/pipeline.log"
  sleep 30
done
exit 1
