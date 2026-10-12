#!/usr/bin/env bash
set -euo pipefail
BRANCH_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$BRANCH_DIR"
mkdir -p artifacts/operations
exec bash run_env.sh scheduler.py --poll-seconds 5 --admission-samples 3 \
  --max-gpu-util 35 --min-free-mib 8192 --memory-margin-mib 2048 \
  --min-disk-gib 20 --max-workers 8 --max-retries 2
