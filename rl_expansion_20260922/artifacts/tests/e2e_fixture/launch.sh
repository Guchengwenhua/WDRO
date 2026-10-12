#!/usr/bin/env bash
set -euo pipefail
BRANCH_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$BRANCH_DIR"
exec bash run_env.sh orchestrate.py
