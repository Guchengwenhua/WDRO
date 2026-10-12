#!/usr/bin/env bash
set -euo pipefail
cd /remote-home/Guilin/DRO/Codex/net_structure_real_20260922
export TMPDIR="$PWD/artifacts/tmp" XDG_CACHE_HOME="$PWD/artifacts/cache" MPLCONFIGDIR="$PWD/artifacts/matplotlib"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"
exec "$@"
