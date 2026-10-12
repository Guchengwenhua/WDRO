#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PROJ_ROOT="$(cd "$ROOT/.." && pwd)"
BENCH_DIR="$PROJ_ROOT/benchmark"
OUT_ROOT="/remote-home/Chenlin/OPT/DRO/benchmark"
MAX_DEGREE="${1:-50}"
NORMALIZE="${2:-1}"
LOGDIR="$ROOT/bench_dumps_logs"
mkdir -p "$LOGDIR" "$OUT_ROOT"

# locate route binary
ROUTE_BIN="$PROJ_ROOT/build/route"
if [[ ! -x "$ROUTE_BIN" ]]; then
  if [[ -x "$PROJ_ROOT/route" ]]; then
    ROUTE_BIN="$PROJ_ROOT/route"
  else
    ROUTE_BIN="$(which route 2>/dev/null || true)"
  fi
fi
if [[ -z "$ROUTE_BIN" || ! -x "$ROUTE_BIN" ]]; then
  echo "Error: route executable not found. Build it first (cmake --build build)." >&2
  exit 1
fi

cd "$PROJ_ROOT"

for ds in "$BENCH_DIR"/*/; do
  [ -d "$ds" ] || continue
  dsname=$(basename "$ds")
  echo "=== Processing $dsname ==="
  lef=$(find "$ds" -type f \( -iname "*.lef" \) -print -quit || true)
  def=$(find "$ds" -type f \( -iname "*.def" \) -print -quit || true)
  echo "  ds=$dsname lef=$lef def=$def" | tee -a "$LOGDIR/$dsname.log"
  if [[ -z "$lef" || -z "$def" ]]; then
    echo "  skip: missing lef/def" | tee -a "$LOGDIR/$dsname.log"
    continue
  fi
  START=$(date +%s)
  "$ROUTE_BIN" --lef "$lef" --def "$def" --out_root "$OUT_ROOT" --max_degree "$MAX_DEGREE" --normalize "$NORMALIZE" >"$LOGDIR/$dsname.out" 2>"$LOGDIR/$dsname.err"
  RET=$?
  END=$(date +%s)
  echo "  done (ret=$RET, time=$((END-START))s)" | tee -a "$LOGDIR/$dsname.log"
done

echo "All done. Logs: $LOGDIR"