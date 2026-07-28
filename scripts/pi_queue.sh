#!/bin/bash
# Local lane: same queue-file format as cse_queue.sh, runs jobs on this
# machine (nice -n 15, below interactive work). Snapshots go straight
# into <local_runs_dir>/<name>/ (fleet.config.json) — the same dir
# cse_collect.py uploads from, so the periodic collection publishes these
# runs unchanged.
set -uo pipefail
QUEUE=$1
cd "$(dirname "$0")/.."
RUNS_DIR=$(python3 -c "import sys; sys.path.insert(0, '.'); \
from nca import fleetconfig; print(fleetconfig.load()['local_runs_dir'])")

CODE_SHA=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)
export NCA_CODE_SHA=$CODE_SHA
while IFS= read -r line; do
  [ -z "$line" ] && continue
  case "$line" in \#*) continue;; esac
  NAME=$(echo "$line" | awk '{print $1}')
  MODULE=$(echo "$line" | awk '{print $2}')
  ARGS=$(echo "$line" | cut -d' ' -f3-)
  echo "[pi-lane] ==== $NAME ===="
  mkdir -p "$RUNS_DIR/$NAME"
  tar czf "$RUNS_DIR/$NAME/code.tgz" nca/ 2>/dev/null || true
  nice -n 15 .venv/bin/python -m $MODULE $ARGS --snap-dir="$RUNS_DIR/$NAME" \
    || echo "[pi-lane] $NAME exited nonzero"
done < "$QUEUE"
echo "[pi-lane] complete: $QUEUE"
