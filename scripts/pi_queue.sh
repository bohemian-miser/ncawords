#!/bin/bash
# Local Pi lane: same queue-file format as cse_queue.sh, runs jobs on the
# Pi itself (nice -n 15, below interactive work). Snapshots go straight
# into ~/cse_runs/<name>/ — the same dir cse_collect.py uploads from, so
# the existing 2-hourly collection cron publishes these runs unchanged.
set -uo pipefail
QUEUE=$1
cd "$(dirname "$0")/.."

CODE_SHA=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)
export NCA_CODE_SHA=$CODE_SHA
while IFS= read -r line; do
  [ -z "$line" ] && continue
  case "$line" in \#*) continue;; esac
  NAME=$(echo "$line" | awk '{print $1}')
  MODULE=$(echo "$line" | awk '{print $2}')
  ARGS=$(echo "$line" | cut -d' ' -f3-)
  echo "[pi-lane] ==== $NAME ===="
  mkdir -p "$HOME/cse_runs/$NAME"
  tar czf "$HOME/cse_runs/$NAME/code.tgz" nca/ 2>/dev/null || true
  nice -n 15 .venv/bin/python -m $MODULE $ARGS --snap-dir="$HOME/cse_runs/$NAME" \
    || echo "[pi-lane] $NAME exited nonzero"
done < "$QUEUE"
echo "[pi-lane] complete: $QUEUE"
