#!/bin/bash
# Sequential CSE job queue (one lane). Feed it a queue file where each
# line is: <job-name> <python-module> <args...>
# Runs each job to completion in a Pi-held SSH session (retry+resume on
# drops via checkpoints in the CSE NFS home), then collects, uploads to
# the bucket, and removes the remote run dir (2.4GB home quota).
# Run two lanes max — the login VM has 2 shared cores and we nice -n 19.
set -uo pipefail
QUEUE=$1
HOST="${2:-cse}"
cd "$(dirname "$0")/.."

# Ship the source tree and run via PYTHONPATH — pip installs into the
# shared NFS venv race between lanes and silently keep stale files.
rsync -az --delete nca/ "$HOST":nca-src/nca/

mapfile -t QLINES < "$QUEUE"
for line in "${QLINES[@]}"; do
  [ -z "$line" ] && continue
  case "$line" in \#*) continue;; esac
  NAME=$(echo "$line" | awk '{print $1}')
  MODULE=$(echo "$line" | awk '{print $2}')
  ARGS=$(echo "$line" | cut -d' ' -f3-)
  echo "[queue] ==== $NAME ===="
  ssh -n -o BatchMode=yes "$HOST" "mkdir -p nca-runs/$NAME"
  for attempt in $(seq 1 200); do
    ssh -n -o BatchMode=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=4 "$HOST" \
      "cd ~ && PYTHONPATH=\$HOME/nca-src nice -n 19 ./nca-venv/bin/python -m $MODULE $ARGS --snap-dir=\$HOME/nca-runs/$NAME" \
      && break
    echo "[queue] $NAME dropped (attempt $attempt); resuming in 60s"
    sleep 60
  done
  echo "[queue] $NAME done; collecting + cleaning"
  .venv/bin/python scripts/cse_collect.py < /dev/null || true
  ssh -n -o BatchMode=yes "$HOST" "rm -rf nca-runs/$NAME"
done
echo "[queue] lane complete: $QUEUE"
