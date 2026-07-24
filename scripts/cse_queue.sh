#!/bin/bash
# Sequential CSE job queue (one lane). Feed it a queue file where each
# line is: <job-name> <python-module> <args...>
# Runs each job to completion in a Pi-held SSH session (retry+resume on
# drops via checkpoints in the CSE NFS home), then collects, uploads to
# the bucket, and removes the remote run dir (2.4GB home quota).
# Run two lanes max — the login VM has 2 shared cores and we nice -n 19.
set -uo pipefail
QUEUE=$1
cd "$(dirname "$0")/.."

.venv/bin/python setup.py -q sdist --formats=gztar >/dev/null 2>&1
scp -q dist/nca-0.1.tar.gz cse:nca-latest.tar.gz
ssh -o BatchMode=yes cse \
  "./nca-venv/bin/pip install -q --no-cache-dir --force-reinstall --no-deps ~/nca-latest.tar.gz"

while IFS= read -r line; do
  [ -z "$line" ] && continue
  case "$line" in \#*) continue;; esac
  NAME=$(echo "$line" | awk '{print $1}')
  MODULE=$(echo "$line" | awk '{print $2}')
  ARGS=$(echo "$line" | cut -d' ' -f3-)
  echo "[queue] ==== $NAME ===="
  ssh -o BatchMode=yes cse "mkdir -p nca-runs/$NAME"
  for attempt in $(seq 1 200); do
    ssh -o BatchMode=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=4 cse \
      "cd ~ && nice -n 19 ./nca-venv/bin/python -m $MODULE $ARGS --snap-dir=\$HOME/nca-runs/$NAME" \
      && break
    echo "[queue] $NAME dropped (attempt $attempt); resuming in 60s"
    sleep 60
  done
  echo "[queue] $NAME done; collecting + cleaning"
  .venv/bin/python scripts/cse_collect.py || true
  ssh -o BatchMode=yes cse "rm -rf nca-runs/$NAME"
done < "$QUEUE"
echo "[queue] lane complete: $QUEUE"
