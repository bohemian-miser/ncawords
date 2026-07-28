#!/bin/bash
# Run a training job on CSE (CPU) with the Pi holding the SSH session.
#   scripts/cse_run.sh <job-name> <python-module> [args...]
#
# CSE login VMs kill user processes (and tmux) on logout and give each
# session a private /tmp, so the session MUST stay open for the job's
# lifetime — run this under a background task on the Pi. The venv and
# run outputs live in the NFS home (shared across all vx VMs), and
# checkpoints make dropped sessions resumable: this script retries the
# ssh until the trainer exits cleanly, then rsyncs results back.
# Etiquette: CPU-only, nice -n 19, one or two jobs at a time.
set -uo pipefail
NAME=$1; MODULE=$2; shift 2
RDIR="nca-runs/$NAME"
cd "$(dirname "$0")/.."

.venv/bin/python setup.py -q sdist --formats=gztar >/dev/null 2>&1
scp -q dist/nca-0.1.tar.gz cse:nca-latest.tar.gz
ssh -o BatchMode=yes cse \
  "./nca-venv/bin/pip install -q --no-cache-dir --force-reinstall --no-deps ~/nca-latest.tar.gz && mkdir -p $RDIR"

for attempt in $(seq 1 20); do
  echo "[cse_run] $NAME attempt $attempt"
  ssh -o BatchMode=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=4 cse \
    "cd ~ && nice -n 19 ./nca-venv/bin/python -m $MODULE $* --snap-dir=\$HOME/$RDIR" \
    && break
  echo "[cse_run] $NAME dropped (attempt $attempt); resuming in 30s"
  sleep 30
done

echo "[cse_run] $NAME finished; collecting"
.venv/bin/python scripts/cse_collect.py
