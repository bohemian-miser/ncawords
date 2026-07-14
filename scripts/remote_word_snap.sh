#!/bin/bash
# Train a word model on the fleet with periodic growth snapshots.
#
# The ssh session is held OPEN for the life of the job on purpose: the hosts
# kill user processes when the session ends (systemd KillUserProcesses), so a
# nohup/detached job dies instantly. Run this script itself in the background.
#
# Usage: scripts/remote_word_snap.sh <TEXT> [STEPS] [BATCH]
set -u
TXT="$1"
STEPS="${2:-4000}"
BATCH="${3:-4}"

ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=30 cse bash -s <<EOF 2>&1
set -e
cd ~/projects/ncawords
mkdir -p snaps logs
echo "node=\$(hostname) cores=\$(nproc)"
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 python3 -m nca.train_word \
    --text $TXT --steps $STEPS --batch $BATCH --snap-dir snaps \
    --out weights/word_$TXT.json
OMP_NUM_THREADS=1 nice -n 10 python3 -m nca.ocr_word weights/word_$TXT.json \
    --img-dir grown || echo "OCR-FAIL $TXT"
EOF
