#!/bin/bash
# Train ONE letter on the CSE fleet. Each ssh session lands on some login
# node and gets its own 2-core affinity, so running several of these
# concurrently is how we get parallelism. Runs niced (shared machines).
#
# Usage: scripts/remote_letter.sh <CHAR> [STEPS]
set -u
CH="$1"
STEPS="${2:-2000}"
HEX=$(printf '%04x' "'$CH")

ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 cse bash -s <<EOF 2>&1 | sed "s/^/[\$CH] /"
set -e
cd ~/projects/ncawords
echo "node=\$(hostname) cores=\$(nproc)"
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 python3 -m nca.train \
    --char $CH --steps $STEPS --out weights/$HEX.json
OMP_NUM_THREADS=1 nice -n 10 python3 -m nca.ocr_eval weights/$HEX.json \
    --report ocr/$HEX.json --img-dir grown || true
EOF
