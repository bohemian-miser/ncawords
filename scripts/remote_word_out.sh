#!/bin/bash
# Train a word model on the fleet, writing to an explicit output file.
# Session held open on purpose (the hosts kill detached jobs).
#
# Usage: scripts/remote_word_out.sh <TEXT> <STEPS> <BATCH> <OUT_BASENAME>
set -u
TXT="$1"; STEPS="$2"; BATCH="$3"; OUT="$4"

ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=30 cse bash -s <<EOF 2>&1
set -e
cd ~/projects/ncawords
mkdir -p snaps logs weights
echo "node=\$(hostname) cores=\$(nproc)"
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 python3 -m nca.train_word \
    --text $TXT --steps $STEPS --batch $BATCH --snap-dir snaps \
    --out weights/$OUT
OMP_NUM_THREADS=1 nice -n 10 python3 -m nca.ocr_word weights/$OUT \
    --img-dir grown || echo "OCR-FAIL $TXT"
OMP_NUM_THREADS=1 nice -n 10 python3 -m nca.regen_test weights/$OUT \
    --out grown/regen_$TXT.png || echo "REGEN-FAIL $TXT"
EOF
