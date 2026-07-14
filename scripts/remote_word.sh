#!/bin/bash
# Train ONE whole-word model on the CSE fleet (own node, 2 cores, niced).
# Usage: scripts/remote_word.sh <TEXT> [STEPS] [BATCH]
set -u
TXT="$1"
STEPS="${2:-2500}"
BATCH="${3:-6}"

ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 cse bash -s <<EOF 2>&1 | sed "s/^/[\$TXT] /"
set -e
cd ~/projects/ncawords
echo "node=\$(hostname) cores=\$(nproc)"
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 python3 -m nca.train_word \
    --text $TXT --steps $STEPS --batch $BATCH --out weights/word_$TXT.json
OMP_NUM_THREADS=1 nice -n 10 python3 -m nca.ocr_word weights/word_$TXT.json \
    --img-dir grown || echo "OCR-FAIL $TXT"
EOF
