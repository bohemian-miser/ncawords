#!/bin/bash
# Launch a word model on the fleet DETACHED (survives ssh teardown), with
# periodic growth snapshots written to snaps/ on the remote host.
#
# Usage: scripts/remote_word_bg.sh <TEXT> [STEPS] [BATCH]
set -u
TXT="$1"
STEPS="${2:-4000}"
BATCH="${3:-4}"

ssh -o StrictHostKeyChecking=no cse bash -s <<EOF
set -e
cd ~/projects/ncawords
mkdir -p snaps logs
nohup env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 \
  python3 -m nca.train_word --text $TXT --steps $STEPS --batch $BATCH \
    --snap-dir snaps --out weights/word_$TXT.json \
  > logs/word_$TXT.log 2>&1 &
sleep 1
echo "launched $TXT on \$(hostname), pid \$(pgrep -f "train_word --text $TXT" | head -1)"
EOF
