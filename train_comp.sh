#!/bin/bash
set -u
cd "$(dirname "$0")" || exit
export OMP_NUM_THREADS=32
source venv/bin/activate

mkdir -p logs weights

echo "Training C, O, M, P in parallel..."
python -m nca.train --char C --steps 2000 --out weights/0043.json > logs/0043.log 2>&1 &
P1=$!
python -m nca.train --char O --steps 2000 --out weights/004f.json > logs/004f.log 2>&1 &
P2=$!
python -m nca.train --char M --steps 2000 --out weights/004d.json > logs/004d.log 2>&1 &
P3=$!
python -m nca.train --char P --steps 2000 --out weights/0050.json > logs/0050.log 2>&1 &
P4=$!

wait $P1 $P2 $P3 $P4
echo "All 4 letter models finished training!"
