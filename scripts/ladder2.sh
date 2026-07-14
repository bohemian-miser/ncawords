#!/bin/bash
# Ladder v2: warm-restart G/R with LR held high until 70%, then the
# original escalation (double "GO" -> singles O,W -> word "GROW").
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export OMP_NUM_THREADS=4

log() { echo "[ladder2 $(date +%H:%M:%S)] $*"; }

# --- Rung 1: warm-retrain singles G, R ------------------------------------
log "warm-retraining G, R (+1500 steps)"
OMP_NUM_THREADS=2 $PY -m nca.train --char G --steps 1500 --init weights/0047.json \
    --out weights/0047.json > logs/0047_warm.log 2>&1 &
P1=$!
OMP_NUM_THREADS=2 $PY -m nca.train --char R --steps 1500 --init weights/0052.json \
    --out weights/0052.json > logs/0052_warm.log 2>&1 &
P2=$!
wait $P1 $P2
log "OCR gate: singles G, R"
$PY -m nca.ocr_eval weights/0047.json weights/0052.json \
    --report ocr/singles_GR.json --img-dir grown || { log "GATE FAIL: G/R singles"; exit 1; }
log "PASS singles G,R"

# --- Rung 2: double letters, one grid ("GO") -------------------------------
log "training double 'GO' (1600 steps)"
$PY -m nca.train_word --text GO --steps 1600 --out weights/word_GO.json \
    --snap-dir snaps > logs/word_GO.log 2>&1 || { log "GATE FAIL: GO training crashed"; exit 2; }
$PY -m nca.ocr_word weights/word_GO.json || { log "GATE FAIL: GO OCR"; exit 2; }
log "PASS double GO"
$PY scripts/build_report.py

# --- Rung 3: remaining singles O, W in parallel ----------------------------
log "training singles O, W (2000 steps)"
OMP_NUM_THREADS=2 $PY -m nca.train --char O --steps 2000 --out weights/004f.json > logs/004f.log 2>&1 &
P1=$!
OMP_NUM_THREADS=2 $PY -m nca.train --char W --steps 2000 --out weights/0057.json > logs/0057.log 2>&1 &
P2=$!
wait $P1 $P2
$PY -m nca.ocr_eval weights/004f.json weights/0057.json \
    --report ocr/singles_OW.json --img-dir grown || log "WARN: O/W OCR imperfect (non-fatal)"
$PY scripts/build_report.py

# --- Rung 4: whole word "GROW", one grid ------------------------------------
log "training word 'GROW' (2000 steps)"
$PY -m nca.train_word --text GROW --steps 2000 --batch 4 \
    --out weights/word_GROW.json --snap-dir snaps > logs/word_GROW.log 2>&1 \
    || { log "GATE FAIL: GROW training crashed"; exit 4; }
$PY -m nca.ocr_word weights/word_GROW.json || { log "GATE FAIL: GROW OCR"; exit 4; }
$PY scripts/build_report.py
log "PASS word GROW — ladder complete"
