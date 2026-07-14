#!/bin/bash
# Escalation toward the goal: ONE model that grows "COMP6441" on one grid.
#   rung 1: doubles "CO" and "64" (letters and digits both work)
#   rung 2: "COMP"
#   rung 3: "COMP6441"
# Each rung is OCR-gated (tesseract must read the whole word back).
# Rungs 1 runs its two models concurrently; later rungs are single models.
set -u
cd "$(dirname "$0")/.."

log() { echo "[wordladder $(date +%H:%M:%S)] $*"; }

sync_back() {
  rsync -az cse:projects/ncawords/weights/ weights/ 2>/dev/null
  rsync -az cse:projects/ncawords/grown/   grown/   2>/dev/null
  .venv/bin/python scripts/build_report.py >/dev/null 2>&1
}

# --- Rung 1: doubles, in parallel -----------------------------------------
log "rung 1: doubles CO + 64 (2500 steps each)"
./scripts/remote_word.sh CO 2500 6 > logs/word_CO.log 2>&1 &
P1=$!
./scripts/remote_word.sh 64 2500 6 > logs/word_64.log 2>&1 &
P2=$!
wait $P1 $P2
sync_back
grep -h "OCR" logs/word_CO.log logs/word_64.log
if grep -q "OCR-FAIL\|FAIL (" logs/word_CO.log logs/word_64.log; then
  log "GATE FAIL: doubles did not read back — stopping"; exit 1
fi
log "PASS rung 1 (doubles)"

# --- Rung 2: COMP ----------------------------------------------------------
log "rung 2: COMP (3500 steps)"
./scripts/remote_word.sh COMP 3500 6 > logs/word_COMP.log 2>&1
sync_back
grep -h "OCR" logs/word_COMP.log
if grep -q "OCR-FAIL\|FAIL (" logs/word_COMP.log; then
  log "GATE FAIL: COMP did not read back — stopping"; exit 2
fi
log "PASS rung 2 (COMP)"

# --- Rung 3: COMP6441 ------------------------------------------------------
log "rung 3: COMP6441 (5000 steps)"
./scripts/remote_word.sh COMP6441 5000 4 > logs/word_COMP6441.log 2>&1
sync_back
grep -h "OCR" logs/word_COMP6441.log
if grep -q "OCR-FAIL\|FAIL (" logs/word_COMP6441.log; then
  log "GATE FAIL: COMP6441 did not read back"; exit 3
fi
log "PASS rung 3 — COMP6441 grown from seeds on ONE model, OCR-verified"
