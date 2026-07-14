#!/bin/bash
# Escalation toward the goal: ONE model that grows "COMP6441" on one grid.
#   rung 1: doubles "CO" and "64" (letters and digits, both hollow-centred)
#   rung 2: "COMP"
#   rung 3: "COMP6441"
#
# Each rung is OCR-gated. The gate requires an EXPLICIT "OK" line from the
# judge: a killed or crashed run produces no OCR line at all, and treating
# "no failure printed" as success is how a dead run gets certified as a pass.
set -u
cd "$(dirname "$0")/.."

log() { echo "[wordladder $(date +%H:%M:%S)] $*"; }

sync_back() {
  rsync -az cse:projects/ncawords/weights/ weights/ 2>/dev/null || true
  rsync -az cse:projects/ncawords/grown/   grown/   2>/dev/null || true
  .venv/bin/python scripts/build_report.py >/dev/null 2>&1 || true
}

# gate <logfile> <TEXT> -- passes only on an explicit "-> OCR 'TEXT' OK" line
gate() {
  local logf="$1" want="$2"
  if grep -qE "OCR '${want}' OK" "$logf"; then
    log "PASS: tesseract read '${want}' back"
    return 0
  fi
  log "GATE FAIL for '${want}'. Judge said:"
  grep -E "OCR|FAIL|Error|Traceback" "$logf" | tail -3 | sed 's/^/    /'
  return 1
}

# --- Rung 1: doubles, in parallel -----------------------------------------
log "rung 1: doubles CO + 64 (2500 steps each)"
./scripts/remote_word.sh CO 2500 6 > logs/word_CO.log 2>&1 &
P1=$!
./scripts/remote_word.sh 64 2500 6 > logs/word_64.log 2>&1 &
P2=$!
wait $P1 $P2
sync_back
gate logs/word_CO.log CO || exit 1
gate logs/word_64.log 64 || exit 1
log "PASS rung 1 (doubles: letters and digits both grow)"

# --- Rung 2: COMP ----------------------------------------------------------
log "rung 2: COMP (3500 steps)"
./scripts/remote_word.sh COMP 3500 6 > logs/word_COMP.log 2>&1
sync_back
gate logs/word_COMP.log COMP || exit 2
log "PASS rung 2 (COMP)"

# --- Rung 3: COMP6441 ------------------------------------------------------
log "rung 3: COMP6441 (5000 steps)"
./scripts/remote_word.sh COMP6441 5000 4 > logs/word_COMP6441.log 2>&1
sync_back
gate logs/word_COMP6441.log COMP6441 || exit 3
log "PASS rung 3 — COMP6441 grown from 8 seeds by ONE model, OCR-verified"
