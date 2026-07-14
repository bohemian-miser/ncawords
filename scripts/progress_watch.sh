#!/bin/bash
# Pull training snapshots from the fleet, rebuild progress strips, push.
# Runs until the named word's training finishes.
#
# Usage: scripts/progress_watch.sh COMP6441 [INTERVAL_SEC]
set -u
cd "$(dirname "$0")/.."
TXT="${1:-COMP6441}"
IVL="${2:-240}"

while true; do
  rsync -az cse:projects/ncawords/snaps/   snaps/   2>/dev/null || true
  rsync -az cse:projects/ncawords/weights/ weights/ 2>/dev/null || true
  rsync -az cse:projects/ncawords/grown/   grown/   2>/dev/null || true

  .venv/bin/python scripts/progress_strip.py "$TXT" >/dev/null 2>&1 || true
  .venv/bin/python scripts/build_report.py   >/dev/null 2>&1 || true
  .venv/bin/python scripts/contact_sheet.py  >/dev/null 2>&1 || true

  LAST=$(ls snaps/${TXT}_*.png 2>/dev/null | tail -1 | sed "s/.*_0*//;s/\.png//")
  git add -A >/dev/null 2>&1
  if ! git diff --cached --quiet; then
    git commit -q -m "training progress: ${TXT} @ step ${LAST:-0}

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01USN6buEBYoEF7qh9QWEjHQ" \
      && git push -q origin main \
      && echo "[watch $(date +%H:%M:%S)] pushed ${TXT} @ step ${LAST:-0}"
  fi

  # stop once training has exited on the remote and weights exist
  RUNNING=$(ssh -o ConnectTimeout=10 cse \
    "ps -eo args --no-headers | grep -c \"[t]rain_word --text ${TXT}\"" 2>/dev/null || echo 1)
  if [ "${RUNNING:-1}" = "0" ] && [ -f "weights/word_${TXT}.json" ]; then
    echo "[watch $(date +%H:%M:%S)] ${TXT} training finished"
    break
  fi
  sleep "$IVL"
done
