#!/bin/bash
# Train the alphabet across the CSE fleet, N letters concurrently.
# Each letter = one ssh session = its own node + 2 cores.
#
# Usage: scripts/fleet.sh [CHARS] [CONCURRENCY] [STEPS]
set -u
cd "$(dirname "$0")/.."
CHARS="${1:-ABCDEFGHIJKLMNOPQRSTUVWXYZ}"
CONC="${2:-8}"
STEPS="${3:-2000}"

echo "[fleet $(date +%H:%M:%S)] training ${#CHARS} letters, ${CONC} at a time, ${STEPS} steps each"
echo "$CHARS" | fold -w1 | grep -v '^$' \
  | xargs -P "$CONC" -I{} ./scripts/remote_letter.sh {} "$STEPS"
echo "[fleet $(date +%H:%M:%S)] all letters done — syncing results back"

rsync -az cse:projects/ncawords/weights/ weights/
rsync -az cse:projects/ncawords/grown/   grown/
rsync -az cse:projects/ncawords/ocr/     ocr/
.venv/bin/python scripts/build_report.py
echo "[fleet $(date +%H:%M:%S)] synced. OCR summary:"
.venv/bin/python - <<'PY'
import json, pathlib
rep = json.loads(pathlib.Path("docs/ocr_report.json").read_text())
ok = "".join(r["char"] for r in rep["results"] if r["ok"])
bad = "".join(r["char"] for r in rep["results"] if not r["ok"])
print(f"  PASS ({len(ok)}): {ok}")
print(f"  FAIL ({len(bad)}): {bad}")
PY
