#!/bin/bash
# Sync results from the fleet, rebuild the site artifacts, commit and push.
# Usage: scripts/publish.sh "commit message"
set -eu
cd "$(dirname "$0")/.."
MSG="${1:-training results update}"

rsync -az cse:projects/ncawords/weights/ weights/ 2>/dev/null || true
rsync -az cse:projects/ncawords/grown/   grown/   2>/dev/null || true
rsync -az cse:projects/ncawords/ocr/     ocr/     2>/dev/null || true

.venv/bin/python scripts/build_report.py
.venv/bin/python scripts/contact_sheet.py

git add -A
if git diff --cached --quiet; then
  echo "nothing new to publish"
  exit 0
fi
git commit -q -m "$MSG

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01USN6buEBYoEF7qh9QWEjHQ"
git push -q origin main
echo "pushed: $MSG"
.venv/bin/python - <<'PY'
import json, pathlib
p = pathlib.Path("docs/ocr_report.json")
if p.exists():
    rep = json.loads(p.read_text())
    ok = "".join(r["char"] for r in rep["results"] if r["ok"])
    bad = "".join(r["char"] for r in rep["results"] if not r["ok"])
    print(f"OCR: {rep['ok']}/{rep['total']} pass | PASS: {ok} | FAIL: {bad}")
PY
