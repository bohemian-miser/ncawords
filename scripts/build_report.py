"""Aggregate OCR results and weight files into site-servable artifacts.

- weights/index.json        {"chars": [...], "words": [...]}
- docs/ocr_report.json      merged report, img paths relative to docs/
- syncs weights/*.json and grown/*.png into docs/ (GitHub Pages can't
  follow symlinks, so the site keeps real copies)
- re-runs nothing; only collects what exists.

Usage: .venv/bin/python scripts/build_report.py
"""

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    chars, words = [], []
    for f in sorted((ROOT / "weights").glob("*.json")):
        if f.name == "index.json":
            continue
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        if d.get("kind") == "word":
            words.append(d["text"])
        elif "char" in d:
            chars.append(d["char"])
    (ROOT / "weights" / "index.json").write_text(
        json.dumps({"chars": sorted(set(chars)), "words": sorted(set(words))}))

    results = []
    for f in sorted((ROOT / "ocr").glob("*.json")):
        try:
            rep = json.loads(f.read_text())
        except Exception:
            continue
        for r in rep.get("results", []):
            img = Path(r.get("img", ""))
            r["img"] = f"grown/{img.name}" if img.name else ""
            results.append(r)
    # dedup by char, last wins
    by_char = {r["char"]: r for r in results}
    results = [by_char[c] for c in sorted(by_char)]
    report = {"ok": sum(r["ok"] for r in results), "total": len(results),
              "results": results}
    (ROOT / "docs" / "ocr_report.json").write_text(json.dumps(report, indent=1))

    for src, dst in ((ROOT / "weights", ROOT / "docs" / "weights"),
                     (ROOT / "grown", ROOT / "docs" / "grown")):
        dst.mkdir(exist_ok=True)
        for f in src.iterdir():
            if f.is_file():
                shutil.copy2(f, dst / f.name)

    print(f"index: {len(set(chars))} chars, {len(set(words))} words; "
          f"report: {report['ok']}/{report['total']} OCR pass")


if __name__ == "__main__":
    main()
