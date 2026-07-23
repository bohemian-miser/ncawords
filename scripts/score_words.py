"""OCR quality control for word-growing runs.

Pulls each run's latest COMP snapshot from the public bucket, runs
tesseract on it, and scores against the expected text: exact match,
uppercase-normalised match, and character edit distance. The loss can lie
(dead boards, blurry vibes); OCR reads what a human would.

Usage:
  python scripts/score_words.py --prefixes cls-,fanword-,cw- [--expect COMP]
"""
import argparse
import difflib
import io
import json
import re
import urllib.request

from PIL import Image
import pytesseract

BUCKET = "https://storage.googleapis.com/recipe-lanes-nca-jobs"
API = "https://storage.googleapis.com/storage/v1/b/recipe-lanes-nca-jobs/o"


def list_runs(prefix):
    with urllib.request.urlopen(
            f"{API}?prefix={prefix}&delimiter=/&fields=prefixes&maxResults=500") as r:
        return [p.rstrip("/") for p in json.load(r).get("prefixes", [])]


def latest_comp(run):
    with urllib.request.urlopen(
            f"{API}?prefix={run}/COMP_&fields=items(name)&maxResults=1000") as r:
        items = [i["name"] for i in json.load(r).get("items", [])]
    return sorted(items)[-1] if items else None


def ocr_score(run, expect):
    name = latest_comp(run)
    if not name:
        return None
    with urllib.request.urlopen(f"{BUCKET}/{name}") as r:
        img = Image.open(io.BytesIO(r.read())).convert("L")
    img = img.resize((img.width * 3, img.height * 3), Image.LANCZOS)
    txt = pytesseract.image_to_string(
        img, config="--psm 7 -c tessedit_char_whitelist="
                    "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    txt = re.sub(r"[^A-Z0-9]", "", txt.upper())
    ratio = difflib.SequenceMatcher(None, txt, expect).ratio()
    return {"run": run, "read": txt or "(nothing)", "match": txt == expect,
            "sim": round(ratio, 2), "img": name}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--prefixes", default="cls-")
    p.add_argument("--expect", default="COMP")
    a = p.parse_args()
    rows = []
    for pref in a.prefixes.split(","):
        for run in list_runs(pref.strip()):
            try:
                s = ocr_score(run, a.expect)
                if s:
                    rows.append(s)
                    print(f"{s['run']:<28} read='{s['read']}' "
                          f"sim {s['sim']}{'  MATCH' if s['match'] else ''}",
                          flush=True)
            except Exception as e:
                print(f"{run}: skip ({str(e)[:50]})", flush=True)
    rows.sort(key=lambda r: -r["sim"])
    n_match = sum(r["match"] for r in rows)
    print(f"\n{n_match}/{len(rows)} exact OCR matches; best:")
    for r in rows[:8]:
        print(f"  {r['run']:<28} '{r['read']}' sim {r['sim']}")
