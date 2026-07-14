"""Build a contact sheet of every grown glyph next to its target, with the
OCR verdict — one image you can eyeball to judge model quality.

Usage: .venv/bin/python scripts/contact_sheet.py [-o docs/contact_sheet.png]
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

CELL = 128       # thumbnail size
LABEL_H = 34
PAD = 10
COLS = 7


def load_report():
    rep = ROOT / "docs" / "ocr_report.json"
    if not rep.exists():
        return []
    return json.loads(rep.read_text()).get("results", [])


def main(out):
    results = load_report()
    words = sorted((ROOT / "grown").glob("word_*.png"))
    if not results and not words:
        print("nothing to render yet")
        return

    f_lbl = ImageFont.truetype(FONT, 15)
    f_ch = ImageFont.truetype(FONT_B, 17)

    rows = (len(results) + COLS - 1) // COLS
    grid_h = rows * (CELL + LABEL_H + PAD) + PAD
    # word strips at the bottom
    word_imgs = [Image.open(w) for w in words]
    word_h = sum(min(w.height, 96) + LABEL_H + PAD for w in word_imgs)
    W = COLS * (CELL + PAD) + PAD
    header = 54
    H = header + grid_h + (word_h + 24 if word_imgs else 0)

    sheet = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    n_ok = sum(r["ok"] for r in results)
    d.text((PAD, 14), f"Grown glyphs — OCR verdict: {n_ok}/{len(results)} "
                      f"read back correctly by tesseract",
           font=ImageFont.truetype(FONT_B, 19), fill=(20, 20, 20))

    for i, r in enumerate(results):
        cx = PAD + (i % COLS) * (CELL + PAD)
        cy = header + (i // COLS) * (CELL + LABEL_H + PAD)
        p = ROOT / "grown" / Path(r["img"]).name
        if p.exists():
            im = Image.open(p).convert("RGB").resize((CELL, CELL), Image.NEAREST)
            sheet.paste(im, (cx, cy))
        ok = r["ok"]
        d.rectangle([cx, cy, cx + CELL, cy + CELL],
                    outline=(34, 150, 60) if ok else (200, 50, 50), width=2)
        got = (r.get("ocr") or "").strip() or "-"
        d.text((cx + 3, cy + CELL + 4), f"{r['char']}", font=f_ch,
               fill=(20, 20, 20))
        d.text((cx + 22, cy + CELL + 6),
               f"{'OK' if ok else 'read ' + repr(got)}", font=f_lbl,
               fill=(34, 150, 60) if ok else (200, 50, 50))

    y = header + grid_h + 8
    for w, im in zip(words, word_imgs):
        name = w.stem.replace("word_", "")
        scale = min(1.0, 96 / im.height, (W - 2 * PAD) / im.width)
        im2 = im.convert("RGB").resize(
            (int(im.width * scale), int(im.height * scale)), Image.NEAREST)
        d.text((PAD, y), f"one model, one grid: {name}", font=f_ch,
               fill=(20, 20, 20))
        y += LABEL_H - 8
        sheet.paste(im2, (PAD, y))
        d.rectangle([PAD, y, PAD + im2.width, y + im2.height],
                    outline=(160, 160, 160))
        y += im2.height + PAD

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    print(f"contact sheet -> {out}  ({len(results)} glyphs, {len(words)} words)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-o", "--out", default=str(ROOT / "docs" / "contact_sheet.png"))
    main(p.parse_args().out)
