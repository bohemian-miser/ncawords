"""Regeneration test: grow, wound, heal, and ask the OCR judge again.

Grows a word model from its seeds, cuts a hole in it, lets the CA run on, and
checks that tesseract can still read the string. This is the claim the paper's
damage training is supposed to buy, and the one a live demo depends on: a
model that grows a pretty picture but cannot repair it will fail on stage.

Also writes a before/wound/after strip for the site.

Usage: python -m nca.regen_test weights/word_COMP6441.json [--cuts 3]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from nca.model import make_seed, to_rgb
from nca.ocr_eval import load_model
from nca.ocr_word import ocr_word
from nca.train_word import make_word_seed

FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def render(x):
    img = to_rgb(x)[0].clamp(0, 1).permute(1, 2, 0).numpy()
    return Image.fromarray((img * 255).astype(np.uint8))


def cut(x, cx, cy, r):
    """Zero every channel inside a disc — the same wound a mouse drag makes."""
    _, _, h, w = x.shape
    yy, xx = torch.meshgrid(torch.arange(h, dtype=torch.float32),
                            torch.arange(w, dtype=torch.float32),
                            indexing="ij")
    keep = ((xx - cx) ** 2 + (yy - cy) ** 2 >= r ** 2).float()
    return x * keep[None, None]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("weights")
    p.add_argument("--grow", type=int, default=90)
    p.add_argument("--heal", type=int, default=90)
    p.add_argument("--cuts", type=int, default=3)
    p.add_argument("--radius", type=int, default=8)
    p.add_argument("--out", default="docs/regen_strip.png")
    a = p.parse_args()

    model, d = load_model(a.weights)
    text = d["text"]
    torch.manual_seed(7)

    x = make_word_seed(text, d["channel_n"], seeds=d.get("seeds"))
    with torch.no_grad():
        x = model(x, steps=a.grow)
    grown = render(x)
    ocr_before = ocr_word(grown.resize((grown.width * 4, grown.height * 4),
                                       Image.LANCZOS))

    # Wound it: evenly spaced cuts across the string, so several characters
    # are hit rather than one unlucky corner.
    _, _, h, w = x.shape
    with torch.no_grad():
        for i in range(a.cuts):
            cx = w * (i + 0.5) / a.cuts
            x = cut(x, cx, h / 2, a.radius)
    wounded = render(x)

    with torch.no_grad():
        x = model(x, steps=a.heal)
    healed = render(x)
    ocr_after = ocr_word(healed.resize((healed.width * 4, healed.height * 4),
                                       Image.LANCZOS))

    ok = ocr_after == text.upper()
    print(f"grown   -> OCR '{ocr_before}'")
    print(f"{a.cuts} cuts of radius {a.radius} punched out")
    print(f"healed  -> OCR '{ocr_after}' {'RECOVERED' if ok else 'FAILED TO RECOVER'}")

    # before / wound / after strip
    scale = 3
    pad, lbl = 12, 26
    iw, ih = grown.width * scale, grown.height * scale
    sheet = Image.new("RGB", (iw + 2 * pad, 3 * (ih + lbl) + 4 * pad),
                      (255, 255, 255))
    dr = ImageDraw.Draw(sheet)
    f = ImageFont.truetype(FONT_B, 15)
    y = pad
    for img, label in ((grown, f"grown  (OCR: '{ocr_before}')"),
                       (wounded, f"wounded  ({a.cuts} holes cut)"),
                       (healed, f"healed  (OCR: '{ocr_after}')"
                                f"{'  ✓' if ok else '  ✗'}")):
        dr.text((pad, y), label, font=f, fill=(20, 20, 20))
        y += lbl
        sheet.paste(img.resize((iw, ih), Image.NEAREST), (pad, y))
        dr.rectangle([pad, y, pad + iw, y + ih], outline=(200, 200, 200))
        y += ih + pad

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(a.out)
    print(f"strip -> {a.out}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
