"""Build a progress strip from training snapshots: the same word grown from
its seeds, at increasing training steps. Shows the model learning to write.

Usage: .venv/bin/python scripts/progress_strip.py COMP6441
"""

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

MAX_ROWS = 12
PAD = 8
LBL_W = 92


def main(text):
    snaps = sorted((ROOT / "snaps").glob(f"{text}_*.png"))
    if not snaps:
        print(f"no snapshots for {text} yet")
        return 1

    # Evenly sample up to MAX_ROWS snapshots, always keeping the latest.
    if len(snaps) > MAX_ROWS:
        idx = [round(i * (len(snaps) - 1) / (MAX_ROWS - 1))
               for i in range(MAX_ROWS)]
        snaps = [snaps[i] for i in sorted(set(idx))]

    imgs = [Image.open(p).convert("RGB") for p in snaps]
    scale = 2
    iw, ih = imgs[0].width * scale, imgs[0].height * scale
    W = LBL_W + iw + 2 * PAD
    H = 44 + len(imgs) * (ih + PAD) + PAD

    sheet = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    d.text((PAD, 12), f"'{text}' — one model, one grid, learning to write",
           font=ImageFont.truetype(FONT_B, 17), fill=(20, 20, 20))

    f = ImageFont.truetype(FONT, 14)
    y = 44
    for p, im in zip(snaps, imgs):
        step = p.stem.split("_")[-1].lstrip("0") or "0"
        im = im.resize((iw, ih), Image.NEAREST)
        d.text((PAD, y + ih // 2 - 8), f"step {step}", font=f, fill=(90, 90, 90))
        sheet.paste(im, (LBL_W, y))
        d.rectangle([LBL_W, y, LBL_W + iw, y + ih], outline=(210, 210, 210))
        y += ih + PAD

    out = ROOT / "docs" / f"progress_{text}.png"
    sheet.save(out)
    print(f"progress strip -> {out} ({len(imgs)} snapshots)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "COMP6441"))
