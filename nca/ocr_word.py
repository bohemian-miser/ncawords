"""Grow a word model from its seeds and OCR the whole picture as one word.

Usage: python -m nca.ocr_word weights/word_GO.json [--steps 80]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from nca import tess
import torch

from nca.ocr_eval import load_model
from nca.train_word import grow_word_image


def ocr_word(img, threshold=235):
    """OCR the whole picture as a single word.

    Same "any visible ink" threshold as the per-letter judge (the render is
    1-a+rgb on white, so gray < 235 means alpha above ~0.08).
    """
    g = img.convert("L")
    arr = np.asarray(g)
    bw = Image.fromarray(np.where(arr < threshold, 0, 255).astype(np.uint8))
    txt = tess.image_to_string(
        bw, config="--psm 8 -c tessedit_char_whitelist="
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    return txt.strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("weights")
    p.add_argument("--steps", type=int, default=80)
    p.add_argument("--img-dir", default="grown")
    a = p.parse_args()

    model, d = load_model(a.weights)
    assert d.get("kind") == "word"
    img = grow_word_image(model, d["text"], d["channel_n"], a.steps)
    Path(a.img_dir).mkdir(exist_ok=True)
    img_path = Path(a.img_dir) / f"word_{d['text']}.png"
    img.save(img_path)
    got = ocr_word(img)
    ok = got == d["text"].upper()
    print(f"'{d['text']}' -> OCR '{got}' {'OK' if ok else 'FAIL'} ({img_path})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
