"""Grow each trained character from a seed and verify it with tesseract OCR.

Usage:
  python -m nca.ocr_eval weights/*.json [--steps 96] [--report report.json]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from nca import tess

from nca.model import NCA, make_seed, to_rgb


def load_model(path):
    d = json.loads(Path(path).read_text())
    model = NCA(d["channel_n"], d["fire_rate"], d["hidden_n"])
    sd = model.state_dict()
    w0 = torch.tensor(d["fc0_w"])
    if d.get("layout") == "blocked":
        # Exported column order is [state | sobel_x | sobel_y]; the PyTorch
        # grouped conv produces interleaved [id_c0, sx_c0, sy_c0, id_c1, ...]
        # features, so invert the export reorder.
        c = d["channel_n"]
        w0 = w0.reshape(-1, 3, c).permute(0, 2, 1).reshape(-1, 3 * c)
    sd["fc0.weight"] = w0[:, :, None, None]
    sd["fc0.bias"] = torch.tensor(d["fc0_b"])
    sd["fc1.weight"] = torch.tensor(d["fc1_w"])[:, :, None, None]
    model.load_state_dict(sd)
    return model, d


def grow_image(model, grid, steps=96, upscale=4, seed_pos=None):
    """Grow from the seed the model was TRAINED on (see train.ink_seed_pos);
    growing a letter model from the grid center instead is a different
    initial condition and can simply die."""
    with torch.no_grad():
        x = make_seed(grid, model.channel_n, pos=seed_pos)
        x = model(x, steps=steps)
    img = to_rgb(x)[0].clamp(0, 1).permute(1, 2, 0).numpy()
    img = Image.fromarray((img * 255).astype(np.uint8))
    return img.resize((grid * upscale,) * 2, Image.LANCZOS)


def seed_pos_of(d):
    """Seed (x, y) recorded in a weight file, or the center for old files."""
    seeds = d.get("seeds")
    if seeds:
        return (seeds[0]["x"], seeds[0]["y"])
    return (d["grid"] // 2, d["grid"] // 2)


def ocr_char(img, threshold=235):
    """OCR a single-character image; returns recognized text.

    Binarize at "any visible ink" (render is 1-a+rgb on white, so gray<235
    means alpha above ~0.08); try single-char mode, fall back to word mode.
    """
    g = img.convert("L")
    arr = np.asarray(g)
    if (arr < threshold).sum() == 0:
        return ""
    bw = Image.fromarray(np.where(arr < threshold, 0, 255).astype(np.uint8))
    wl = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    for psm in (10, 8):
        txt = tess.image_to_string(
            bw, config=f"--psm {psm} -c tessedit_char_whitelist={wl}").strip()
        if txt:
            return txt
    return ""


def verdict(got, want):
    """Two verdicts, because tesseract's single-char mode has a quirk.

    strict:  the exact string equals the target (modulo letter case).
    relaxed: every character tesseract returned is the target letter. psm 10
             often emits a letter twice in both cases ('Cc' for C, 'oO' for 0)
             when the glyph is case-ambiguous; that is the judge hedging on
             case, not the CA growing the wrong shape. A relaxed pass still
             requires tesseract to have seen ONLY the intended character —
             'e' for 8 or '-' for I fails both verdicts.
    """
    got_s = got.strip()
    strict = got_s.lower() == want.lower() and got_s != ""
    seen = {c.upper() for c in got_s if c.isalnum()}
    relaxed = seen == {want.upper()}
    return strict, relaxed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("weights", nargs="+")
    p.add_argument("--steps", type=int, default=80)
    p.add_argument("--report", default="ocr_report.json")
    p.add_argument("--img-dir", default="grown")
    p.add_argument("--gate", choices=("strict", "relaxed"), default="relaxed",
                   help="which verdict decides the exit code")
    a = p.parse_args()

    results = []
    for wpath in a.weights:
        model, d = load_model(wpath)
        img = grow_image(model, d["grid"], a.steps, seed_pos=seed_pos_of(d))
        Path(a.img_dir).mkdir(exist_ok=True)
        img_path = Path(a.img_dir) / f"{ord(d['char']):04x}.png"
        img.save(img_path)
        got = ocr_char(img)
        strict, relaxed = verdict(got, d["char"])
        results.append({"char": d["char"], "ocr": got, "ok": relaxed,
                        "strict": strict, "img": str(img_path)})
        tag = "OK" if strict else ("OK(case-dup)" if relaxed else "FAIL")
        print(f"  {d['char']} -> OCR '{got}' {tag}")

    n_strict = sum(r["strict"] for r in results)
    n_ok = sum(r["ok"] for r in results)
    print(f"\n{n_strict}/{len(results)} exact, {n_ok}/{len(results)} recognized "
          f"(allowing tesseract's case-duplicate output)")
    Path(a.report).write_text(json.dumps(
        {"ok": n_ok, "strict": n_strict, "total": len(results),
         "results": results}, indent=1))
    gate = n_strict if a.gate == "strict" else n_ok
    sys.exit(0 if gate == len(results) else 1)


if __name__ == "__main__":
    main()
