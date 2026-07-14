"""Train ONE NCA that grows a whole multi-character string on a single grid.

Each character gets its own seed cell, placed at the center of a fixed-pitch
slot. Seeds are distinguished by a 6-bit binary code (the character's index
in CHARSET) written into hidden channels 4..9 at seed time; remaining hidden
channels are 1. The model must learn code -> glyph, so different seeds grow
different characters while sharing one update rule.

State layout (channel_n=16): 0-2 RGB, 3 alpha, 4-9 char code, 10-15 free.

Usage:
  python -m nca.train_word --text COMP --steps 2500 --out weights/word_COMP.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from nca.model import NCA, to_rgba, to_rgb
from nca.train import FONT_PATH, char_color, damage_mask, SamplePool

PITCH = 24          # px per character slot
MARGIN = 12         # left/right margin
GRID_H = 32
CODE_CH0 = 4        # first code channel
CODE_BITS = 6       # 6 bits = 64 slots, enough for the 36-char set

CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def char_code(ch):
    """6-bit binary code for A-Z and 0-9, as floats 0/1."""
    i = CHARSET.find(ch.upper())
    assert i >= 0, f"unsupported character {ch!r} (allowed: {CHARSET})"
    return [(i >> b) & 1 for b in range(CODE_BITS)]


def word_geometry(text):
    w = MARGIN * 2 + PITCH * len(text)
    seeds = []
    for i, ch in enumerate(text):
        seeds.append({"x": MARGIN + PITCH * i + PITCH // 2, "y": GRID_H // 2,
                      "code": char_code(ch), "char": ch})
    return w, GRID_H, seeds


def render_word(text, glyph=22, font_path=FONT_PATH):
    """Each char centered in its slot, per-char color, premultiplied RGBA.

    Seed positions are then snapped onto ink within each character's slot:
    a seed sitting on background (the hollow middle of C, O, 6, 0...) is
    told by the loss to switch itself off, which kills the whole grid.
    """
    w, h, seeds = word_geometry(text)
    font = ImageFont.truetype(font_path, glyph)
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    for s in seeds:
        ch = s["char"]
        l, t, r, b = draw.textbbox((0, 0), ch, font=font)
        draw.text((s["x"] - (r - l) / 2 - l, (h - (b - t)) / 2 - t), ch,
                  font=font, fill=char_color(ch) + (255,))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr[..., :3] *= arr[..., 3:]
    target = arr.transpose(2, 0, 1)  # [4, H, W]

    from nca.train import ink_seed_pos
    tt = torch.from_numpy(target)
    for i, s in enumerate(seeds):
        x0 = MARGIN + PITCH * i
        sx, sy = ink_seed_pos(tt, region=(x0, x0 + PITCH))
        s["x"], s["y"] = sx, sy
    return target, seeds


def make_word_seed(text, channel_n=16, n=1, seeds=None, glyph=22):
    """One seed per character. `seeds` must be the ink-anchored positions from
    render_word (or a weight file); recomputing slot centers here would seed
    hollow glyphs on background and kill the grid."""
    w, h, _ = word_geometry(text)
    if seeds is None:
        _, seeds = render_word(text, glyph)
    x = torch.zeros(n, channel_n, h, w)
    for s in seeds:
        x[:, 3:, s["y"], s["x"]] = 1.0
        for b, bit in enumerate(s["code"]):
            x[:, CODE_CH0 + b, s["y"], s["x"]] = float(bit)
    return x


def train(text, steps=1200, glyph=22, channel_n=16, hidden_n=80,
          batch=6, pool_size=128, lr=2e-3, damage_n=1, ca_min=36, ca_max=56,
          log_every=100, out=None, snap_dir=None):
    torch.manual_seed(sum(map(ord, text)) + 99)
    tgt, seeds = render_word(text, glyph)
    target = torch.from_numpy(tgt)[None].repeat(batch, 1, 1, 1)

    model = NCA(channel_n, hidden_n=hidden_n)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=[int(steps * 0.7)], gamma=0.1)

    seed = make_word_seed(text, channel_n, seeds=seeds)
    print(f"[{text}] seeds (ink-anchored): "
          f"{[(s['char'], s['x'], s['y']) for s in seeds]}", flush=True)
    pool = SamplePool(seed, pool_size)
    h, w = seed.shape[2], seed.shape[3]

    t0 = time.time()
    for step in range(steps):
        idx, x = pool.sample(batch)
        with torch.no_grad():
            loss_rank = F.mse_loss(to_rgba(x), target, reduction="none") \
                .mean(dim=(1, 2, 3)).argsort(descending=True)
        x = x[loss_rank]
        x[:1] = seed
        if damage_n:
            # non-square grid: build mask on short axis scale
            m = damage_mask(damage_n, max(h, w), "cpu")[:, :, :h, :w]
            x[-damage_n:] *= m

        n_ca = int(torch.randint(ca_min, ca_max + 1, (1,)))
        x = model(x, steps=n_ca)
        loss = F.mse_loss(to_rgba(x), target)

        opt.zero_grad()
        loss.backward()
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    p.grad /= (p.grad.norm() + 1e-8)
        opt.step()
        sched.step()
        pool.commit(idx, x)

        if step % log_every == 0 or step == steps - 1:
            print(f"[{text}] step {step} loss {loss.item():.5f} "
                  f"({(time.time() - t0):.0f}s)", flush=True)
            if snap_dir:
                save_word_png(model, text, channel_n,
                              Path(snap_dir) / f"{text}_{step:05d}.png")

    if out:
        export_word(model, text, glyph, seeds, Path(out))
    return model


def grow_word_image(model, text, channel_n, n_steps=80, upscale=4, seeds=None):
    with torch.no_grad():
        x = make_word_seed(text, channel_n, seeds=seeds)
        x = model(x, steps=n_steps)
    img = to_rgb(x)[0].clamp(0, 1).permute(1, 2, 0).numpy()
    im = Image.fromarray((img * 255).astype(np.uint8))
    return im.resize((im.width * upscale, im.height * upscale), Image.LANCZOS)


def save_word_png(model, text, channel_n, path, n_steps=80):
    path.parent.mkdir(parents=True, exist_ok=True)
    grow_word_image(model, text, channel_n, n_steps, upscale=2).save(path)


def export_word(model, text, glyph, seeds, path):
    from nca.train import export_weights
    export_weights(model, text, 0, glyph, path)  # grid=0 placeholder
    d = json.loads(path.read_text())
    w, h, _ = word_geometry(text)
    d.update({"kind": "word", "text": text, "grid_w": w, "grid_h": h,
              "grid": None,
              "seeds": [{"x": s["x"], "y": s["y"], "code": s["code"],
                         "char": s["char"]} for s in seeds],
              "code_ch0": CODE_CH0, "code_bits": CODE_BITS})
    path.write_text(json.dumps(d))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--text", required=True)
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--glyph", type=int, default=22)
    p.add_argument("--channels", type=int, default=16)
    p.add_argument("--hidden", type=int, default=80)
    p.add_argument("--batch", type=int, default=6)
    p.add_argument("--out", default=None)
    p.add_argument("--snap-dir", default=None)
    a = p.parse_args()
    train(a.text, steps=a.steps, glyph=a.glyph, channel_n=a.channels,
          hidden_n=a.hidden, batch=a.batch,
          out=a.out or f"weights/word_{a.text}.json",
          snap_dir=a.snap_dir)
