"""Train ONE NCA that grows a whole multi-character string from a SINGLE seed.

Unlike train_word.py which multiplexes alphabets with explicit positional
seeds per character, this trains a single system to organically grow the
entire word from one central pixel, relying on emergence and a larger capacity,
just like the original Mordvintsev distill.pub NCA.

Usage:
  python -m nca.train_one_seed --text COMP --steps 4000
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

def word_geometry(text):
    w = MARGIN * 2 + PITCH * len(text)
    return w, GRID_H

def render_word(text, glyph=22, font_path=FONT_PATH):
    """Whole string on a transparent canvas, per-char color."""
    w, h = word_geometry(text)
    font = ImageFont.truetype(font_path, glyph)
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    for i, ch in enumerate(text):
        x_center = MARGIN + PITCH * i + PITCH // 2
        l, t, r, b = draw.textbbox((0, 0), ch, font=font)
        draw.text((x_center - (r - l) / 2 - l, (h - (b - t)) / 2 - t), ch,
                  font=font, fill=char_color(ch) + (255,))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr[..., :3] *= arr[..., 3:]
    return arr.transpose(2, 0, 1)  # [4, H, W]

def make_single_seed(text, channel_n=16, n=1, tgt=None):
    w, h = word_geometry(text)
    x = torch.zeros(n, channel_n, h, w)
    
    # Place a single living pixel near the center, 
    # but strictly on a pixel where target is non-empty to prevent early 'suicide' local minima
    cy, cx = h // 2, w // 2
    if tgt is not None:
        # find pixels where alpha (channel 3) > 0.5
        y_ids, x_ids = np.where(tgt[3] > 0.5)
        if len(y_ids) > 0:
            distances = (y_ids - cy)**2 + (x_ids - cx)**2
            best_idx = np.argmin(distances)
            cy, cx = y_ids[best_idx], x_ids[best_idx]
            
    x[:, 3:, cy, cx] = 1.0
    return x

def make_offset_seed(text, channel_n=16, n=1):
    w, h = word_geometry(text)
    x = torch.zeros(n, channel_n, h, w)
    # A single living pixel in the center of the first letter
    x[:, 3:, h // 2, MARGIN + PITCH // 2] = 1.0
    return x

def make_noise_seed(text, channel_n=16, n=1):
    w, h = word_geometry(text)
    x = torch.randn(n, channel_n, h, w) * 0.1
    x[:, 3:, h // 2, w // 2] = 1.0 # break symmetry heavily at the center
    return x

def train(text, steps=4000, glyph=22, channel_n=32, hidden_n=128,
          batch=8, pool_size=256, lr=2e-3, damage_n=1, ca_min=64, ca_max=96,
          log_every=10, out=None, snap_dir=None, seed_type="single"):
    torch.manual_seed(sum(map(ord, text)) + 99)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")
    
    tgt = render_word(text, glyph)
    target = torch.from_numpy(tgt)[None].repeat(batch, 1, 1, 1).to(device)

    model = NCA(channel_n, hidden_n=hidden_n).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=[int(steps * 0.8)], gamma=0.1)

    if seed_type == "noise":
        seed = make_noise_seed(text, channel_n)
    elif seed_type == "offset":
        seed = make_offset_seed(text, channel_n)
    else:
        seed = make_single_seed(text, channel_n, tgt=tgt)
    
    pool = SamplePool(seed, pool_size)
    h, w = seed.shape[2], seed.shape[3]

    t0 = time.time()
    for step in range(steps):
        idx, x = pool.sample(batch)
        x = x.to(device)
        with torch.no_grad():
            loss_rank = F.mse_loss(to_rgba(x), target, reduction="none") \
                .mean(dim=(1, 2, 3)).argsort(descending=True)
        x = x[loss_rank]
        x[:1] = seed.to(device) # Anchored to scratch
        if damage_n:
            m = damage_mask(damage_n, max(h, w), device)[:, :, :h, :w]
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
        pool.commit(idx, x.cpu())

        if step % log_every == 0 or step == steps - 1:
            print(f"[{text}] step {step} loss {loss.item():.5f} "
                  f"({(time.time() - t0):.1f}s)", flush=True)
            if snap_dir:
                save_word_png(model, text, channel_n,
                              Path(snap_dir) / f"{text}_{step:05d}.png",
                              seed_type, device)

    if out:
        export_model(model, text, glyph, Path(out))
    print(f"Final loss for {text} (seed={seed_type}): {loss.item():.5f}")
    return model

def grow_word_image(model, text, channel_n, n_steps=120, upscale=4, seed_type="single", device="cpu"):
    with torch.no_grad():
        if seed_type == "noise":
            x = make_noise_seed(text, channel_n)
        elif seed_type == "offset":
            x = make_offset_seed(text, channel_n)
        else:
            tgt = render_word(text, 22)
            x = make_single_seed(text, channel_n, tgt=tgt)
        x = model(x.to(device), steps=n_steps)
    img = to_rgb(x)[0].cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    im = Image.fromarray((img * 255).astype(np.uint8))
    rez_method = getattr(Image, "Resampling", Image).LANCZOS
    return im.resize((im.width * upscale, im.height * upscale), rez_method)

def save_word_png(model, text, channel_n, path, seed_type, device, n_steps=120):
    path.parent.mkdir(parents=True, exist_ok=True)
    grow_word_image(model, text, channel_n, n_steps, upscale=2, seed_type=seed_type, device=device).save(path)

def export_model(model, text, glyph, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"[{text}] exported pytorch weights -> {path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--text", required=True)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--glyph", type=int, default=22)
    p.add_argument("--channels", type=int, default=32)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--damage", type=int, default=1)
    p.add_argument("--camin", type=int, default=64)
    p.add_argument("--camax", type=int, default=96)
    p.add_argument("--out", default=None)
    p.add_argument("--snap-dir", default=None)
    p.add_argument("--seed-type", default="single")
    a = p.parse_args()
    
    train(a.text, steps=a.steps, glyph=a.glyph, channel_n=a.channels,
          hidden_n=a.hidden, batch=a.batch, damage_n=a.damage,
          ca_min=a.camin, ca_max=a.camax,
          out=a.out or f"weights/oneseed_{a.text}.pt",
          snap_dir=a.snap_dir,
          seed_type=a.seed_type)
