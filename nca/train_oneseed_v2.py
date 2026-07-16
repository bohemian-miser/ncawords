"""Train ONE NCA that grows a whole multi-character string from a SINGLE seed.
V2: Closer letters, light web, lower resolution (pixel art style).
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

PITCH = 14          # Lower pitch for closer letters
MARGIN = 6          # Lower margin
GRID_H = 20         # Lower height for lower res

def word_geometry(text):
    w = MARGIN * 2 + PITCH * len(text)
    return w, GRID_H

def render_word(text, glyph=12, font_path=FONT_PATH):
    """Whole string on a transparent canvas, per-char color, with a light connecting web."""
    w, h = word_geometry(text)
    font = ImageFont.truetype(font_path, glyph)
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Draw a "light web" (faint line) connecting the letters
    # Along the center
    draw.line((MARGIN, h // 2, w - MARGIN, h // 2), fill=(128, 128, 128, 64))
    # Along the bottom
    draw.line((MARGIN, h - 4, w - MARGIN, h - 4), fill=(128, 128, 128, 32))

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
    
    cy, cx = h // 2, w // 2
    if tgt is not None:
        y_ids, x_ids = np.where(tgt[3] > 0.5)
        if len(y_ids) > 0:
            distances = (y_ids - cy)**2 + (x_ids - cx)**2
            best_idx = np.argmin(distances)
            cy, cx = y_ids[best_idx], x_ids[best_idx]
            
    x[:, 3:, cy, cx] = 1.0
    return x

def make_noise_seed(text, channel_n=16, n=1):
    w, h = word_geometry(text)
    x = torch.randn(n, channel_n, h, w) * 0.1
    x[:, 3:, h // 2, w // 2] = 1.0 
    return x

def train(text, steps=4000, glyph=12, channel_n=32, hidden_n=128,
          batch=8, pool_size=256, lr=2e-3, damage_n=1, ca_min=64, ca_max=96,
          log_every=100, out=None, snap_dir=None, seed_type="single"):
    torch.manual_seed(sum(map(ord, text)) + 99)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")
    
    tgt = render_word(text, glyph)
    target = torch.from_numpy(tgt)[None].repeat(batch, 1, 1, 1).to(device)

    # Save target image for verification
    if snap_dir:
        Path(snap_dir).mkdir(parents=True, exist_ok=True)
        tgt_img = (tgt.transpose(1, 2, 0) * 255).astype(np.uint8)
        Image.fromarray(tgt_img).save(Path(snap_dir) / "target.png")

    model = NCA(channel_n, hidden_n=hidden_n).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=[int(steps * 0.8)], gamma=0.1)

    if seed_type == "noise":
        seed = make_noise_seed(text, channel_n)
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
        x[:1] = seed.to(device) 
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
        # Placeholder for export, since we don't need it yet
        pass
    print(f"Final loss for {text} (seed={seed_type}): {loss.item():.5f}")
    return model

def save_word_png(model, text, channel_n, path, seed_type, device, n_steps=120):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Upscale by 8 for visibility
    grow_word_image(model, text, channel_n, n_steps, upscale=8, seed_type=seed_type, device=device).save(path)

def grow_word_image(model, text, channel_n, n_steps=120, upscale=8, seed_type="single", device="cpu"):
    with torch.no_grad():
        if seed_type == "noise":
            x = make_noise_seed(text, channel_n)
        else:
            tgt = render_word(text, 12)
            x = make_single_seed(text, channel_n, tgt=tgt)
        x = model(x.to(device), steps=n_steps)
    img = to_rgb(x)[0].cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    im = Image.fromarray((img * 255).astype(np.uint8))
    rez_method = getattr(Image, "Resampling", Image).NEAREST # Use NEAREST for pixel art style
    return im.resize((im.width * upscale, im.height * upscale), rez_method)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--text", required=True)
    p.add_argument("--steps", type=int, default=2000) # Lower steps for testing
    p.add_argument("--out", default=None)
    p.add_argument("--snap-dir", default=None)
    p.add_argument("--seed-type", default="single")
    a = p.parse_args()
    
    train(a.text, steps=a.steps, out=a.out, snap_dir=a.snap_dir, seed_type=a.seed_type)
