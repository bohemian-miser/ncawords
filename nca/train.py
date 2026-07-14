"""Train one NCA to grow a single character glyph.

Usage:
  python -m nca.train --char A --out weights/0041.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from nca.model import NCA, make_seed, to_rgba

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# A palette so grown letters aren't all black; hue varies per character.
def char_color(ch):
    import colorsys
    hue = (ord(ch) * 47 % 360) / 360.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 0.80)
    return int(r * 255), int(g * 255), int(b * 255)


def render_glyph(ch, grid=32, glyph=22, font_path=FONT_PATH, color=None):
    """Render `ch` centered on a transparent grid x grid canvas.
    Returns float32 RGBA array [4, H, W] with premultiplied alpha."""
    font = ImageFont.truetype(font_path, glyph)
    img = Image.new("RGBA", (grid, grid), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    l, t, r, b = draw.textbbox((0, 0), ch, font=font)
    draw.text(((grid - (r - l)) / 2 - l, (grid - (b - t)) / 2 - t), ch,
              font=font, fill=(color or char_color(ch)) + (255,))
    arr = np.asarray(img, dtype=np.float32) / 255.0  # [H, W, 4]
    arr[..., :3] *= arr[..., 3:]  # premultiply alpha
    return arr.transpose(2, 0, 1)


class SamplePool:
    def __init__(self, seed, size=256):
        self.pool = seed.repeat(size, 1, 1, 1)

    def sample(self, n):
        idx = torch.randperm(self.pool.shape[0])[:n]
        return idx, self.pool[idx].clone()

    def commit(self, idx, x):
        self.pool[idx] = x.detach()


def damage_mask(n, size, device):
    """1 outside a random circle, 0 inside (multiply to damage).
    Center in middle half of grid, radius 0.1-0.4 of half-size."""
    x = torch.linspace(-1, 1, size, device=device)
    yy, xx = torch.meshgrid(x, x, indexing="ij")
    cx = torch.rand(n, 1, 1, device=device) - 0.5
    cy = torch.rand(n, 1, 1, device=device) - 0.5
    r = torch.rand(n, 1, 1, device=device) * 0.3 + 0.1
    mask = ((xx[None] - cx) ** 2 + (yy[None] - cy) ** 2 >= r ** 2).float()
    return mask[:, None]


def train(ch, steps=1000, grid=32, glyph=22, channel_n=12, hidden_n=64,
          batch=8, pool_size=256, lr=2e-3, damage_n=1, ca_min=36, ca_max=56,
          device="cpu", log_every=100, out=None, snap_dir=None,
          milestone=0.7, init=None):
    torch.manual_seed(ord(ch) + 1234)
    target = torch.from_numpy(render_glyph(ch, grid, glyph))[None].to(device)
    target = target.repeat(batch, 1, 1, 1)

    if init:
        from nca.ocr_eval import load_model
        model, d0 = load_model(init)
        model = model.to(device)
        print(f"[{ch}] warm start from {init}", flush=True)
    else:
        model = NCA(channel_n, hidden_n=hidden_n).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=[int(steps * milestone)], gamma=0.1)

    seed = make_seed(grid, channel_n)
    pool = SamplePool(seed, pool_size)

    t0 = time.time()
    for step in range(steps):
        idx, x = pool.sample(batch)
        x = x.to(device)
        # Rank by loss: worst sample is replaced with the seed (keeps the
        # pool anchored to growth from scratch); best get damaged.
        with torch.no_grad():
            loss_rank = F.mse_loss(to_rgba(x), target, reduction="none") \
                .mean(dim=(1, 2, 3)).argsort(descending=True)
        x = x[loss_rank]
        x[:1] = seed.to(device)
        if damage_n:
            x[-damage_n:] *= damage_mask(damage_n, grid, device)

        n_ca = int(torch.randint(ca_min, ca_max + 1, (1,)))
        x = model(x, steps=n_ca)
        loss = F.mse_loss(to_rgba(x), target)

        opt.zero_grad()
        loss.backward()
        # Per-parameter gradient L2 normalization (paper's stability trick)
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    p.grad /= (p.grad.norm() + 1e-8)
        opt.step()
        sched.step()
        pool.commit(idx, x.cpu())

        if step % log_every == 0 or step == steps - 1:
            print(f"[{ch}] step {step} loss {loss.item():.5f} "
                  f"({(time.time() - t0):.0f}s)", flush=True)
            if snap_dir:
                save_growth_png(model, grid, channel_n,
                                Path(snap_dir) / f"{ord(ch):04x}_{step:05d}.png")

    if out:
        export_weights(model, ch, grid, glyph, Path(out))
    return model


def save_growth_png(model, grid, channel_n, path, n_steps=80):
    from nca.model import to_rgb
    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        x = make_seed(grid, channel_n)
        x = model(x, steps=n_steps)
        img = to_rgb(x)[0].clamp(0, 1).permute(1, 2, 0).numpy()
    Image.fromarray((img * 255).astype(np.uint8)).save(path)


def export_weights(model, ch, grid, glyph, path):
    """Export weights as JSON for browser inference.

    fc0 rows are reordered from PyTorch's interleaved perception layout
    [id_c0, sx_c0, sy_c0, id_c1, ...] to blocked [all-id | all-sx | all-sy],
    which is the natural order for a from-scratch JS/GLSL implementation.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    c = model.channel_n
    w0 = model.fc0.weight.detach().squeeze(-1).squeeze(-1).numpy()  # [H, 3C]
    w0 = w0.reshape(-1, c, 3).transpose(0, 2, 1).reshape(-1, 3 * c)
    d = {
        "char": ch,
        "grid": grid,
        "glyph": glyph,
        "channel_n": c,
        "hidden_n": model.fc0.out_channels,
        "fire_rate": model.fire_rate,
        "layout": "blocked",  # perception vector = [state | sobel_x | sobel_y]
        "fc0_w": w0.round(5).tolist(),                     # [hidden, 3C]
        "fc0_b": model.fc0.bias.detach().numpy().round(5).tolist(),
        "fc1_w": model.fc1.weight.detach().squeeze(-1).squeeze(-1)
                     .numpy().round(5).tolist(),           # [C, hidden]
    }
    path.write_text(json.dumps(d))
    print(f"[{ch}] exported -> {path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--char", required=True)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--grid", type=int, default=32)
    p.add_argument("--glyph", type=int, default=22)
    p.add_argument("--channels", type=int, default=12)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--damage", type=int, default=1)
    p.add_argument("--out", default=None)
    p.add_argument("--snap-dir", default=None)
    p.add_argument("--milestone", type=float, default=0.7)
    p.add_argument("--init", default=None)
    a = p.parse_args()
    train(a.char, steps=a.steps, grid=a.grid, glyph=a.glyph,
          channel_n=a.channels, hidden_n=a.hidden, batch=a.batch,
          damage_n=a.damage,
          out=a.out or f"weights/{ord(a.char):04x}.json",
          snap_dir=a.snap_dir, milestone=a.milestone, init=a.init)
