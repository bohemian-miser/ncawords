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


def ink_seed_pos(target, region=None):
    """Pick a seed cell that sits ON the glyph, as central as possible.

    target: [4, H, W] premultiplied RGBA. region: optional (x0, x1) column
    range to search within (used for per-character slots in word models).

    Seeding on background is fatal (see make_seed), so choose among ink
    pixels. Prefer ink that is far from the glyph's edge (the thick part of
    a stroke) and close to the region's center, so growth starts from solid
    tissue rather than a fragile antialiased rim.
    """
    alpha = target[3].numpy() if hasattr(target[3], "numpy") else target[3]
    h, w = alpha.shape
    x0, x1 = region if region is not None else (0, w)
    ink = alpha[:, x0:x1] > 0.5
    if not ink.any():
        return ((x0 + x1) // 2, h // 2)

    # Distance to the nearest non-ink pixel: how deep inside a stroke we are.
    depth = np.zeros_like(ink, dtype=np.float32)
    for dy in range(-3, 4):
        for dx in range(-3, 4):
            shifted = np.roll(np.roll(ink, dy, axis=0), dx, axis=1)
            depth += shifted.astype(np.float32)

    yy, xx = np.mgrid[0:h, 0:(x1 - x0)]
    cy, cx = h / 2.0, (x1 - x0) / 2.0
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    # Rank ink pixels: deep strokes first, then proximity to center.
    score = np.where(ink, depth - 0.35 * dist, -1e9)
    iy, ix = np.unravel_index(int(score.argmax()), score.shape)
    return (int(ix + x0), int(iy))


class SamplePool:
    def __init__(self, seed, size=256):
        self.pool = seed.repeat(size, 1, 1, 1)

    def sample(self, n):
        idx = torch.randperm(self.pool.shape[0])[:n]
        return idx, self.pool[idx].clone()

    def commit(self, idx, x):
        self.pool[idx] = x.detach()


def damage_mask(n, h, w=None, device="cpu", r_frac=(0.20, 0.55)):
    """1 outside a random circle, 0 inside (multiply to damage).

    Works in PIXEL space so it is correct on non-square grids: a word model is
    216x32, and building the disc in normalized square coords put nearly every
    circle off the strip — i.e. no damage at all, which quietly disables the
    regeneration training this is here to do.

    Centers land anywhere on the grid; radius is a fraction of the SHORT side
    (the glyph height), so a hole always takes a real bite out of a character.
    """
    if w is None:
        w = h
    yy, xx = torch.meshgrid(
        torch.arange(h, device=device, dtype=torch.float32),
        torch.arange(w, device=device, dtype=torch.float32), indexing="ij")
    cx = torch.rand(n, 1, 1, device=device) * w
    cy = torch.rand(n, 1, 1, device=device) * h
    lo, hi = r_frac
    r = (torch.rand(n, 1, 1, device=device) * (hi - lo) + lo) * min(h, w)
    mask = ((xx[None] - cx) ** 2 + (yy[None] - cy) ** 2 >= r ** 2).float()
    return mask[:, None]


def is_dead(model, grid, channel_n, device, probe_steps=64, pos=None):
    """True if growing from a fresh seed leaves no living cell.

    Once every cell is dead the state is all zeros, the update is computed
    from zeros, and the gradient path is severed — the run can never recover,
    so it is worth detecting rather than burning the remaining steps.
    """
    with torch.no_grad():
        x = make_seed(grid, channel_n, pos=pos).to(device)
        x = model(x, steps=probe_steps)
        return x[:, 3].max().item() <= 0.1


def train(ch, steps=1000, grid=32, glyph=22, channel_n=12, hidden_n=64,
          batch=8, pool_size=256, lr=2e-3, damage_n=3, ca_min=36, ca_max=56,
          device="cpu", log_every=100, out=None, snap_dir=None,
          milestone=0.7, init=None, max_restarts=3, damage_start=0.3):
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

    seed_pos = ink_seed_pos(target[0])
    seed = make_seed(grid, channel_n, pos=seed_pos)
    pool = SamplePool(seed, pool_size)
    print(f"[{ch}] seed at {seed_pos} (grid center is "
          f"{(grid // 2, grid // 2)})", flush=True)

    # An all-dead grid outputs zeros, so its loss is exactly mean(target^2).
    # That state is absorbing (dead cells zero the gradient path), so if we
    # land in it, no amount of further training escapes: re-init and retry.
    dead_loss = (target ** 2).mean().item()
    restarts = 0

    t0 = time.time()
    step = 0
    while step < steps:
        idx, x = pool.sample(batch)
        x = x.to(device)
        # Rank by loss: worst sample is replaced with the seed (keeps the
        # pool anchored to growth from scratch); best get damaged.
        with torch.no_grad():
            loss_rank = F.mse_loss(to_rgba(x), target, reduction="none") \
                .mean(dim=(1, 2, 3)).argsort(descending=True)
        x = x[loss_rank]
        x[:1] = seed.to(device)
        # Damage only once growth is established; carving holes in seeds from
        # step 0 pushes alpha under the alive threshold and invites death.
        # After that, the lowest-loss (best grown) samples get holes punched
        # in them and the model is graded on the repair — this is the paper's
        # "Regenerating" regime, and it is what makes the pattern robust.
        if damage_n and step > steps * damage_start:
            x[-damage_n:] *= damage_mask(damage_n, grid, grid, device)

        n_ca = int(torch.randint(ca_min, ca_max + 1, (1,)))
        x = model(x, steps=n_ca)
        loss = F.mse_loss(to_rgba(x), target)

        # Cheap gate: an all-dead batch sits exactly at dead_loss. Only pay
        # for the rollout probe when the loss looks like death.
        if loss.item() >= dead_loss * 0.995 and step > 50 \
                and is_dead(model, grid, channel_n, device, pos=seed_pos):
            if restarts >= max_restarts:
                print(f"[{ch}] DEAD and out of restarts — giving up", flush=True)
                break
            restarts += 1
            print(f"[{ch}] died at step {step}; restart {restarts}"
                  f"/{max_restarts} with a fresh init", flush=True)
            torch.manual_seed(ord(ch) + 1234 + restarts * 7919)
            model = NCA(channel_n, hidden_n=hidden_n).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=lr)
            sched = torch.optim.lr_scheduler.MultiStepLR(
                opt, milestones=[int(steps * milestone)], gamma=0.1)
            pool = SamplePool(seed, pool_size)
            step = 0
            continue

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
                                Path(snap_dir) / f"{ord(ch):04x}_{step:05d}.png",
                                pos=seed_pos)
        step += 1

    if out:
        export_weights(model, ch, grid, glyph, Path(out), seed_pos=seed_pos)
    return model


def save_growth_png(model, grid, channel_n, path, n_steps=80, pos=None):
    from nca.model import to_rgb
    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        x = make_seed(grid, channel_n, pos=pos)
        x = model(x, steps=n_steps)
        img = to_rgb(x)[0].clamp(0, 1).permute(1, 2, 0).numpy()
    Image.fromarray((img * 255).astype(np.uint8)).save(path)


def export_weights(model, ch, grid, glyph, path, seed_pos=None):
    """Export weights as JSON for browser inference.

    fc0 rows are reordered from PyTorch's interleaved perception layout
    [id_c0, sx_c0, sy_c0, id_c1, ...] to blocked [all-id | all-sx | all-sy],
    which is the natural order for a from-scratch JS/GLSL implementation.

    The seed position is exported as a one-entry `seeds` list (the same field
    the word models use, so the browser engine needs no special case). It is
    not cosmetic: the model is only trained to grow from THAT cell.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    c = model.channel_n
    w0 = model.fc0.weight.detach().squeeze(-1).squeeze(-1).numpy()  # [H, 3C]
    w0 = w0.reshape(-1, c, 3).transpose(0, 2, 1).reshape(-1, 3 * c)
    if seed_pos is None:
        seed_pos = (grid // 2, grid // 2)
    d = {
        "char": ch,
        "grid": grid,
        "glyph": glyph,
        "channel_n": c,
        "hidden_n": model.fc0.out_channels,
        "fire_rate": model.fire_rate,
        "layout": "blocked",  # perception vector = [state | sobel_x | sobel_y]
        "seeds": [{"x": int(seed_pos[0]), "y": int(seed_pos[1]),
                   "code": [], "char": ch}],
        "code_ch0": 4,
        "code_bits": 0,
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
