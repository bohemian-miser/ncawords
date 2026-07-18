"""Staged curriculum: grow -> restore (fisheye+noise) -> heal holes -> negotiate.

Stage 1  Normal seed/pool growth toward the word with a 3-line scaffold
         (the three straight strands through the letter band).
Stage 2  Seed injection off. Grown states are corrupted with a mild
         fisheye magnification plus noise; train to contract back.
         Advances when the moving-average loss gets good (or at a cap).
Stage 3  Circular holes punched into grown states; train to heal.
Stage 4  Negotiation on a double-size canvas: one partial word
         surrounded by smaller partial rivals; the big one completes,
         the rivals dissolve.

All stages train the SAME model — one set of local rules that can grow,
restore, heal, and negotiate.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from nca.model import NCA, to_rgba
from nca.train import FONT_PATH, char_color
from nca.checkpoint import save_checkpoint, try_resume
from nca.runmeta import RunMeta, export_run_weights

PITCH, MARGIN, GRID_H = 14, 6, 20


def render_word_3_line(text, glyph=12):
    """Word with exactly three straight strands: top, center, bottom."""
    w = MARGIN * 2 + PITCH * len(text)
    h = GRID_H
    font = ImageFont.truetype(FONT_PATH, glyph)
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    boxes = []
    for i, ch in enumerate(text):
        xc = MARGIN + PITCH * i + PITCH // 2
        l, t, r, b = draw.textbbox((0, 0), ch, font=font)
        x = xc - (r - l) / 2 - l
        y = (h - (b - t)) / 2 - t
        boxes.append((x + l, y + t, x + r, y + b, ch, x, y))
    top = min(b[1] for b in boxes)
    bot = max(b[3] for b in boxes)
    mid = (top + bot) / 2
    x0, x1 = boxes[0][0], boxes[-1][2]
    for yy in (top, mid, bot):
        draw.line((x0, yy, x1, yy), fill=(128, 128, 128, 64))
    for (L, T, R, B, ch, x, y) in boxes:
        draw.text((x, y), ch, font=font, fill=char_color(ch) + (255,))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr[..., :3] *= arr[..., 3:]
    return arr.transpose(2, 0, 1)  # [4,h,w]


def make_seed(tgt, channel_n):
    _, h, w = tgt.shape
    x = torch.zeros(1, channel_n, h, w)
    ys, xs = np.where(tgt[3] > 0.5)
    cy, cx = h // 2, w // 2
    if len(ys):
        i = np.argmin((ys - cy) ** 2 + (xs - cx) ** 2)
        cy, cx = ys[i], xs[i]
    x[:, 3:, cy, cx] = 1.0
    return x


def fisheye(x, k):
    """Magnify the center by k (0..~0.4) via grid_sample, all channels."""
    B, C, H, W = x.shape
    ys = torch.linspace(-1, 1, H, device=x.device)
    xs = torch.linspace(-1, 1, W, device=x.device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    r2 = gx ** 2 + gy ** 2
    scale = 1.0 - k * (1.0 - r2).clamp(min=0)
    grid = torch.stack([gx * scale, gy * scale], dim=-1)[None].expand(B, H, W, 2)
    return F.grid_sample(x, grid, align_corners=False, padding_mode="zeros")


def holes(x, n, rng):
    B, C, H, W = x.shape
    yy, xx = torch.meshgrid(torch.arange(H, device=x.device),
                            torch.arange(W, device=x.device), indexing="ij")
    mask = torch.ones(H, W, device=x.device)
    for _ in range(n):
        cy = rng.uniform(0.15, 0.85) * H
        cx = rng.uniform(0.15, 0.85) * W
        r = rng.uniform(2.0, 5.0)
        mask = mask * (((yy - cy) ** 2 + (xx - cx) ** 2) > r * r).float()
    return x * mask[None, None]


def reveal_order(tgt):
    ys, xs = np.where(tgt[3] > 0.05)
    cy, cx = ys.mean(), xs.mean()
    order = np.argsort((ys - cy) ** 2 + (xs - cx) ** 2)
    return ys[order], xs[order]


def partial_np(tgt, ys, xs, frac):
    out = np.zeros_like(tgt)
    k = int(len(ys) * np.clip(frac, 0, 1))
    if k:
        out[:, ys[:k], xs[:k]] = tgt[:, ys[:k], xs[:k]]
    return out


def train(text="COMP", steps=14000, glyph=12, channel_n=16, hidden_n=80,
          batch=16, pool_size=256, lr=2e-3, ca_min=48, ca_max=72,
          gate=0.012, replay_p=0.0, log_every=100, ckpt_every=500,
          snap_dir=None, rng_seed=0):
    torch.manual_seed(sum(map(ord, text)) + 3)
    rng = np.random.default_rng(rng_seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")

    tgt_np = render_word_3_line(text, glyph)
    _, h, w = tgt_np.shape
    target = torch.from_numpy(tgt_np)[None].repeat(batch, 1, 1, 1).to(device)
    ys_o, xs_o = reveal_order(tgt_np)

    # stage-4 canvas: double size
    H2, W2 = h * 3, w * 2
    caps = {1: int(steps * 0.25), 2: int(steps * 0.25),
            3: int(steps * 0.15), 4: steps}   # stage 4 runs to the end

    if snap_dir:
        Path(snap_dir).mkdir(parents=True, exist_ok=True)
        vis = (1 - tgt_np[3] + tgt_np[:3]).clip(0, 1).transpose(1, 2, 0)
        Image.fromarray((vis * 255).astype(np.uint8)) \
            .resize((w * 8, h * 8), Image.NEAREST).save(Path(snap_dir) / "target.png")

    model = NCA(channel_n, hidden_n=hidden_n).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=[int(steps * 0.85)], gamma=0.1)

    seed = make_seed(tgt_np, channel_n).to(device)
    pool = seed.repeat(pool_size, 1, 1, 1)

    start_step, ckpt_extra = try_resume(snap_dir, model, opt, sched, device=device)
    stage = ckpt_extra.get("stage", 1) if ckpt_extra else 1
    stage_start = ckpt_extra.get("stage_start", 0) if ckpt_extra else 0

    meta = RunMeta(snap_dir, text, "nca.train_staged",
                   {"steps": steps, "glyph": glyph, "batch": batch, "lr": lr,
                    "gate": gate, "replay_p": replay_p, "rng_seed": rng_seed},
                   channel_n, hidden_n, "staged", steps, device)

    recent = []
    t0 = time.time()
    for step in range(start_step, steps):
        # Rehearse earlier stages so later ones don't overwrite them
        # (the four-skill exam showed stage-4 training erased stage-1 growth).
        active = stage
        if replay_p > 0 and stage > 1 and rng.random() < replay_p:
            active = int(rng.integers(1, stage))
        if active == 1:
            idx = torch.randperm(pool_size, device=device)[:batch]
            x = pool[idx]
            with torch.no_grad():
                rank = F.mse_loss(to_rgba(x), target, reduction="none") \
                    .mean(dim=(1, 2, 3)).argsort(descending=True)
            x = x[rank]; idx = idx[rank]
            x[:1] = seed
            tgt_b = target
        elif active == 2:
            idx = torch.randperm(pool_size, device=device)[:batch]
            x = pool[idx].clone()
            k = float(rng.uniform(0.08, 0.35))
            x = fisheye(x, k)
            namp = float(rng.uniform(0.05, 0.30))
            x[:, :4] = x[:, :4] * (1 - namp) + torch.rand_like(x[:, :4]) * namp
            tgt_b = target
        elif active == 3:
            idx = torch.randperm(pool_size, device=device)[:batch]
            x = pool[idx].clone()
            x = holes(x, int(rng.integers(2, 5)), rng)
            tgt_b = target
        else:  # stage 4: negotiation scenes on the big canvas
            inps, tgts = [], []
            for _ in range(batch):
                inp = np.zeros((4, H2, W2), np.float32)
                tg = np.zeros((4, H2, W2), np.float32)
                fmain = rng.uniform(0.45, 0.8)
                my = int(rng.integers(0, H2 - h)); mx = int(rng.integers(0, W2 - w))
                pm = partial_np(tgt_np, ys_o, xs_o, fmain)
                inp[:, my:my+h, mx:mx+w] = np.maximum(inp[:, my:my+h, mx:mx+w], pm)
                adv = partial_np(tgt_np, ys_o, xs_o, min(1.0, fmain + 0.2))
                tg[:, my:my+h, mx:mx+w] = np.maximum(tg[:, my:my+h, mx:mx+w], adv)
                for _ in range(int(rng.integers(1, 4))):
                    fr = rng.uniform(0.1, fmain - 0.15)
                    ry = int(rng.integers(0, H2 - h)); rx = int(rng.integers(0, W2 - w))
                    if abs(ry - my) < h * 0.8 and abs(rx - mx) < w * 0.8:
                        continue
                    pr = partial_np(tgt_np, ys_o, xs_o, fr)
                    inp[:, ry:ry+h, rx:rx+w] = np.maximum(inp[:, ry:ry+h, rx:rx+w], pr)
                    sh = partial_np(tgt_np, ys_o, xs_o, max(0.0, fr - 0.2))
                    tg[:, ry:ry+h, rx:rx+w] = np.maximum(tg[:, ry:ry+h, rx:rx+w], sh)
                inps.append(inp); tgts.append(tg)
            x = torch.zeros(batch, channel_n, H2, W2, device=device)
            x[:, :4] = torch.from_numpy(np.stack(inps)).to(device)
            hn = torch.rand(batch, channel_n - 4, H2, W2, device=device) * 0.1
            x[:, 4:] = hn * (x[:, 3:4] > 0.05)
            tgt_b = torch.from_numpy(np.stack(tgts)).to(device)
            idx = None

        x_start = x[-1:].detach().clone()
        n_ca = int(torch.randint(ca_min, ca_max + 1, (1,)))
        x = model(x, steps=n_ca)
        loss = F.mse_loss(to_rgba(x), tgt_b)

        opt.zero_grad()
        loss.backward()
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    p.grad /= (p.grad.norm() + 1e-8)
        opt.step()
        sched.step()
        if active == 1:
            with torch.no_grad():
                pool[idx] = x.detach()

        if active == stage:
            recent.append(loss.item())
        if len(recent) > 50:
            recent.pop(0)
        avg = sum(recent) / len(recent) if recent else float("inf")
        in_stage = step - stage_start
        if stage < 4 and ((len(recent) == 50 and avg < gate) or in_stage >= caps[stage]):
            print(f"=== stage {stage} done at step {step} (avg {avg:.4f}) ===", flush=True)
            stage += 1
            stage_start = step
            recent.clear()

        if step % log_every == 0 or step == steps - 1:
            print(f"[staged_{text}] step {step} stage {stage} loss {loss.item():.5f} "
                  f"avg {avg:.5f} ({time.time() - t0:.1f}s)", flush=True)
            if snap_dir:
                s = f"{step:05d}"
                for tag, t in [("START", to_rgba(x_start)[0]), ("COMP", to_rgba(x)[-1]),
                               ("TARGET", tgt_b[-1] if tgt_b.dim() == 4 else tgt_b)]:
                    img = t.detach().cpu().clamp(0, 1)
                    vis = (1 - img[3:4] + img[:3]).clamp(0, 1).permute(1, 2, 0).numpy()
                    up = 8 if vis.shape[0] <= 32 else 4
                    Image.fromarray((vis * 255).astype(np.uint8)) \
                        .resize((vis.shape[1] * up, vis.shape[0] * up), Image.NEAREST) \
                        .save(Path(snap_dir) / f"{tag}_{s}.png")
                torch.save(model.state_dict(), str(Path(snap_dir) / "latest.pth"))
                meta.log(step, loss.item(), stage=stage)
                export_run_weights(model, snap_dir, text, glyph,
                                   grid_w=W2, grid_h=H2)
        if snap_dir and (step % ckpt_every == 0 or step == steps - 1):
            save_checkpoint(snap_dir, step, model, opt, sched,
                            extra={"stage": stage, "stage_start": stage_start})

    print(f"Final: stage {stage}, loss {loss.item():.5f}")
    return model


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--text", default="COMP")
    p.add_argument("--steps", type=int, default=14000)
    p.add_argument("--gate", type=float, default=0.012)
    p.add_argument("--replay-p", type=float, default=0.0)
    p.add_argument("--rng-seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--snap-dir", default=None)
    a = p.parse_args()
    train(a.text, steps=a.steps, gate=a.gate, replay_p=a.replay_p,
          rng_seed=a.rng_seed, log_every=a.log_every, snap_dir=a.snap_dir)
