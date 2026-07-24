"""Vanilla Growing-NCA on emoji targets — the classic recipe, no tricks.

Seed-grown, full RGBA supervision (an emoji IS its colors), sample pool,
damage training, fire rate 0.5. hidden_n defaults to 96 (not 128) so two
models can be stepped per-cell-blended in the browser coexistence demo at
interactive rates. Weights export continuously, so the web engine can run
each organism as soon as training starts.
"""
import argparse
import io
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from nca.model import NCA, to_rgba
from nca.train_web_hidden import damage_mask_rect
from nca.checkpoint import save_checkpoint, try_resume
from nca.runmeta import RunMeta, export_run_weights
from nca.rollout import fester


def emoji_rgba(code, H=64, W=64, size=44):
    url = ("https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/72x72/"
           f"{code}.png")
    with urllib.request.urlopen(url) as r:
        img = Image.open(io.BytesIO(r.read())).convert("RGBA")
    img = img.resize((size, size), Image.LANCZOS)
    a = np.asarray(img, np.float32) / 255.0
    out = np.zeros((4, H, W), np.float32)
    y0, x0 = (H - size) // 2, (W - size) // 2
    for c in range(3):
        out[c, y0:y0 + size, x0:x0 + size] = a[..., c] * a[..., 3]
    out[3, y0:y0 + size, x0:x0 + size] = a[..., 3]
    return out


def train(emoji="1f642", label=None, steps=8000, channel_n=16, hidden_n=96,
          batch=16, pool_size=256, lr=2e-3, ca_min=64, ca_max=96,
          damage_p=0.3, damage_n=2, fester_p=0.0, rng_seed=0,
          log_every=200, ckpt_every=500, snap_dir=None):
    label = label or emoji
    torch.manual_seed(sum(map(ord, emoji)) + rng_seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device {device}, emoji {emoji} ({label})")

    tgt_np = emoji_rgba(emoji)
    _, h, w = tgt_np.shape
    target = torch.from_numpy(tgt_np)[None].repeat(batch, 1, 1, 1).to(device)

    model = NCA(channel_n, fire_rate=0.5, hidden_n=hidden_n).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=[int(steps * 0.85)], gamma=0.1)

    seed = torch.zeros(1, channel_n, h, w, device=device)
    seed[:, 3:, h // 2, w // 2] = 1.0
    pool = seed.repeat(pool_size, 1, 1, 1)

    start_step, _ = try_resume(snap_dir, model, opt, sched, device=device)

    if snap_dir:
        Path(snap_dir).mkdir(parents=True, exist_ok=True)
        vis = (1 - tgt_np[3] + tgt_np[:3].transpose(1, 2, 0).mean(-1))
        img = (1 - tgt_np[3:4] + tgt_np[:3]).clip(0, 1).transpose(1, 2, 0)
        Image.fromarray((img * 255).astype(np.uint8)) \
            .resize((w * 6, h * 6), Image.NEAREST).save(Path(snap_dir) / "target.png")
    meta = RunMeta(snap_dir, label.upper(), "nca.train_emoji_vanilla",
                   {"emoji": emoji, "steps": steps, "batch": batch, "lr": lr,
                    "hidden_n": hidden_n, "rng_seed": rng_seed},
                   channel_n, hidden_n, "single", steps, device,
                   tags=["emoji", "vanilla", label])

    t0 = time.time()
    for step in range(start_step, steps):
        idx = torch.randperm(pool_size, device=device)[:batch]
        x = pool[idx]
        with torch.no_grad():
            rank = F.mse_loss(to_rgba(x), target, reduction="none") \
                .mean(dim=(1, 2, 3)).argsort(descending=True)
        x = x[rank]; idx = idx[rank]
        x[:1] = seed
        if torch.rand(1).item() < damage_p:
            nd = int(torch.randint(1, damage_n + 1, (1,)))
            m = damage_mask_rect(nd, h, w, device)
            x[-nd:] = x[-nd:] * m
        if fester_p > 0 and torch.rand(1).item() < fester_p:
            x = fester(model, x,
                       damage_fn=lambda z: z * damage_mask_rect(z.shape[0], h, w, device))
        # most-damaged/festered input state, snapshotted as START_ so the
        # damage->recovery training is visible, not just its endpoint
        x_start = x[-1:].detach().clone()

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
        with torch.no_grad():
            pool[idx] = x.detach()

        if step % log_every == 0 or step == steps - 1:
            print(f"[vanilla-{label}] step {step} loss {loss.item():.5f} "
                  f"({time.time() - t0:.1f}s)", flush=True)
            if snap_dir:
                s = f"{step:05d}"
                for tag, t in [("COMP", to_rgba(x)[-1]), ("START", to_rgba(x_start)[0])]:
                    img = t.detach().cpu().clamp(0, 1)
                    vis = (1 - img[3:4] + img[:3]).clamp(0, 1).permute(1, 2, 0).numpy()
                    Image.fromarray((vis * 255).astype(np.uint8)) \
                        .resize((w * 6, h * 6), Image.NEAREST) \
                        .save(Path(snap_dir) / f"{tag}_{s}.png")
                torch.save(model.state_dict(), str(Path(snap_dir) / "latest.pth"))
                meta.log(step, loss.item())
                export_run_weights(model, snap_dir, label.upper(), 12,
                                   grid_w=w, grid_h=h)
        if snap_dir and (step % ckpt_every == 0 or step == steps - 1):
            save_checkpoint(snap_dir, step, model, opt, sched)

    print(f"Final loss: {loss.item():.5f}")
    return model


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--emoji", default="1f642")
    p.add_argument("--label", default=None)
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--damage-p", type=float, default=0.3)
    p.add_argument("--damage-n", type=int, default=2)
    p.add_argument("--fester-p", type=float, default=0.0)
    p.add_argument("--rng-seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--snap-dir", default=None)
    a = p.parse_args()
    train(emoji=a.emoji, label=a.label, steps=a.steps, damage_p=a.damage_p,
          damage_n=a.damage_n, fester_p=a.fester_p, rng_seed=a.rng_seed,
          log_every=a.log_every, snap_dir=a.snap_dir)
