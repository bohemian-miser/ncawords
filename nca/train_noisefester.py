"""Continue-training side quest: noise x fester robustness curriculum.

Loads an existing trained word model (default cls-fan3-r1) and keeps
training it through a staged schedule:

  warm      damage + fester on the trained model (baseline hardening)
  n<a>      batch states blended with uniform noise, fraction a:
            x <- (1-a)*x + a*U(0,1), train recovery to target
  n<a>f<N>  same noised start, then EXACTLY N no-grad fester steps
            (half the time with damage) before the graded rollout

Noise levels: 0.05%, 1%, 2%, 5%, 10%, then 15% increments to 100%.
Fester durations: 10, 50, 100, 200, 500. Every phase logs (noise,
fester_n, loss) so the robustness surface falls out of one run.
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
from nca.train_staged import render_word_3_line, render_word_3_line_fan, make_seed
from nca.train_web_hidden import damage_mask_rect
from nca.checkpoint import save_checkpoint, try_resume
from nca.runmeta import RunMeta, export_run_weights
from nca.rollout import fester

from nca.fleetconfig import bucket_url

BUCKET = bucket_url()
LEVELS = [0.0005, 0.01, 0.02, 0.05, 0.10,
          0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.00]
FESTERS = [10, 50, 100, 200, 500]



def mixed_corrupt(x, model, target_shape, step, steps, device, rng_state=None):
    """Per-sample mixed corruption with a ramping severity ceiling.

    Every sample in the batch gets its own noise level, damage count and
    fester duration, so one batch spans clean-ish to obliterated. The
    ceiling rises with training: early batches are mostly light, late
    batches routinely include pure-noise and long-fester samples.
    Returns (x, info) where info summarises the mix for logging.
    """
    B, C, h, w = x.shape
    frac = min(1.0, step / max(1, steps * 0.75))
    ceil_noise = 0.05 + 0.95 * frac          # 5% -> 100%
    max_dmg = 1 + int(6 * frac)              # 1 -> 7 boxes
    max_fest = int(20 + 580 * frac)          # 20 -> 600 steps

    # per-sample noise, biased low but reaching the ceiling
    a = (torch.rand(B, 1, 1, 1, device=device) ** 1.5) * ceil_noise
    # 12% of samples: start from PURE noise regardless of ceiling
    pure = (torch.rand(B, 1, 1, 1, device=device) < 0.12).float()
    a = torch.maximum(a, pure)
    x = (1 - a) * x + a * torch.rand_like(x)

    # per-sample multi-box damage of random sizes
    for b in range(B):
        if torch.rand(1).item() < 0.6:
            for _ in range(int(torch.randint(1, max_dmg + 1, (1,)))):
                bh = int(torch.randint(3, max(4, h // 2), (1,)))
                bw = int(torch.randint(3, max(4, w // 2), (1,)))
                y0 = int(torch.randint(0, max(1, h - bh), (1,)))
                x0 = int(torch.randint(0, max(1, w - bw), (1,)))
                x[b, :, y0:y0 + bh, x0:x0 + bw] = 0.0

    # random-duration fester on a random subset, with damage during it
    fest_n = 0
    if torch.rand(1).item() < 0.55:
        fest_n = int(torch.exp(torch.rand(1) * np.log(max_fest / 10.0)).item() * 10)
        k = int(torch.randint(1, B + 1, (1,)))
        idx = torch.randperm(B, device=device)[:k]
        sub = x[idx]
        with torch.no_grad():
            for t in range(fest_n):
                sub = model.step(sub) if hasattr(model, "step") else model(sub, steps=1)
                if t and t % 120 == 0 and torch.rand(1).item() < 0.5:
                    sub = sub * damage_mask_rect(sub.shape[0], h, w, device)
        x = x.clone()
        x[idx] = sub.detach()
    return x, {"ceil": round(float(ceil_noise), 3),
               "mean_noise": round(float(a.mean()), 3),
               "fest": fest_n, "max_dmg": max_dmg}


def build_phases(warm=1000, phase_len=250):
    phases = [("warm", 0.0, 0, warm)]
    for a in LEVELS:
        phases.append((f"n{a:g}", a, 0, phase_len))
        for n in FESTERS:
            phases.append((f"n{a:g}f{n}", a, n, phase_len))
    return phases


def train(source="cls-fan3-r1", text="COMP", scaffold="fan3", mix=False,
          channel_n=16, hidden_n=128,
          batch=16, pool_size=256, lr=1e-3, ca_min=64, ca_max=96,
          damage_p=0.4, phase_len=250, rng_seed=0,
          log_every=100, ckpt_every=500, snap_dir=None):
    torch.manual_seed(900 + rng_seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tgt = (render_word_3_line(text, 12) if scaffold == "3line"
           else render_word_3_line_fan(text, 12))
    _, h, w = tgt.shape
    target = torch.from_numpy(tgt)[None].repeat(batch, 1, 1, 1).to(device)

    model = NCA(channel_n, fire_rate=0.5, hidden_n=hidden_n).to(device)
    with urllib.request.urlopen(f"{BUCKET}/{source}/latest.pth") as r:
        model.load_state_dict(torch.load(io.BytesIO(r.read()),
                                         map_location=device,
                                         weights_only=True))
    print(f"Device {device}: continuing {source}", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    phases = build_phases(phase_len=phase_len)
    steps = 20000 if mix else sum(p[3] for p in phases)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, [int(steps * 0.9)], 0.1)

    seed = make_seed(tgt, channel_n).to(device)
    pool = seed.repeat(pool_size, 1, 1, 1)

    start_step, _ = try_resume(snap_dir, model, opt, sched, device=device)

    if snap_dir:
        Path(snap_dir).mkdir(parents=True, exist_ok=True)
        vis = (1 - tgt[3] + tgt[:3].mean(0)).clip(0, 1)
        Image.fromarray(((1 - tgt[3]) * 255).astype(np.uint8)) \
            .resize((w * 6, h * 6), Image.NEAREST).save(Path(snap_dir) / "target.png")
    meta = RunMeta(snap_dir, text, "nca.train_noisefester",
                   {"source": source, "steps": steps, "phase_len": phase_len,
                    "levels": LEVELS, "festers": FESTERS, "batch": batch,
                    "lr": lr, "damage_p": damage_p, "rng_seed": rng_seed},
                   channel_n, hidden_n, "single", steps, device,
                   tags=["noisefester", "continue", source])

    # phase lookup by cumulative step
    bounds = []
    acc = 0
    for ph in phases:
        bounds.append((acc, acc + ph[3], ph))
        acc += ph[3]

    def phase_at(step):
        for lo, hi, ph in bounds:
            if lo <= step < hi:
                return ph
        return phases[-1]

    t0 = time.time()
    for step in range(start_step, steps):
        name, noise_a, fest_n, _len = phase_at(step)

        idx = torch.randperm(pool_size, device=device)[:batch]
        x = pool[idx]
        with torch.no_grad():
            rank = F.mse_loss(to_rgba(x), target, reduction="none") \
                .mean(dim=(1, 2, 3)).argsort(descending=True)
        x = x[rank]; idx = idx[rank]
        x[:1] = seed
        if torch.rand(1).item() < damage_p:
            m = damage_mask_rect(2, h, w, device)
            x[-2:] = x[-2:] * m
        if mix:
            x, mixinfo = mixed_corrupt(x, model, target.shape, step, steps, device)
            name = f"mix(c{mixinfo['ceil']},f{mixinfo['fest']})"
        else:
            if noise_a > 0:
                x = (1 - noise_a) * x + noise_a * torch.rand_like(x)
            if fest_n > 0:
                x = fester(model, x,
                           damage_fn=lambda z: z * damage_mask_rect(
                               z.shape[0], h, w, device),
                           min_steps=fest_n, max_steps=fest_n)
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
            print(f"[nf-{text}] step {step} phase {name} loss {loss.item():.5f} "
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
                meta.log(step, loss.item(), phase=name,
                         noise=noise_a, fester_n=fest_n, ca_steps=n_ca)
                export_run_weights(model, snap_dir, text, 12,
                                   grid_w=w + 20, grid_h=h + 10)
        if snap_dir and (step % ckpt_every == 0 or step == steps - 1):
            save_checkpoint(snap_dir, step, model, opt, sched)

    print(f"Final loss: {loss.item():.5f}")
    return model


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="cls-fan3-r1")
    p.add_argument("--text", default="COMP")
    p.add_argument("--phase-len", type=int, default=250)
    p.add_argument("--scaffold", default="fan3", choices=["fan3", "3line"])
    p.add_argument("--mix", action="store_true",
                   help="mixed per-sample severity with ramping ceiling")
    p.add_argument("--channel-n", type=int, default=16)
    p.add_argument("--hidden-n", type=int, default=128)
    p.add_argument("--rng-seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--snap-dir", default=None)
    a = p.parse_args()
    train(source=a.source, text=a.text, scaffold=a.scaffold, mix=a.mix,
          channel_n=a.channel_n, hidden_n=a.hidden_n, phase_len=a.phase_len,
          rng_seed=a.rng_seed, log_every=a.log_every, snap_dir=a.snap_dir)
