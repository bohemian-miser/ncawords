"""Bloom: replace the scaffold with an emergent 'success cloud'.

The scaffold (pre-drawn guide strands) is a cheat — it hands the model
the letter positions. Bloom tests whether formed letters can instead
grow their OWN positional guides, so a word completes from a bare seed
with no scaffold. Two coupled mechanisms, applied between CA steps:

  1. Success cloud (morphogen): diffuse the alpha field outward from
     whatever structure exists and expose it in a dedicated channel.
     A void cell reads the cloud as 'you are near finished structure —
     grow'. This is the long-range positional relay the bare recipe
     lacks (letters too far apart for signal to bridge in one rollout).

  2. Living mist (void randomness): step() zeroes dead cells, so the
     cloud cannot survive in empty void. In the cloud's halo — the ring
     just outside existing structure — inject a little alpha + hidden
     noise, keeping those cells marginally alive so the cloud persists
     and stochastic nucleation can seed the next letter.

Injection runs only for the first `explore_frac` of the rollout, then a
clean tail lets the model consolidate and retract stray mist before the
loss is read (explore, THEN settle). Loss is alpha-only (RGB free), with
a void-weighted term (warmed up) that upweights target-on cells still dark.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from nca.model import NCA, to_rgba
from nca.train_staged import render_word_3_line, make_seed
from nca.train_web_hidden import render_word_9_line, damage_mask_rect
from nca.checkpoint import save_checkpoint, try_resume
from nca.runmeta import RunMeta, export_run_weights
from nca.rollout import fester


def diffuse(field, iters):
    """Spread a field outward with repeated 3x3 box blur (cheap morphogen)."""
    for _ in range(iters):
        field = F.avg_pool2d(field, 3, stride=1, padding=1)
    return field


def bloom_rollout(model, x, n_steps, morph_ch, diffuse_iters,
                  mist_alpha, bloom_noise, halo_lo, halo_hi, explore_frac,
                  fire_rate=0.5):
    """Manual rollout with cloud + mist injection during the explore phase."""
    explore_steps = int(n_steps * explore_frac)
    for t in range(n_steps):
        x = model.step(x, fire_rate)
        if t < explore_steps:
            alpha = x[:, 3:4].clamp(0, 1)
            cloud = diffuse(alpha, diffuse_iters)
            # halo = ring around structure that is currently void
            halo = ((cloud > halo_lo) & (cloud < halo_hi) &
                    (alpha < 0.1)).float()
            # keep the halo marginally alive so the cloud persists there
            x = x + torch.cat([
                torch.zeros_like(x[:, :3]),
                halo * mist_alpha,                                 # alpha bump
                torch.randn_like(x[:, 4:]) * bloom_noise * halo,   # hidden mist
            ], dim=1)
            # expose the cloud as a readable channel (overwrite, non-gameable)
            x = x.clone()
            x[:, morph_ch:morph_ch + 1] = cloud
    return x


def train(text="COMP", steps=12000, glyph=12, channel_n=16, hidden_n=128,
          batch=16, pool_size=256, lr=1e-3, ca_min=64, ca_max=96,
          scaffold="none", bloom=True, diffuse_iters=6, mist_alpha=0.15,
          bloom_noise=0.05, halo_lo=0.02, halo_hi=0.4, explore_frac=0.7,
          void_w=4.0, warmup=2000, damage_p=0.3, fire_rate=0.7, fester_p=0.0,
          rng_seed=0, log_every=200, ckpt_every=500, snap_dir=None):
    torch.manual_seed(sum(map(ord, text)) + 23 + rng_seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}, scaffold {scaffold}, bloom {bloom}, "
          f"fire {fire_rate}, fester {fester_p}")

    if scaffold == "3line":
        tgt_np = render_word_3_line(text, glyph)
    else:
        tgt_np = render_word_9_line(text, glyph, char_alpha=255, strand_alpha=0)
    _, h, w = tgt_np.shape
    tgt_a = torch.from_numpy(tgt_np[3:4])[None].repeat(batch, 1, 1, 1).to(device)
    morph_ch = channel_n - 1   # last hidden channel carries the cloud

    if snap_dir:
        Path(snap_dir).mkdir(parents=True, exist_ok=True)
        Image.fromarray(((1 - tgt_np[3]) * 255).astype(np.uint8)) \
            .resize((w * 8, h * 8), Image.NEAREST).save(Path(snap_dir) / "target.png")

    model = NCA(channel_n, hidden_n=hidden_n).to(device)
    model.fire_rate = fire_rate   # star-sweep winner (0.7); used by both paths
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=[int(steps * 0.85)], gamma=0.1)

    seed = make_seed(tgt_np, channel_n).to(device)
    pool = seed.repeat(pool_size, 1, 1, 1)

    start_step, _ = try_resume(snap_dir, model, opt, sched, device=device)

    meta = RunMeta(snap_dir, text, "nca.train_bloom",
                   {"steps": steps, "batch": batch, "lr": lr,
                    "scaffold": scaffold, "bloom": bloom,
                    "diffuse_iters": diffuse_iters, "mist_alpha": mist_alpha,
                    "bloom_noise": bloom_noise, "explore_frac": explore_frac,
                    "void_w": void_w, "rng_seed": rng_seed},
                   channel_n, hidden_n, "single", steps, device,
                   tags=["bloom"] + ([scaffold] if scaffold != "none" else []))

    t0 = time.time()
    for step in range(start_step, steps):
        idx = torch.randperm(pool_size, device=device)[:batch]
        x = pool[idx]
        with torch.no_grad():
            rank = F.mse_loss(x[:, 3:4], tgt_a, reduction="none") \
                .mean(dim=(1, 2, 3)).argsort(descending=True)
        x = x[rank]; idx = idx[rank]
        x[:1] = seed
        if torch.rand(1).item() < damage_p:
            m = damage_mask_rect(2, h, w, device)
            x[-2:] = x[-2:] * m

        if fester_p > 0 and torch.rand(1).item() < fester_p:
            x = fester(model, x,
                       damage_fn=lambda z: z * damage_mask_rect(z.shape[0], h, w, device))

        n_ca = int(torch.randint(ca_min, ca_max + 1, (1,)))
        if bloom:
            x = bloom_rollout(model, x, n_ca, morph_ch, diffuse_iters,
                              mist_alpha, bloom_noise, halo_lo, halo_hi,
                              explore_frac, fire_rate=fire_rate)
        else:
            x = model(x, steps=n_ca)

        # void-weighted alpha loss: upweight target-on cells still dark,
        # ramp the weight in to avoid the absorbing-dead collapse
        vw = void_w * min(1.0, step / max(1, warmup))
        wmap = 1.0 + vw * (tgt_a > 0.5).float() * (x[:, 3:4] < 0.3).float()
        loss = (wmap * (x[:, 3:4] - tgt_a) ** 2).mean()

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
            print(f"[bloom-{text}-{scaffold}] step {step} loss {loss.item():.5f} "
                  f"({time.time() - t0:.1f}s)", flush=True)
            if snap_dir:
                s = f"{step:05d}"
                img = to_rgba(x)[0].detach().cpu().clamp(0, 1)
                vis = (1 - img[3:4] + img[:3]).clamp(0, 1).permute(1, 2, 0).numpy()
                Image.fromarray((vis * 255).astype(np.uint8)) \
                    .resize((w * 8, h * 8), Image.NEAREST) \
                    .save(Path(snap_dir) / f"COMP_{s}.png")
                a = img[3].numpy()
                Image.fromarray(((1 - a) * 255).astype(np.uint8)) \
                    .resize((w * 8, h * 8), Image.NEAREST) \
                    .save(Path(snap_dir) / f"ALPHA_{s}.png")
                # the success cloud itself
                cl = diffuse(x[:, 3:4].clamp(0, 1), diffuse_iters)[0, 0] \
                    .detach().cpu().numpy()
                Image.fromarray(((1 - cl / (cl.max() + 1e-8)) * 255).astype(np.uint8)) \
                    .resize((w * 8, h * 8), Image.NEAREST) \
                    .save(Path(snap_dir) / f"CLOUD_{s}.png")
                torch.save(model.state_dict(), str(Path(snap_dir) / "latest.pth"))
                meta.log(step, loss.item())
                export_run_weights(model, snap_dir, text, glyph,
                                   grid_w=w + 20, grid_h=h + 10)
        if snap_dir and (step % ckpt_every == 0 or step == steps - 1):
            save_checkpoint(snap_dir, step, model, opt, sched)

    print(f"Final loss: {loss.item():.5f}")
    return model


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--text", default="COMP")
    p.add_argument("--steps", type=int, default=12000)
    p.add_argument("--scaffold", default="none", choices=["none", "3line"])
    p.add_argument("--no-bloom", dest="bloom", action="store_false")
    p.add_argument("--bloom-noise", type=float, default=0.05)
    p.add_argument("--mist-alpha", type=float, default=0.15)
    p.add_argument("--explore-frac", type=float, default=0.7)
    p.add_argument("--void-w", type=float, default=4.0)
    p.add_argument("--fire-rate", "--fire_rate", type=float, default=0.7)
    p.add_argument("--fester-p", type=float, default=0.0)
    p.add_argument("--rng-seed", "--rng_seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--snap-dir", default=None)
    a = p.parse_args()
    train(a.text, steps=a.steps, scaffold=a.scaffold, bloom=a.bloom,
          bloom_noise=a.bloom_noise, mist_alpha=a.mist_alpha,
          explore_frac=a.explore_frac, void_w=a.void_w,
          fire_rate=a.fire_rate, fester_p=a.fester_p, rng_seed=a.rng_seed,
          log_every=a.log_every, snap_dir=a.snap_dir)
