"""Noise ladder on top of normal seed/pool growth training.

Standard NCA training (sample pool, seed injection, loss-ranked reseeding)
where the TARGET noise level follows the ladder schedule: 90% -> 0% in
10% decrements spread over the run. A fraction of batches are 'normal'
(clean target, no noise) so plain growth is trained throughout. Optional
--damage-occasional applies the circular damage tool to random pool
samples at random steps, like a user scribbling on the playground canvas.

Perf notes vs the older scripts: pool lives on the GPU (no per-step CPU
round-trip), batch 32, and full checkpoints are written every 500 steps
rather than every log interval (ckpt.pth includes the pool and was the
dominant GCS write cost).
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from nca.model import NCA, to_rgba
from nca.train_web_hidden import render_word_9_line, make_single_seed, damage_mask_rect
from nca.checkpoint import save_checkpoint, try_resume
from nca.runmeta import RunMeta, export_run_weights
from nca.rollout import adaptive_rollout


def train(text, steps=8000, glyph=12, channel_n=16, hidden_n=80,
          batch=32, pool_size=256, lr=2e-3, ca_min=64, ca_max=96,
          normal_p=0.25, damage_occasional=False, damage_p=0.3,
          rho_target=0.0, rho_w=0.0, adaptive=False,
          log_every=100, ckpt_every=500, snap_dir=None):
    torch.manual_seed(sum(map(ord, text)) + 55)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")

    tgt = render_word_9_line(text, glyph, char_alpha=255, strand_alpha=0)
    target = torch.from_numpy(tgt)[None].repeat(batch, 1, 1, 1).to(device)
    h, w = tgt.shape[1], tgt.shape[2]

    if snap_dir:
        Path(snap_dir).mkdir(parents=True, exist_ok=True)
        Image.fromarray((tgt.transpose(1, 2, 0) * 255).astype(np.uint8)) \
            .save(Path(snap_dir) / "target.png")

    model = NCA(channel_n, hidden_n=hidden_n).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=[int(steps * 0.8)], gamma=0.1)

    seed = make_single_seed(text, channel_n, tgt=tgt).to(device)
    pool = seed.repeat(pool_size, 1, 1, 1)   # stays on GPU

    start_step, _ = try_resume(snap_dir, model, opt, sched, device=device)

    meta = RunMeta(snap_dir, text, "nca.train_ladder_seed",
                   {"steps": steps, "batch": batch, "lr": lr,
                    "normal_p": normal_p, "damage_occasional": damage_occasional,
                    "damage_p": damage_p, "rho_target": rho_target,
                    "rho_w": rho_w, "adaptive": adaptive},
                   channel_n, hidden_n, "single", steps, device,
                   tags=["ladder-seed"] + (["adaptive"] if adaptive else []))

    stage_steps = max(1, steps // 10)
    t0 = time.time()
    for step in range(start_step, steps):
        noise_idx = max(0.0, 0.9 - 0.1 * (step // stage_steps))

        idx = torch.randperm(pool_size, device=device)[:batch]
        x = pool[idx]
        with torch.no_grad():
            loss_rank = F.mse_loss(to_rgba(x), target, reduction="none") \
                .mean(dim=(1, 2, 3)).argsort(descending=True)
        x = x[loss_rank]
        idx = idx[loss_rank]
        x[:1] = seed

        if damage_occasional and torch.rand(1).item() < damage_p:
            n_dmg = int(torch.randint(1, 4, (1,)))
            m = damage_mask_rect(n_dmg, h, w, device)
            x[-n_dmg:] = x[-n_dmg:] * m

        x_start = x[-1:].detach().clone()   # damaged-most input state for snapshots
        if adaptive:
            x, _used = adaptive_rollout(model, x, target, chunk=12, max_chunks=8)
        else:
            n_ca = int(torch.randint(ca_min, ca_max + 1, (1,)))
            x = model(x, steps=n_ca)

        normal_batch = torch.rand(1).item() < normal_p
        if noise_idx > 0 and not normal_batch:
            target_noisy = target * (1.0 - noise_idx) + torch.rand_like(target) * noise_idx
            loss = F.mse_loss(to_rgba(x), target_noisy)
        else:
            loss = F.mse_loss(to_rgba(x), target)

        # Edge-of-chaos regularizer: pull the deterministic step's local
        # Jacobian gain toward rho_target (one JVP, ~2 extra forwards).
        if rho_w > 0 and step % 2 == 0:
            xp = x[:1].detach()
            v = torch.randn_like(xp)
            v = v / (torch.linalg.vector_norm(v) + 1e-8)
            _, jv = torch.autograd.functional.jvp(
                lambda y: model.step(y, fire_rate=1.0), (xp,), (v,),
                create_graph=True)
            rho_est = torch.linalg.vector_norm(jv)
            loss = loss + rho_w * (rho_est - rho_target) ** 2

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
            print(f"[ladder_seed_{text}] step {step} loss {loss.item():.5f} "
                  f"noise {noise_idx:.1f}{' (normal)' if normal_batch else ''} "
                  f"({time.time() - t0:.1f}s)", flush=True)
            if snap_dir:
                s = f"{step:05d}"
                for tag, t in [("COMP", to_rgba(x)[-1]), ("START", to_rgba(x_start)[0])]:
                    img = t.detach().cpu().clamp(0, 1)
                    a = img[3:4]
                    vis = (1 - a + img[:3]).clamp(0, 1).permute(1, 2, 0).numpy()
                    Image.fromarray((vis * 255).astype(np.uint8)) \
                        .resize((w * 8, h * 8), Image.NEAREST) \
                        .save(Path(snap_dir) / f"{tag}_{s}.png")
                torch.save(model.state_dict(), str(Path(snap_dir) / "latest.pth"))
                meta.log(step, loss.item(), noise_idx=noise_idx)
                export_run_weights(model, snap_dir, text, glyph)
        if snap_dir and (step % ckpt_every == 0 or step == steps - 1):
            save_checkpoint(snap_dir, step, model, opt, sched)

    print(f"Final loss for {text} (ladder_seed): {loss.item():.5f}")
    return model


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--text", required=True)
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--normal-p", type=float, default=0.25)
    p.add_argument("--damage-occasional", action="store_true")
    p.add_argument("--rho-target", type=float, default=0.0)
    p.add_argument("--rho-w", type=float, default=0.0)
    p.add_argument("--adaptive", action="store_true")
    p.add_argument("--snap-dir", default=None)
    a = p.parse_args()

    train(a.text, steps=a.steps, log_every=a.log_every, normal_p=a.normal_p,
          damage_occasional=a.damage_occasional, rho_target=a.rho_target,
          rho_w=a.rho_w, adaptive=a.adaptive, snap_dir=a.snap_dir)
