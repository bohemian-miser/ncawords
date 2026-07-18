"""Grow valid Spectre tilings, zooming out as the model improves.

Curriculum: stages of (target_noise, zoom) descend together — early
stages see a few huge tiles through heavy noise; each time the moving-
average loss beats the gate, the view zooms out (more tiles, more
matching constraints) and the noise drops.

Adaptive rollout: instead of a fixed random number of CA sub-steps, the
CA runs in chunks and stops when a chunk no longer improves the batch
loss by a relative threshold (or at a hard cap). Gradients flow through
every executed chunk.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from nca.model import NCA, to_rgba
from nca.spectre import spectre_leaves, rasterize, SPECTRE, apply
from nca.checkpoint import save_checkpoint, try_resume
from nca.runmeta import RunMeta, export_run_weights

CANVAS = 72
STAGES = [(0.80, 14.0), (0.65, 10.0), (0.50, 7.0), (0.35, 5.0),
          (0.20, 3.5), (0.0, 3.5)]


def train(steps=16000, channel_n=16, hidden_n=96, batch=12, pool_size=128,
          lr=2e-3, chunk=8, max_chunks=10, improve_eps=0.02, gate=0.02,
          adaptive=True, ca_min=48, ca_max=72,
          log_every=100, ckpt_every=500, snap_dir=None, rng_seed=0):
    torch.manual_seed(101)
    rng = np.random.default_rng(rng_seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")

    print("Generating Spectre tiling (substitution, 3 iterations)...")
    leaves = spectre_leaves(3)
    pts = np.concatenate([apply(T, SPECTRE) for _, T in leaves])
    center = pts.mean(axis=0)
    targets_np = [rasterize(leaves, CANVAS, scale, center=center, upscale=4)
                  for _, scale in STAGES]
    print(f"{len(leaves)} tiles; {len(STAGES)} zoom stages rendered")

    if snap_dir:
        Path(snap_dir).mkdir(parents=True, exist_ok=True)
        vis = (1 - targets_np[-1][3] + targets_np[-1][:3]).clip(0, 1).transpose(1, 2, 0)
        Image.fromarray((vis * 255).astype(np.uint8)) \
            .resize((CANVAS * 6,) * 2, Image.NEAREST).save(Path(snap_dir) / "target.png")

    model = NCA(channel_n, hidden_n=hidden_n).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=[int(steps * 0.85)], gamma=0.1)

    pool = torch.rand(pool_size, channel_n, CANVAS, CANVAS, device=device)

    start_step, ckpt_extra = try_resume(snap_dir, model, opt, sched, device=device)
    stage = ckpt_extra.get("stage", 0) if ckpt_extra else 0
    stage_start = ckpt_extra.get("stage_start", 0) if ckpt_extra else 0

    meta = RunMeta(snap_dir, "SPECTRE", "nca.train_spectre",
                   {"steps": steps, "batch": batch, "lr": lr, "gate": gate,
                    "adaptive": adaptive, "chunk": chunk,
                    "max_chunks": max_chunks, "improve_eps": improve_eps},
                   channel_n, hidden_n, "noise", steps, device)

    recent = []
    stage_cap = max(1, int(steps / len(STAGES) * 1.5))
    t0 = time.time()
    for step in range(start_step, steps):
        noise_lvl, _scale = STAGES[stage]
        target = torch.from_numpy(targets_np[stage])[None] \
            .repeat(batch, 1, 1, 1).to(device)

        idx = torch.randperm(pool_size, device=device)[:batch]
        x = pool[idx]
        with torch.no_grad():
            rank = F.mse_loss(to_rgba(x), target, reduction="none") \
                .mean(dim=(1, 2, 3)).argsort(descending=True)
        x = x[rank]; idx = idx[rank]
        # worst sample restarts from fresh noise (keeps nucleation trained)
        x[:1] = torch.rand_like(x[:1])

        used_steps = 0
        if adaptive:
            prev = None
            for _ in range(max_chunks):
                x = model(x, steps=chunk)
                used_steps += chunk
                cur = float(F.mse_loss(to_rgba(x), target).detach())
                if prev is not None and (prev - cur) < improve_eps * prev:
                    break
                prev = cur
        else:
            n_ca = int(torch.randint(ca_min, ca_max + 1, (1,)))
            x = model(x, steps=n_ca)
            used_steps = n_ca

        if noise_lvl > 0:
            tnoisy = target * (1 - noise_lvl) + torch.rand_like(target) * noise_lvl
            loss = F.mse_loss(to_rgba(x), tnoisy)
        else:
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

        recent.append(loss.item())
        if len(recent) > 50:
            recent.pop(0)
        avg = sum(recent) / len(recent)
        if stage < len(STAGES) - 1 and \
                ((len(recent) == 50 and avg < gate) or step - stage_start >= stage_cap):
            print(f"=== stage {stage} (noise {noise_lvl}, zoom) done at {step} "
                  f"(avg {avg:.4f}) ===", flush=True)
            stage += 1
            stage_start = step
            recent.clear()

        if step % log_every == 0 or step == steps - 1:
            print(f"[spectre] step {step} stage {stage} loss {loss.item():.5f} "
                  f"avg {avg:.5f} ca_steps {used_steps} ({time.time() - t0:.1f}s)",
                  flush=True)
            if snap_dir:
                s = f"{step:05d}"
                for tag, t in [("COMP", to_rgba(x)[0]), ("TARGET", target[0])]:
                    img = t.detach().cpu().clamp(0, 1)
                    vis = (1 - img[3:4] + img[:3]).clamp(0, 1).permute(1, 2, 0).numpy()
                    Image.fromarray((vis * 255).astype(np.uint8)) \
                        .resize((CANVAS * 6,) * 2, Image.NEAREST) \
                        .save(Path(snap_dir) / f"{tag}_{s}.png")
                torch.save(model.state_dict(), str(Path(snap_dir) / "latest.pth"))
                meta.log(step, loss.item(), stage=stage, ca_steps=used_steps)
                export_run_weights(model, snap_dir, "SPECTRE",
                                   grid_w=CANVAS, grid_h=CANVAS, seed_type="noise")
        if snap_dir and (step % ckpt_every == 0 or step == steps - 1):
            save_checkpoint(snap_dir, step, model, opt, sched,
                            extra={"stage": stage, "stage_start": stage_start})

    print(f"Final: stage {stage}, loss {loss.item():.5f}")
    return model


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=16000)
    p.add_argument("--gate", type=float, default=0.02)
    p.add_argument("--no-adaptive", action="store_true")
    p.add_argument("--rng-seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--snap-dir", default=None)
    a = p.parse_args()
    train(steps=a.steps, gate=a.gate, adaptive=not a.no_adaptive,
          rng_seed=a.rng_seed, log_every=a.log_every, snap_dir=a.snap_dir)
