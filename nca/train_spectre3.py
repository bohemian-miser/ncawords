"""Spectre v3: grow the tiling outward from a seed, tile by tile.

The v2 lesson: matching a global arrangement from noise is unanchorable.
v3 uses the paradigm that works — growth from a seed. Stage k's target
is the N_k tiles nearest the anchor tile's centroid, rendered crisp at
fixed scale and center, so the seed provides the global reference frame
and each stage adds a ring of neighbors. Stages gate on loss.

Interiors can be 'fill' (exact class colors) or 'free' (only present &
distinct from the outline — machinery latitude), with edges strongly
supervised either way (the v2 flat-gray collapse came from a weak edge
term).
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from nca.model import NCA, to_rgba
from nca.spectre import (spectre_leaves, rasterize_crisp, crisp_target,
                         SPECTRE, apply)
from nca.train_web_hidden import damage_mask_rect
from nca.checkpoint import save_checkpoint, try_resume
from nca.runmeta import RunMeta, export_run_weights
from nca.rollout import adaptive_rollout, fester

CANVAS = 96
SCALE = 5.0
RINGS = [1, 2, 5, 12, 25, 50]
EDGE_W = 3.0


def stage_targets(render):
    leaves = spectre_leaves(3)
    cents = np.array([apply(T, SPECTRE).mean(axis=0) for _, T in leaves])
    mid = cents.mean(axis=0)
    anchor = int(np.argmin(((cents - mid) ** 2).sum(axis=1)))
    order = np.argsort(((cents - cents[anchor]) ** 2).sum(axis=1))
    out = []
    for n in RINGS:
        sub = [leaves[i] for i in order[:n]]
        edge, interior, labels = rasterize_crisp(
            sub, CANVAS, SCALE, center=cents[anchor])
        out.append((crisp_target(edge, interior, labels, render), edge,
                    interior >= 0))
    return out, cents[anchor]


def train(steps=16000, channel_n=16, hidden_n=96, batch=12, pool_size=192,
          lr=2e-3, ca_min=48, ca_max=80, gate=0.01, render="fill",
          damage_p=0.3, fester_p=0.25, adaptive=True,
          log_every=200, ckpt_every=500, snap_dir=None):
    torch.manual_seed(1201)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}, render {render}")

    stages, _anchor = stage_targets(render)
    print(f"{len(stages)} ring stages: {RINGS}")

    if snap_dir:
        Path(snap_dir).mkdir(parents=True, exist_ok=True)
        t = stages[-1][0]
        vis = (1 - t[3] + t[:3]).clip(0, 1).transpose(1, 2, 0)
        Image.fromarray((vis * 255).astype(np.uint8)) \
            .resize((CANVAS * 5,) * 2, Image.NEAREST).save(Path(snap_dir) / "target.png")

    model = NCA(channel_n, hidden_n=hidden_n).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=[int(steps * 0.85)], gamma=0.1)

    seed = torch.zeros(1, channel_n, CANVAS, CANVAS, device=device)
    seed[:, 3:, CANVAS // 2, CANVAS // 2] = 1.0
    pool = seed.repeat(pool_size, 1, 1, 1)

    start_step, ckpt_extra = try_resume(snap_dir, model, opt, sched, device=device)
    stage = ckpt_extra.get("stage", 0) if ckpt_extra else 0
    stage_start = ckpt_extra.get("stage_start", 0) if ckpt_extra else 0

    meta = RunMeta(snap_dir, "SPECTRE3", "nca.train_spectre3",
                   {"steps": steps, "batch": batch, "lr": lr, "gate": gate,
                    "render": render, "rings": RINGS, "fester_p": fester_p,
                    "adaptive": adaptive},
                   channel_n, hidden_n, "single", steps, device,
                   tags=["spectre", "v3", render])

    recent = []
    stage_cap = max(1, int(steps / len(RINGS) * 1.6))
    t0 = time.time()
    for step in range(start_step, steps):
        tgt_np, edge_m, inter_m = stages[stage]
        target = torch.from_numpy(tgt_np)[None].repeat(batch, 1, 1, 1).to(device)

        idx = torch.randperm(pool_size, device=device)[:batch]
        x = pool[idx]
        with torch.no_grad():
            rank = F.mse_loss(to_rgba(x), target, reduction="none") \
                .mean(dim=(1, 2, 3)).argsort(descending=True)
        x = x[rank]; idx = idx[rank]
        x[:1] = seed
        if torch.rand(1).item() < damage_p:
            m = damage_mask_rect(2, CANVAS, CANVAS, device)
            x[-2:] = x[-2:] * m
        if fester_p > 0 and torch.rand(1).item() < fester_p:
            x = fester(model, x, min_steps=100, max_steps=350)

        if adaptive:
            x, _u = adaptive_rollout(model, x, target, chunk=10, max_chunks=8)
        else:
            n_ca = int(torch.randint(ca_min, ca_max + 1, (1,)))
            x = model(x, steps=n_ca)

        em = torch.from_numpy(edge_m).to(device)
        if render == "free":
            im_ = torch.from_numpy(inter_m).to(device)
            loss = F.mse_loss(x[:, 3:4], target[:, 3:4])
            loss = loss + EDGE_W * ((x[:, :3] - 0.08) ** 2 * em[None, None]).sum() \
                / (em.sum() * 3 * batch + 1e-8)
            bright = x[:, :3].mean(dim=1, keepdim=True)
            loss = loss + (F.relu(0.35 - bright) ** 2 * im_[None, None]).sum() \
                / (im_.sum() * batch + 1e-8)
        else:
            loss = F.mse_loss(to_rgba(x), target)
            loss = loss + (EDGE_W - 1) * \
                ((to_rgba(x) - target) ** 2 * em[None, None]).sum() \
                / (em.sum() * 4 * batch + 1e-8)

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
        avg = sum(recent) / len(recent) if recent else float("inf")
        if stage < len(RINGS) - 1 and \
                ((len(recent) == 50 and avg < gate) or step - stage_start >= stage_cap):
            print(f"=== ring {RINGS[stage]} done at {step} (avg {avg:.4f}) ===",
                  flush=True)
            stage += 1
            stage_start = step
            recent.clear()

        if step % log_every == 0 or step == steps - 1:
            print(f"[spectre3-{render}] step {step} ring {RINGS[stage]} "
                  f"loss {loss.item():.5f} avg {avg:.5f} ({time.time() - t0:.1f}s)",
                  flush=True)
            if snap_dir:
                s = f"{step:05d}"
                for tag, t in [("COMP", to_rgba(x)[0]), ("TARGET", target[0])]:
                    img = t.detach().cpu().clamp(0, 1)
                    vis = (1 - img[3:4] + img[:3]).clamp(0, 1).permute(1, 2, 0).numpy()
                    Image.fromarray((vis * 255).astype(np.uint8)) \
                        .resize((CANVAS * 5,) * 2, Image.NEAREST) \
                        .save(Path(snap_dir) / f"{tag}_{s}.png")
                torch.save(model.state_dict(), str(Path(snap_dir) / "latest.pth"))
                meta.log(step, loss.item(), ring=RINGS[stage])
                export_run_weights(model, snap_dir, "SPECTRE3",
                                   grid_w=CANVAS, grid_h=CANVAS)
        if snap_dir and (step % ckpt_every == 0 or step == steps - 1):
            save_checkpoint(snap_dir, step, model, opt, sched,
                            extra={"stage": stage, "stage_start": stage_start})

    print(f"Final: ring {RINGS[stage]}, loss {loss.item():.5f}")
    return model


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=16000)
    p.add_argument("--render", default="fill", choices=["fill", "free"])
    p.add_argument("--gate", type=float, default=0.01)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--snap-dir", default=None)
    a = p.parse_args()
    train(steps=a.steps, render=a.render, gate=a.gate,
          log_every=a.log_every, snap_dir=a.snap_dir)
