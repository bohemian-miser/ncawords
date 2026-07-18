"""Grow Hilbert curves — global connectivity from local rules.

Stage-1 baseline: pool/seed growth toward the exact-lattice curve target
(arclength color gradient). Optional differentiable validity loss
(--degree-w): a valid path cell has exactly two path-neighbors, so we
penalize |neighbor_count - 2| on confident path cells — rewarding ANY
locally-path-like structure, not just the template. Exact global
invariants (components/cycles) are logged for evaluation, not trained on.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from nca.model import NCA, to_rgba
from nca.hilbert import rasterize_curve, field_invariants
from nca.train_web_hidden import damage_mask_rect
from nca.checkpoint import save_checkpoint, try_resume
from nca.runmeta import RunMeta, export_run_weights
from nca.rollout import adaptive_rollout, fester

NEIGH = torch.tensor([[0., 1., 0.], [1., 0., 1.], [0., 1., 0.]])[None, None]


def degree_loss(alpha):
    """Penalize path cells whose 4-neighborhood count differs from 2."""
    n = F.conv2d(alpha, NEIGH.to(alpha.device), padding=1)
    onish = (alpha > 0.5).float()
    return ((n - 2.0) ** 2 * onish).sum() / (onish.sum() + 1e-6)


def train(order=3, steps=12000, canvas=None, channel_n=16, hidden_n=96,
          batch=12, pool_size=256, lr=2e-3, ca_min=64, ca_max=96,
          damage_p=0.3, degree_w=0.0, adaptive=False, fester_p=0.0,
          log_every=200, ckpt_every=500, snap_dir=None):
    torch.manual_seed(4000 + order)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}, order {order}")

    canvas = canvas or {1: 64, 2: 64, 3: 64, 4: 72, 5: 132}[order]
    tgt_np = rasterize_curve(order, canvas)
    target = torch.from_numpy(tgt_np)[None].repeat(batch, 1, 1, 1).to(device)

    if snap_dir:
        Path(snap_dir).mkdir(parents=True, exist_ok=True)
        vis = (1 - tgt_np[3] + tgt_np[:3]).clip(0, 1).transpose(1, 2, 0)
        Image.fromarray((vis * 255).astype(np.uint8)) \
            .resize((canvas * 5,) * 2, Image.NEAREST).save(Path(snap_dir) / "target.png")

    model = NCA(channel_n, hidden_n=hidden_n).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=[int(steps * 0.85)], gamma=0.1)

    seed = torch.zeros(1, channel_n, canvas, canvas, device=device)
    ys, xs = np.where(tgt_np[3] > 0.5)
    seed[:, 3:, ys[0], xs[0]] = 1.0   # start of the curve
    pool = seed.repeat(pool_size, 1, 1, 1)

    start_step, _ = try_resume(snap_dir, model, opt, sched, device=device)

    meta = RunMeta(snap_dir, f"H{order}", "nca.train_hilbert",
                   {"order": order, "steps": steps, "batch": batch, "lr": lr,
                    "degree_w": degree_w, "adaptive": adaptive,
                    "damage_p": damage_p, "fester_p": fester_p},
                   channel_n, hidden_n, "single", steps, device,
                   tags=["hilbert", f"order{order}"]
                        + (["adaptive"] if adaptive else [])
                        + (["degree"] if degree_w > 0 else []))

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
            m = damage_mask_rect(2, canvas, canvas, device)
            x[-2:] = x[-2:] * m

        if fester_p > 0 and torch.rand(1).item() < fester_p:
            x = fester(model, x,
                       damage_fn=lambda z: z * damage_mask_rect(2, canvas, canvas, device))
        x_start = x[-1:].detach().clone()
        if adaptive:
            x, _u = adaptive_rollout(model, x, target, chunk=12, max_chunks=8)
        else:
            n_ca = int(torch.randint(ca_min, ca_max + 1, (1,)))
            x = model(x, steps=n_ca)

        loss = F.mse_loss(to_rgba(x), target)
        if degree_w > 0:
            loss = loss + degree_w * degree_loss(x[:, 3:4].clamp(0, 1))

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
            a = x[0, 3].detach().cpu().clamp(0, 1).numpy()
            comp, cyc, viol, cover = field_invariants(a, order, canvas)
            print(f"[hilbert{order}] step {step} loss {loss.item():.5f} "
                  f"C={comp} cyc={cyc} viol={viol:.3f} cover={cover:.2f} "
                  f"({time.time() - t0:.1f}s)", flush=True)
            if snap_dir:
                s = f"{step:05d}"
                for tag, t in [("COMP", to_rgba(x)[0]), ("START", to_rgba(x_start)[0])]:
                    img = t.detach().cpu().clamp(0, 1)
                    vis = (1 - img[3:4] + img[:3]).clamp(0, 1).permute(1, 2, 0).numpy()
                    Image.fromarray((vis * 255).astype(np.uint8)) \
                        .resize((canvas * 5,) * 2, Image.NEAREST) \
                        .save(Path(snap_dir) / f"{tag}_{s}.png")
                torch.save(model.state_dict(), str(Path(snap_dir) / "latest.pth"))
                meta.log(step, loss.item(), components=comp, cycles=cyc,
                         degree_viol=round(viol, 4), coverage=round(cover, 3))
                export_run_weights(model, snap_dir, f"H{order}",
                                   grid_w=canvas, grid_h=canvas)
        if snap_dir and (step % ckpt_every == 0 or step == steps - 1):
            save_checkpoint(snap_dir, step, model, opt, sched)

    print(f"Final loss: {loss.item():.5f}")
    return model


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--order", type=int, default=3)
    p.add_argument("--steps", type=int, default=12000)
    p.add_argument("--degree-w", type=float, default=0.0)
    p.add_argument("--adaptive", action="store_true")
    p.add_argument("--fester-p", type=float, default=0.0)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--snap-dir", default=None)
    a = p.parse_args()
    train(order=a.order, steps=a.steps, degree_w=a.degree_w,
          adaptive=a.adaptive, fester_p=a.fester_p,
          log_every=a.log_every, snap_dir=a.snap_dir)
