"""Experiment: can a neural CA learn the rules of Conway's Game of Life?

This is the first step toward the 'static rules, trained machinery' idea:
before we can freeze a physics and optimise creatures inside it, we need a
differentiable physics that faithfully reproduces a known discrete rule.

Three things the growing-NCA architecture gets WRONG for Life, fixed here:
  1. Perception. Sobel filters give directional gradients; Life's rule is a
     function of the LIVE-NEIGHBOUR COUNT. We add a Moore-sum filter
     (3x3 box minus centre). --perception lets us A/B sobel vs sum vs both,
     to show the inductive bias is what matters.
  2. Alive-masking. Growing-NCA zeroes cells with no live neighbours — but
     Life BIRTHS cells from empty neighbourhoods. Removed.
  3. Async fire rate. Life is synchronous. Every cell updates every step.

The net learns a direct next-state map (Life is Markov — no memory needed;
hidden channels are optional scratch). Trained on random boards + gliders,
supervised over a multi-step rollout so it learns the rule, not one-step
statistics. Toroidal world (circular padding) to match standard Life.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from nca.runmeta import RunMeta

GLIDER = np.array([[0, 1, 0], [0, 0, 1], [1, 1, 1]], np.float32)


def gol_step(b):
    """One synchronous Life step on [N,H,W] {0,1}, toroidal."""
    nb = sum(np.roll(np.roll(b, dy, axis=1), dx, axis=2)
             for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dx, dy) != (0, 0))
    return ((nb == 3) | ((b == 1) & (nb == 2))).astype(np.float32)


def make_batch(n, H, W, rng, glider_frac=0.25):
    b = (rng.random((n, H, W)) < rng.uniform(0.15, 0.4, (n, 1, 1))).astype(np.float32)
    for i in range(n):
        if rng.random() < glider_frac:
            y, x = rng.integers(0, H - 3), rng.integers(0, W - 3)
            b[i, y:y + 3, x:x + 3] = GLIDER
    return b


class GoLNet(nn.Module):
    def __init__(self, hidden_n=64, perception="sum", vis=1, hid=0):
        super().__init__()
        self.vis, self.hid, self.C = vis, hid, vis + hid
        ident = torch.tensor([[0., 0, 0], [0, 1, 0], [0, 0, 0]])
        moore = torch.tensor([[1., 1, 1], [1, 0, 1], [1, 1, 1]])   # neighbour count
        sx = torch.tensor([[-1., 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8
        sy = sx.T
        filt = [ident]
        if perception in ("sum", "both"):
            filt.append(moore)
        if perception in ("sobel", "both"):
            filt += [sx, sy]
        self.P = len(filt)
        k = torch.stack(filt).repeat(self.C, 1, 1)[:, None]
        self.register_buffer("kern", k)
        self.fc0 = nn.Conv2d(self.C * self.P, hidden_n, 1)
        self.fc1 = nn.Conv2d(hidden_n, self.C, 1)

    def perceive(self, x):
        xp = F.pad(x, (1, 1, 1, 1), mode="circular")
        return F.conv2d(xp, self.kern, groups=self.C)

    def step(self, x):
        y = self.fc1(F.relu(self.fc0(self.perceive(x))))
        vis = torch.sigmoid(y[:, :self.vis])          # direct next-state map
        hid = y[:, self.vis:] if self.hid else y[:, :0]
        return torch.cat([vis, hid], 1)

    def forward(self, x, steps=1):
        for _ in range(steps):
            x = self.step(x)
        return x


def train(steps=8000, H=32, W=32, batch=32, hidden_n=64, perception="sum",
          hid=0, rollout=4, lr=2e-3, rng_seed=0, log_every=200, snap_dir=None):
    torch.manual_seed(1234 + rng_seed)
    rng = np.random.default_rng(rng_seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device {device}, perception {perception}, hid {hid}, rollout {rollout}")

    model = GoLNet(hidden_n, perception, vis=1, hid=hid).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, [int(steps * 0.85)], 0.1)

    # fixed eval: a glider, tracked over its 4-step period (returns shifted)
    ev = np.zeros((1, H, W), np.float32); ev[0, 2:5, 2:5] = GLIDER
    ev_states = [ev.copy()]
    for _ in range(rollout):
        ev_states.append(gol_step(ev_states[-1]))
    ev0 = torch.from_numpy(ev).to(device)

    if snap_dir:
        Path(snap_dir).mkdir(parents=True, exist_ok=True)
    meta = RunMeta(snap_dir, "GOL", "nca.train_gol",
                   {"steps": steps, "perception": perception, "hid": hid,
                    "rollout": rollout, "batch": batch, "H": H, "W": W,
                    "rng_seed": rng_seed},
                   1 + hid, hidden_n, "board", steps, device,
                   tags=["gol", "rules", perception])

    t0 = time.time()
    for step in range(steps):
        b = make_batch(batch, H, W, rng)
        seq = [b]
        for _ in range(rollout):
            seq.append(gol_step(seq[-1]))
        x = torch.zeros(batch, model.C, H, W, device=device)
        x[:, 0] = torch.from_numpy(b).to(device)
        loss = 0.0
        for t in range(1, rollout + 1):
            x = model.step(x)
            tgt = torch.from_numpy(seq[t]).to(device)
            loss = loss + F.binary_cross_entropy(x[:, 0].clamp(1e-6, 1 - 1e-6), tgt)
        loss = loss / rollout

        opt.zero_grad(); loss.backward(); opt.step(); sched.step()

        if step % log_every == 0 or step == steps - 1:
            with torch.no_grad():
                # exact-match accuracy on fresh boards, thresholded at 0.5
                vb = make_batch(64, H, W, rng)
                vx = torch.zeros(64, model.C, H, W, device=device)
                vx[:, 0] = torch.from_numpy(vb).to(device)
                acc1 = []
                cur = vx
                for t in range(1, rollout + 1):
                    cur = model.step(cur)
                    pred = (cur[:, 0] > 0.5).float().cpu().numpy()
                    tgt = gol_step(vb if t == 1 else tgt_prev)
                    tgt_prev = tgt
                    acc1.append(float((pred == tgt).mean()))
                cell_acc = sum(acc1) / len(acc1)
            print(f"[gol-{perception}] step {step} bce {loss.item():.5f} "
                  f"cell_acc {cell_acc:.4f} ({time.time() - t0:.1f}s)", flush=True)
            if snap_dir:
                # glider rollout panel: pred (top) vs truth (bottom)
                with torch.no_grad():
                    gx = ev0.clone()
                    gx = torch.cat([gx, torch.zeros(1, model.hid, H, W, device=device)], 1) \
                        if model.hid else gx.unsqueeze(0) if gx.dim() == 3 else gx
                    frames = []
                    cc = torch.zeros(1, model.C, H, W, device=device); cc[:, 0] = ev0
                    for t in range(rollout + 1):
                        frames.append((cc[0, 0] > 0.5).float().cpu().numpy())
                        cc = model.step(cc)
                strip_p = np.concatenate(frames, axis=1)
                strip_t = np.concatenate([s[0] for s in ev_states], axis=1)
                panel = np.concatenate([strip_p, strip_t], axis=0)
                Image.fromarray(((1 - panel) * 255).astype(np.uint8)) \
                    .resize((panel.shape[1] * 6, panel.shape[0] * 6), Image.NEAREST) \
                    .save(Path(snap_dir) / f"GOL_{step:05d}.png")
                torch.save(model.state_dict(), str(Path(snap_dir) / "latest.pth"))
                meta.log(step, loss.item(), cell_acc=round(cell_acc, 4))

    print(f"Final bce {loss.item():.5f}")
    return model


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--perception", default="sum", choices=["sum", "sobel", "both"])
    p.add_argument("--hid", type=int, default=0)
    p.add_argument("--rollout", type=int, default=4)
    p.add_argument("--rng-seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--snap-dir", default=None)
    a = p.parse_args()
    train(steps=a.steps, perception=a.perception, hid=a.hid, rollout=a.rollout,
          rng_seed=a.rng_seed, log_every=a.log_every, snap_dir=a.snap_dir)
