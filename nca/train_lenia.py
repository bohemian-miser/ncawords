"""Trainable-Lenia ladder: gradient descent on the physics itself.

Five variants, one config space (all kernels are parameterised radial ring
functions; all parameters bounded via sigmoid/tanh squashing):

  v1 static1  - 1 channel, 1 kernel, static params
  v2 dyn1     - 1 channel, dynamic kernel: a per-cell NN mixes a fixed ring
                BASIS (this is 'NN from the kernel outputs informs the kernel
                params', implemented efficiently as basis mixing)
  v3 multik   - 1 channel, K kernels each with own growth + gain
  v4 sharedk  - C channels, ONE kernel shape, trained CxC coupling matrix
  v5 full     - C channels, K kernels per (src,dst) pair: C*C*K unique
                kernels, growths and gains, all trained

Targets are tileable geometric textures (dots / hex / tri / square). Loss is
translation-invariant: log-power-spectrum matching on channel 0 (orientation-
sensitive, so hex 6-fold vs square 4-fold still count) + mean density. Other
channels are never supervised (free machinery). Toroidal world.

Isotropy prediction baked into the design: ring kernels should find dots and
hexagonal packings natural, squares/triangles hard without cross-channel
coupling (v4/v5) to manufacture anisotropy.
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
from nca.checkpoint import save_checkpoint, try_resume

KS = 15          # kernel support (odd); rings live inside radius KS//2
RINGS = 3        # bumps per kernel


def sig(x, lo, hi):
    return lo + (hi - lo) * torch.sigmoid(x)


def make_radius(dev):
    r = torch.arange(KS, device=dev) - KS // 2
    return (r[None, :] ** 2 + r[:, None] ** 2).float().sqrt() / (KS // 2)


def make_angle(dev):
    r = (torch.arange(KS, device=dev) - KS // 2).float()
    return torch.atan2(r[:, None], r[None, :])


class KernelBank(nn.Module):
    """N parameterised ring kernels K(r)=sum_j beta_j exp(-((r-a_j)/w_j)^2/2),
    each normalised to sum 1, plus per-kernel growth (mu, sigma) and gain h."""

    def __init__(self, n, aniso=False, harmonics=3):
        super().__init__()
        self.n = n
        self.aniso = aniso
        self.a = nn.Parameter(torch.rand(n, RINGS) * 2 - 1)
        self.w = nn.Parameter(torch.randn(n, RINGS) * 0.5)
        self.b = nn.Parameter(torch.randn(n, RINGS) * 0.5)
        if aniso:
            self.ang_c = nn.Parameter(torch.randn(n, harmonics) * 0.05)
            self.ang_s = nn.Parameter(torch.randn(n, harmonics) * 0.05)
        # init for LIVING dynamics: growth bump low (reachable from noise),
        # gains positive-biased — all-dead is absorbing and must not be step 0
        self.mu = nn.Parameter(torch.randn(n) * 0.3 - 0.8)
        self.sg = nn.Parameter(torch.randn(n) * 0.3)
        self.h = nn.Parameter(torch.randn(n) * 0.3 + 0.7)

    def kernels(self, dev):
        r = make_radius(dev)[None, None]                      # [1,1,KS,KS]
        a = sig(self.a, 0.05, 0.95)[:, :, None, None]
        w = sig(self.w, 0.03, 0.35)[:, :, None, None]
        b = sig(self.b, 0.0, 1.0)[:, :, None, None]
        k = (b * torch.exp(-((r - a) / w) ** 2 / 2)).sum(1)   # [n,KS,KS]
        if getattr(self, "aniso", False):
            # trainable angular Fourier factor (von-Mises style, >0, smooth);
            # init ~0 => isotropic, gradients grow directional lobes on demand
            phi = make_angle(dev)[None]
            m = torch.arange(1, self.ang_c.shape[1] + 1, device=dev) \
                .view(1, -1, 1, 1).float()
            cc = 1.5 * torch.tanh(self.ang_c)[:, :, None, None]
            ss = 1.5 * torch.tanh(self.ang_s)[:, :, None, None]
            ang = (cc * torch.cos(m * phi[:, None]) +
                   ss * torch.sin(m * phi[:, None])).sum(1)
            k = k * torch.exp(ang)
        k = k * (r[0, 0][None] <= 1.0)
        return k / (k.sum(dim=(1, 2), keepdim=True) + 1e-8)

    def growth(self, u, anneal=1.0):
        mu = sig(self.mu, 0.0, 1.0).view(1, -1, 1, 1)
        s = sig(self.sg, 0.02, 0.35).view(1, -1, 1, 1) / anneal
        return 2.0 * torch.exp(-((u - mu) / s) ** 2 / 2) - 1.0

    def gains(self):
        return torch.tanh(self.h)


class WaveBank(nn.Module):
    """Gabor/plane-wave kernels: cos(2*pi*f*(x cos th + y sin th) + phase)
    under a radial gaussian envelope — sinusoids with trainable DIRECTION.
    Kernels are zero-mean, L1-normalised; u can go negative, so growth mu
    ranges over [-0.6, 1]."""

    def __init__(self, n):
        super().__init__()
        self.n = n
        self.th = nn.Parameter(torch.rand(n) * 6.28)
        self.fr = nn.Parameter(torch.randn(n) * 0.5)
        self.ph = nn.Parameter(torch.randn(n) * 0.5)
        self.er = nn.Parameter(torch.randn(n) * 0.5)
        self.ew = nn.Parameter(torch.randn(n) * 0.5)
        self.mu = nn.Parameter(torch.randn(n) * 0.3)
        self.sg = nn.Parameter(torch.randn(n) * 0.3)
        self.h = nn.Parameter(torch.randn(n) * 0.3 + 0.7)

    def kernels(self, dev):
        r = make_radius(dev)[None]
        cy = (torch.arange(KS, device=dev) - KS // 2).float() / (KS // 2)
        xx = cy[None, :].expand(KS, KS)[None]
        yy = cy[:, None].expand(KS, KS)[None]
        th = self.th.view(-1, 1, 1)
        f = sig(self.fr, 0.5, 3.0).view(-1, 1, 1)
        u = xx * torch.cos(th) + yy * torch.sin(th)
        env = torch.exp(-((r - sig(self.er, 0.0, 0.7).view(-1, 1, 1)) /
                          sig(self.ew, 0.15, 0.6).view(-1, 1, 1)) ** 2 / 2)
        k = env * torch.cos(6.2832 * f * u + self.ph.view(-1, 1, 1))
        k = k * (r[0][None] <= 1.0)
        k = k - k.mean(dim=(1, 2), keepdim=True)
        return k / (k.abs().sum(dim=(1, 2), keepdim=True) + 1e-8)

    def growth(self, u, anneal=1.0):
        mu = sig(self.mu, -0.6, 1.0).view(1, -1, 1, 1)
        s = sig(self.sg, 0.02, 0.35).view(1, -1, 1, 1) / anneal
        return 2.0 * torch.exp(-((u - mu) / s) ** 2 / 2) - 1.0

    def gains(self):
        return torch.tanh(self.h)


class Lenia(nn.Module):
    def __init__(self, variant, C=1, K=3, dt=0.25, basis_n=8, hidden=24):
        super().__init__()
        self.variant, self.C, self.K, self.dt = variant, C, K, dt
        if variant == "static1":
            self.bank = KernelBank(1)
        elif variant == "aniso":
            self.bank = KernelBank(K, aniso=True)
        elif variant == "wave":
            self.bank = WaveBank(K)
        elif variant == "dynwave":
            # fixed oriented Gabor basis (8 orientations), per-cell mixing NN:
            # cells steer their own perception direction locally
            rr = make_radius(torch.device("cpu"))
            cy = (torch.arange(KS) - KS // 2).float() / (KS // 2)
            xx, yy = cy[None, :].expand(KS, KS), cy[:, None].expand(KS, KS)
            ks = []
            for i in range(8):
                th = 3.1416 * i / 8
                u = xx * np.cos(th) + yy * np.sin(th)
                k = torch.exp(-(rr / 0.6) ** 2 / 2) * torch.cos(6.2832 * 1.5 * u)
                k = k * (rr <= 1.0); k = k - k.mean()
                ks.append(k / (k.abs().sum() + 1e-8))
            self.register_buffer("basis", torch.stack(ks)[:, None])
            self.mix = nn.Sequential(
                nn.Conv2d(1 + 8, hidden, 1), nn.ReLU(), nn.Conv2d(hidden, 8, 1))
            self.bank = WaveBank(1)   # growth params only
        elif variant == "dyn1":
            # fixed ring basis; NN reads (x, basis outputs) -> per-cell mixing
            self.basis_n = basis_n
            rr = make_radius(torch.device("cpu"))
            ks = []
            for i in range(basis_n):
                a = 0.08 + 0.84 * i / max(1, basis_n - 1)
                k = torch.exp(-((rr - a) / 0.10) ** 2 / 2) * (rr <= 1.0)
                ks.append(k / (k.sum() + 1e-8))
            self.register_buffer("basis", torch.stack(ks)[:, None])  # [B,1,KS,KS]
            self.mix = nn.Sequential(
                nn.Conv2d(1 + basis_n, hidden, 1), nn.ReLU(),
                nn.Conv2d(hidden, basis_n, 1))
            self.bank = KernelBank(1)      # growth params (kernel comes from mix)
        elif variant == "multik":
            self.bank = KernelBank(K)
        elif variant == "sharedk":
            self.bank = KernelBank(1)
            self.H = nn.Parameter(torch.randn(C, C) * 0.3)
        elif variant == "full":
            self.bank = KernelBank(C * C * K)
        else:
            raise ValueError(variant)

    def conv(self, x, k):
        # x [B,1,H,W], k [n,KS,KS] -> [B,n,H,W], toroidal
        xp = F.pad(x, (KS // 2,) * 4, mode="circular")
        return F.conv2d(xp, k[:, None])

    def step(self, x, anneal=1.0):
        B, C, H, W = x.shape
        v = self.variant
        if v in ("static1", "multik", "aniso", "wave"):
            k = self.bank.kernels(x.device)
            u = self.conv(x, k)
            g = self.bank.growth(u, anneal) * self.bank.gains().view(1, -1, 1, 1)
            dx = g.sum(1, keepdim=True)
        elif v == "dynwave":
            u_b = self.conv(x, self.basis[:, 0])
            c = torch.tanh(self.mix(torch.cat([x, u_b], 1)))
            u = (c * u_b).sum(1, keepdim=True)
            g = self.bank.growth(u, anneal) * self.bank.gains().view(1, -1, 1, 1)
            dx = g
        elif v == "dyn1":
            u_b = self.conv(x, self.basis[:, 0])            # [B,Bn,H,W]
            c = torch.tanh(self.mix(torch.cat([x, u_b], 1)))
            u = (c * u_b).sum(1, keepdim=True)              # per-cell kernel
            g = self.bank.growth(u, anneal) * self.bank.gains().view(1, -1, 1, 1)
            dx = g
        elif v == "sharedk":
            k = self.bank.kernels(x.device)                 # [1,KS,KS]
            u = self.conv(x.reshape(B * C, 1, H, W), k).reshape(B, C, H, W)
            g = self.bank.growth(u.reshape(B * C, 1, H, W), anneal) \
                .reshape(B, C, H, W)
            dx = torch.einsum("st,bshw->bthw", torch.tanh(self.H), g)
        else:  # full: C*C*K unique kernels
            k = self.bank.kernels(x.device)                 # [C*C*K,KS,KS]
            k = k.view(self.C, self.C, self.K, KS, KS)
            mu_all = self.bank.growth                        # applied per slice below
            dx = torch.zeros_like(x)
            gains = self.bank.gains().view(self.C, self.C, self.K)
            # per source channel: conv against its C*K kernels
            aidx = 0
            for s in range(self.C):
                ksrc = k[s].reshape(self.C * self.K, KS, KS)
                u = self.conv(x[:, s:s + 1], ksrc)          # [B,C*K,H,W]
                # growth params for this source's slice of the bank
                sl = slice(s * self.C * self.K, (s + 1) * self.C * self.K)
                mu = sig(self.bank.mu[sl], 0.0, 1.0).view(1, -1, 1, 1)
                sgm = sig(self.bank.sg[sl], 0.02, 0.35).view(1, -1, 1, 1) / anneal
                g = 2.0 * torch.exp(-((u - mu) / sgm) ** 2 / 2) - 1.0
                g = g * gains[s].reshape(1, self.C * self.K, 1, 1)
                dx = dx + g.view(B, self.C, self.K, H, W).sum(2)
        xn = x + self.dt * dx
        c = xn.clamp(0, 1)
        return c + 0.05 * (xn - c)   # leaky clamp: gradient survives saturation


# ---------------------------------------------------------------- targets

def _lines(H, W, dirs, spacing, width):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    out = np.zeros((H, W), np.float32)
    for th in dirs:
        p = xx * np.cos(th) + yy * np.sin(th)
        d = np.abs(((p / spacing) % 1.0) - 0.5) * spacing
        out = np.maximum(out, (d < width).astype(np.float32))
    return out


def make_target(kind, H=64, W=64):
    if kind == "dots":
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        s = 12.0
        best = np.full((H, W), 1e9, np.float32)
        for i in range(-1, H // int(s * 0.87) + 2):
            for j in range(-2, W // int(s) + 2):
                cy, cx = i * s * 0.866, j * s + (i % 2) * s / 2
                best = np.minimum(best, (yy - cy) ** 2 + (xx - cx) ** 2)
        return (best < 3.2 ** 2).astype(np.float32)
    if kind == "hex":
        return _lines(H, W, [0, np.pi / 3, 2 * np.pi / 3], 11.0, 1.1)
    if kind == "tri":
        return _lines(H, W, [np.pi / 6, np.pi / 2, 5 * np.pi / 6], 11.0, 1.1)
    if kind == "square":
        return _lines(H, W, [0, np.pi / 2], 12.0, 1.2)
    raise ValueError(kind)


def spec(x):
    """Blurred log power spectrum of [B,H,W] (channel 0), translation-invariant."""
    p = torch.fft.rfft2(x - x.mean(dim=(-2, -1), keepdim=True))
    p = torch.log1p(p.abs())
    return F.avg_pool2d(p[:, None], 3, 1, 1)[:, 0]


# ---------------------------------------------------------------- training

def train(variant="static1", target="dots", C=1, K=3, steps=6000, batch=8,
          size=64, lr=5e-3, t_min=16, t_max=48, rng_seed=0,
          log_every=150, ckpt_every=500, snap_dir=None):
    if variant in ("static1", "dyn1", "multik", "aniso", "wave", "dynwave"):
        C = 1
    torch.manual_seed(400 + rng_seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Lenia(variant, C=C, K=K).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"Device {device}, {variant} C={C} K={K}, {n_par} physics params")

    tgt = torch.from_numpy(make_target(target, size, size)).to(device)
    tgt_spec = spec(tgt[None])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, [int(steps * 0.85)], 0.1)

    if snap_dir:
        Path(snap_dir).mkdir(parents=True, exist_ok=True)
        Image.fromarray(((1 - tgt.cpu().numpy()) * 255).astype(np.uint8)) \
            .resize((size * 5,) * 2, Image.NEAREST).save(Path(snap_dir) / "target.png")
    meta = RunMeta(snap_dir, f"LENIA-{variant}-{target}", "nca.train_lenia",
                   {"variant": variant, "target": target, "C": C, "K": K,
                    "steps": steps, "batch": batch, "lr": lr,
                    "rng_seed": rng_seed, "params": n_par},
                   C, 0, "noise", steps, device, tags=["lenia", variant, target])

    start_step, _ = try_resume(snap_dir, model, opt, sched, device=device)

    t0 = time.time()
    for step in range(start_step, steps):
        # growth-bump annealing: wide (easy gradients) -> trained width
        anneal = 0.5 + 0.5 * min(1.0, step / (steps * 0.3))
        # horizon curriculum inside [t_min, t_max]
        hi = t_min + int((t_max - t_min) * min(1.0, step / (steps * 0.5)))
        T = int(torch.randint(t_min, hi + 1, (1,)))
        x = torch.rand(batch, C, size, size, device=device) * 0.6
        for _ in range(T):
            x = model.step(x, anneal)
        loss = F.mse_loss(spec(x[:, 0]), tgt_spec.expand(batch, -1, -1)) \
            + 2.0 * (x[:, 0].mean() - tgt.mean()) ** 2

        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()

        if step % log_every == 0 or step == steps - 1:
            print(f"[lenia-{variant}-{target}] step {step} loss {loss.item():.4f} "
                  f"T {T} ({time.time() - t0:.1f}s)", flush=True)
            if snap_dir:
                s = f"{step:05d}"
                img = x[0, 0].detach().cpu().numpy()
                Image.fromarray(((1 - img) * 255).astype(np.uint8)) \
                    .resize((size * 5,) * 2, Image.NEAREST) \
                    .save(Path(snap_dir) / f"COMP_{s}.png")
                # the learned physics, rendered: kernels tiled side by side
                with torch.no_grad():
                    if variant in ("dyn1", "dynwave"):
                        kimg = model.basis[:, 0].cpu().numpy()
                    else:
                        kimg = model.bank.kernels(device).detach().cpu().numpy()
                kimg = kimg[:24]
                kmin = kimg.min(axis=(1, 2), keepdims=True)
                kn = (kimg - kmin) / (kimg.max(axis=(1, 2), keepdims=True) - kmin + 1e-9)
                row = np.concatenate(list(kn), axis=1)
                Image.fromarray((row * 255).astype(np.uint8)) \
                    .resize((row.shape[1] * 6, row.shape[0] * 6), Image.NEAREST) \
                    .save(Path(snap_dir) / f"KERNEL_{s}.png")
                if variant == "sharedk":
                    Hm = torch.tanh(model.H).detach().cpu().numpy()
                    Hn = (Hm - Hm.min()) / (Hm.ptp() + 1e-9)
                    Image.fromarray((Hn * 255).astype(np.uint8)) \
                        .resize((C * 24,) * 2, Image.NEAREST) \
                        .save(Path(snap_dir) / f"COUPLING_{s}.png")
                torch.save(model.state_dict(), str(Path(snap_dir) / "latest.pth"))
                meta.log(step, loss.item())
        if snap_dir and (step % ckpt_every == 0 or step == steps - 1):
            save_checkpoint(snap_dir, step, model, opt, sched)

    print(f"Final loss {loss.item():.4f}")
    return model


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="static1",
                   choices=["static1", "dyn1", "multik", "sharedk", "full",
                            "aniso", "wave", "dynwave"])
    p.add_argument("--target", default="dots",
                   choices=["dots", "hex", "tri", "square"])
    p.add_argument("--channels", type=int, default=6)
    p.add_argument("--kernels", type=int, default=3)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--rng-seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=150)
    p.add_argument("--snap-dir", default=None)
    a = p.parse_args()
    train(variant=a.variant, target=a.target, C=a.channels, K=a.kernels,
          steps=a.steps, rng_seed=a.rng_seed, log_every=a.log_every,
          snap_dir=a.snap_dir)
