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
        elif variant == "sphere":
            # thin-slab 3D Lenia: channels are LAYERS stacked at learned
            # heights; one spherical-shell kernel K(rho), rho = 3D distance,
            # derives ALL layer-pair couplings from geometry. Per-layer
            # broadcast gains restore excite/inhibit identity. Potentials sum
            # (a true 3D convolution) before a single growth function.
            self.bank = KernelBank(1)
            self.zpos = nn.Parameter(torch.arange(C).float() * 0.25)
            self.gin = nn.Parameter(torch.randn(C) * 0.2 + 0.7)
        elif variant == "dynwave":
            # fixed oriented Gabor basis (8 orientations), per-cell mixing NN:
            # cells steer their own perception direction locally. With C>1 the
            # mixer also reads the extra channels (e.g. a clamped scaffold),
            # so steering can respond to a conditioning field.
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
                nn.Conv2d(C + 8, hidden, 1), nn.ReLU(), nn.Conv2d(hidden, 8, 1))
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
            u_b = self.conv(x[:, :1], self.basis[:, 0])
            c = torch.tanh(self.mix(torch.cat([x, u_b], 1)))
            u = (c * u_b).sum(1, keepdim=True)
            g = self.bank.growth(u, anneal) * self.bank.gains().view(1, -1, 1, 1)
            dx = torch.cat([g, torch.zeros_like(x[:, 1:])], 1) if C > 1 else g
        elif v == "dyn1":
            u_b = self.conv(x, self.basis[:, 0])            # [B,Bn,H,W]
            c = torch.tanh(self.mix(torch.cat([x, u_b], 1)))
            u = (c * u_b).sum(1, keepdim=True)              # per-cell kernel
            g = self.bank.growth(u, anneal) * self.bank.gains().view(1, -1, 1, 1)
            dx = g
        elif v == "sphere":
            r2d = make_radius(x.device)
            dz = self.zpos[:, None] - self.zpos[None, :]          # [Z,Z]
            rho = torch.sqrt(r2d[None, None] ** 2 + dz[:, :, None, None] ** 2)
            a = sig(self.bank.a, 0.05, 0.95)[0]
            wd = sig(self.bank.w, 0.03, 0.35)[0]
            bb = sig(self.bank.b, 0.0, 1.0)[0]
            k = sum(bb[j] * torch.exp(-((rho - a[j]) / wd[j]) ** 2 / 2)
                    for j in range(a.shape[0]))
            k = k * (rho <= 1.0)
            k0 = sum(bb[j] * torch.exp(-((r2d - a[j]) / wd[j]) ** 2 / 2)
                     for j in range(a.shape[0])) * (r2d <= 1.0)
            k = k / (k0.sum() + 1e-8)      # shared norm: distant layers couple less
            weight = k * torch.tanh(self.gin)[None, :, None, None]
            xp = F.pad(x, (KS // 2,) * 4, mode="circular")
            u = F.conv2d(xp, weight)                              # [B,Z,H,W]
            g = self.bank.growth(u.reshape(B * C, 1, H, W), anneal) \
                .reshape(B, C, H, W)
            dx = self.bank.gains()[0] * g
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


def emoji_target(code, H=64, W=64, size=44):
    """Anchored emoji alpha bitmap (Twemoji PNG by codepoint, e.g. 1f642)."""
    import io as _io
    import urllib.request
    url = ("https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/72x72/"
           f"{code}.png")
    with urllib.request.urlopen(url) as r:
        img = Image.open(_io.BytesIO(r.read())).convert("RGBA")
    img = img.resize((size, size), Image.LANCZOS)
    a = np.asarray(img, np.float32)[..., 3] / 255.0
    out = np.zeros((H, W), np.float32)
    y0, x0 = (H - size) // 2, (W - size) // 2
    out[y0:y0 + size, x0:x0 + size] = a
    return out


def word_target(text, H=64, W=64, scale=1.0):
    """Anchored word bitmap (letters + faint fan scaffold) centred on the
    canvas, channel-0 alpha only. Words are positioned, not textures, so the
    trainer switches to plain MSE for these. scale upsamples the rendered
    word so letter strokes are resolvable at the kernel's scale."""
    from nca.train_staged import render_word_3_line_fan
    arr = render_word_3_line_fan(text, 12)          # [4,h,w]
    a = arr[3]
    if scale != 1.0:
        im = Image.fromarray((a * 255).astype(np.uint8))
        im = im.resize((int(a.shape[1] * scale), int(a.shape[0] * scale)),
                       Image.BILINEAR)
        a = np.asarray(im, np.float32) / 255.0
    h, w = a.shape
    out = np.zeros((H, W), np.float32)
    y0, x0 = max(0, (H - h) // 2), max(0, (W - w) // 2)
    ys, xs = min(h, H), min(w, W)
    out[y0:y0 + ys, x0:x0 + xs] = a[:ys, :xs]
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




def export_web_weights(model, variant, C, K, size, init_kind="noise",
                       scaf=None, scaf_mode="persistent", seed_xy=None):
    """Evaluate the physics to a plain-array dict for the web engine.
    Returns None for variants without an engine branch (sphere)."""
    if variant == "sphere":
        return None
    out = {"kind": "lenia", "variant": variant, "C": C, "K": K,
           "dt": model.dt, "ks": KS, "leak": 0.05, "size": size,
           "init": init_kind, "scaf_mode": scaf_mode}
    with torch.no_grad():
        bank = model.bank
        if variant in ("dyn1", "dynwave"):
            out["basis"] = model.basis[:, 0].cpu().numpy().round(6).tolist()
            w0, b0 = model.mix[0].weight, model.mix[0].bias
            w2, b2 = model.mix[2].weight, model.mix[2].bias
            out["mix"] = {
                "w0": w0[:, :, 0, 0].cpu().numpy().round(6).tolist(),
                "b0": b0.cpu().numpy().round(6).tolist(),
                "w2": w2[:, :, 0, 0].cpu().numpy().round(6).tolist(),
                "b2": b2.cpu().numpy().round(6).tolist(),
            }
        else:
            out["kernels"] = bank.kernels(
                next(model.parameters()).device).detach().cpu()                 .numpy().round(6).tolist()
        lo = -0.6 if variant in ("wave", "dynwave") else 0.0
        out["mu"] = sig(bank.mu, lo, 1.0).detach().cpu().numpy().round(6).tolist()
        out["sg"] = sig(bank.sg, 0.02, 0.35).detach().cpu().numpy().round(6).tolist()
        out["h"] = torch.tanh(bank.h).detach().cpu().numpy().round(6).tolist()
        if variant == "sharedk":
            out["H"] = torch.tanh(model.H).detach().cpu().numpy().round(6).tolist()
    if scaf is not None:
        out["scaffold"] = scaf.round(4).tolist()
    if seed_xy is not None:
        out["seed_x"], out["seed_y"] = int(seed_xy[0]), int(seed_xy[1])
    return out


# ---------------------------------------------------------------- training

def train(variant="static1", target="dots", C=1, K=3, steps=6000, batch=8,
          size=64, lr=5e-3, t_min=16, t_max=48, word_full=False, grok=False,
          cond="none", train_init=False, word_scale=1.0, scaf_strength=0.5,
          scaf_holes=0, scaf_noise=0.0, scaf_persistent=False,
          rng_seed=0, log_every=150, ckpt_every=500, snap_dir=None):
    if variant in ("static1", "dyn1", "multik", "aniso", "wave"):
        C = 1   # single-channel families; sphere/sharedk/full/dynwave keep C
    if cond == "scaffold":
        # conditioning channel: a clamped prepattern the physics can read.
        # Uniform rules cannot memorise WHERE letters go (positional info has
        # nowhere to live in ~10^2 params) — the scaffold channel supplies
        # position, the physics learns local development. Needs >=2 channels.
        C = max(C, 2)
    elif variant == "dynwave":
        C = 1
    torch.manual_seed(400 + rng_seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Lenia(variant, C=C, K=K).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"Device {device}, {variant} C={C} K={K}, {n_par} physics params")

    # word:<TEXT> targets are anchored bitmaps grown letter-by-letter
    # (curriculum "C" -> "CO" -> ... gated on normalised loss); textures use
    # the translation-invariant spectral loss.
    is_word = target.startswith("word:") or target.startswith("emoji:")
    if target.startswith("word:"):
        full = target[5:]
        stages = [full] if word_full else \
            [full[:i + 1] for i in range(len(full))]
        stage, stage_start = 0, 0
        tgt = torch.from_numpy(
            word_target(stages[0], size, size, word_scale)).to(device)
    elif target.startswith("emoji:"):
        stages = [target[6:]]
        stage, stage_start = 0, 0
        tgt = torch.from_numpy(emoji_target(target[6:], size, size)).to(device)
    else:
        tgt = torch.from_numpy(make_target(target, size, size)).to(device)
    tgt_spec = None if is_word else spec(tgt[None])
    if grok:
        # grokking recipe: constant LR + weight decay, run far past the
        # apparent plateau and watch for late phase transitions in loss_rel
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda e: 1.0)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.MultiStepLR(opt, [int(steps * 0.85)], 0.1)

    def loss_fn(x, t, ts, step=10**9):
        if is_word:
            # letter pixels are ~3% of the canvas: unweighted MSE makes the
            # all-dead board an excellent solution (census caught trained
            # word runs settling at alive=0). Upweight letter-on pixels,
            # warmed in over 2000 steps (organic-reveal lesson: full pressure
            # from step 0 collapses training the other way).
            lw = 1.0 + 8.0 * min(1.0, step / 2000.0) * (t > 0.3).float()
            return (lw * (x[:, 0] - t.expand(x.shape[0], -1, -1)) ** 2).mean()
        return F.mse_loss(spec(x[:, 0]), ts.expand(x.shape[0], -1, -1)) \
            + 2.0 * (x[:, 0].mean() - t.mean()) ** 2

    scaf = None
    if cond == "scaffold":
        scaf = (tgt * scaf_strength)[None, None]   # faint prepattern, clamped in

    init_logits = None
    if train_init:
        # jointly-learned initial board (all channels except the scaffold):
        # position lives in the STATE, dynamics in the rules
        init_logits = torch.full(
            (1, C - (1 if scaf is not None else 0), size, size), -2.0,
            device=device, requires_grad=True)

    def make_init(t):
        x = torch.rand(batch, C, size, size, device=device) * (0.15 if is_word else 0.6)
        if init_logits is not None:
            nb = init_logits.shape[1]
            x[:, :nb] = torch.sigmoid(init_logits) \
                + torch.randn(batch, nb, size, size, device=device) * 0.03
        elif is_word and cond != "scaffold":
            # bright blob at the fan origin (left-middle of the word)
            cols = (t > 0.3).any(dim=0).nonzero()
            x0c = int(cols.min()) if len(cols) else size // 2
            x[:, :, size // 2 - 2:size // 2 + 3, max(0, x0c - 2):x0c + 3] = 1.0
        if scaf is not None:
            x[:, -1:] = scaf
        return x

    if init_logits is not None:
        opt.add_param_group({"params": [init_logits]})

    # loss of an untrained noise board: the normalisation baseline that makes
    # losses comparable across targets ('loss_rel' = loss / baseline)
    with torch.no_grad():
        baseline = float(loss_fn(make_init(tgt), tgt, tgt_spec)) + 1e-8

    if snap_dir:
        Path(snap_dir).mkdir(parents=True, exist_ok=True)
        Image.fromarray(((1 - tgt.cpu().numpy()) * 255).astype(np.uint8)) \
            .resize((size * 5,) * 2, Image.NEAREST).save(Path(snap_dir) / "target.png")
    meta = RunMeta(snap_dir, f"LENIA-{variant}-{target}", "nca.train_lenia",
                   {"variant": variant, "target": target, "C": C, "K": K,
                    "steps": steps, "batch": batch, "lr": lr,
                    "rng_seed": rng_seed, "params": n_par, "cond": cond,
                    "train_init": train_init, "word_scale": word_scale,
                    "scaf_strength": scaf_strength, "scaf_holes": scaf_holes,
                    "scaf_noise": scaf_noise,
                    "scaf_persistent": scaf_persistent,
                    "size": size, "grok": grok},
                   C, 0, "noise", steps, device, tags=["lenia", variant, target])

    start_step, ck = try_resume(snap_dir, model, opt, sched, device=device)
    if is_word and ck:
        stage = min(ck.get("stage", 0), len(stages) - 1)
        stage_start = ck.get("stage_start", 0)
        tgt = torch.from_numpy(word_target(stages[stage], size, size)).to(device)

    recent = []
    stage_cap = max(1, int(steps / (len(stages) if is_word else 1) * 1.5))
    t0 = time.time()
    for step in range(start_step, steps):
        # growth-bump annealing: wide (easy gradients) -> trained width
        anneal = 0.5 + 0.5 * min(1.0, step / (steps * 0.3))
        # horizon curriculum inside [t_min, t_max]
        hi = t_min + int((t_max - t_min) * min(1.0, step / (steps * 0.5)))
        T = int(torch.randint(t_min, hi + 1, (1,)))
        x = make_init(tgt)
        # Anti-amplifier scaffold corruptions (the interrogation of phase 1
        # showed a noiseless every-step clamp admits a trivial pointwise
        # amplifier — these force actual development):
        #   holes: per-sample boxes zeroed for the WHOLE rollout, loss still
        #          on the full target => nonlocal completion required
        #   noise: fresh per-step noise => temporal integration required
        #   t0:    clamp only the first step => memory required
        scaf_b = None
        hole_mask = None
        if scaf is not None:
            scaf_b = scaf.expand(batch, 1, size, size).clone()
            if scaf_holes > 0:
                hole_mask = torch.ones(batch, 1, size, size, device=device)
                on = (tgt > 0.3).nonzero()
                for b in range(batch):
                    for _ in range(scaf_holes):
                        if len(on) == 0:
                            break
                        cy, cx = on[torch.randint(len(on), (1,))][0].tolist()
                        hole_mask[b, :, max(0, cy - 3):cy + 3,
                                  max(0, cx - 3):cx + 3] = 0.0
                scaf_b = scaf_b * hole_mask
        for ti in range(T):
            x = model.step(x, anneal)
            # DEFAULT: the scaffold is shown ONCE (t0) — a blueprint, not
            # a crutch. Persistent every-step clamping is an explicit
            # opt-in for morphogen-biology comparisons; the interrogation
            # showed it degenerates training into stencil-amplification.
            if scaf_b is not None and (ti == 0 or scaf_persistent):
                s = scaf_b
                if scaf_noise > 0:
                    s = (s + torch.randn_like(s) * scaf_noise).clamp(0, 1)
                x = torch.cat([x[:, :-1], s], 1)
        loss = loss_fn(x, tgt, tgt_spec, step)
        if hole_mask is not None:
            # hole-region completion, reported separately so gaming is visible
            hm = (1 - hole_mask[:, 0]) * (tgt > 0.3).float()
            hole_mse = float(((x[:, 0] - tgt) ** 2 * hm).sum()
                             / (hm.sum() + 1e-8))
        rel = loss.item() / baseline

        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()

        if is_word:
            recent.append(rel)
            if len(recent) > 50:
                recent.pop(0)
            avg = sum(recent) / len(recent)
            if stage < len(stages) - 1 and \
                    ((len(recent) == 50 and avg < 0.22) or
                     step - stage_start >= stage_cap):
                stage += 1; stage_start = step; recent.clear()
                print(f"=== word stage '{stages[stage]}' at step {step} "
                      f"(avg rel {avg:.3f}) ===", flush=True)
                tgt = torch.from_numpy(
                    word_target(stages[stage], size, size, word_scale)).to(device)
                if scaf is not None:
                    scaf = (tgt * scaf_strength)[None, None]
                with torch.no_grad():
                    baseline = float(loss_fn(make_init(tgt), tgt, None)) + 1e-8
                if snap_dir:
                    Image.fromarray(((1 - tgt.cpu().numpy()) * 255)
                                    .astype(np.uint8)) \
                        .resize((size * 5,) * 2, Image.NEAREST) \
                        .save(Path(snap_dir) / "target.png")

        if step % log_every == 0 or step == steps - 1:
            print(f"[lenia-{variant}-{target}] step {step} loss {loss.item():.4f} "
                  f"rel {rel:.3f} T {T} ({time.time() - t0:.1f}s)", flush=True)
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
                    # signed diverging colormap, zero pinned to mid-gray and
                    # scale fixed by max|H| — an undifferentiated matrix reads
                    # as neutral gray, not black (min-max on a flat matrix
                    # rendered noise/black before)
                    Hm = torch.tanh(model.H).detach().cpu().numpy()
                    scale = max(1e-6, float(np.abs(Hm).max()))
                    Hn = Hm / scale                       # [-1, 1]
                    rgb = np.zeros((*Hm.shape, 3), np.uint8)
                    rgb[..., 0] = (127 + 127 * np.clip(Hn, 0, 1)).astype(np.uint8)
                    rgb[..., 2] = (127 + 127 * np.clip(-Hn, 0, 1)).astype(np.uint8)
                    rgb[..., 1] = (127 * (1 - np.abs(Hn))).astype(np.uint8)
                    Image.fromarray(rgb).resize((C * 24,) * 2, Image.NEAREST) \
                        .save(Path(snap_dir) / f"COUPLING_{s}.png")
                torch.save(model.state_dict(), str(Path(snap_dir) / "latest.pth"))
                # native web export: weights.json is always current, no
                # post-hoc export pass needed
                init_kind = "scaffold" if cond == "scaffold" else \
                    ("seedblob" if is_word else "noise")
                seed_xy = None
                if init_kind == "seedblob":
                    cols = (tgt > 0.3).any(dim=0).nonzero()
                    seed_xy = (int(cols.min()) if len(cols) else size // 2,
                               size // 2)
                ww = export_web_weights(
                    model, variant, C, K, size, init_kind,
                    scaf=(tgt * scaf_strength).detach().cpu().numpy()
                    if cond == "scaffold" else None,
                    scaf_mode="persistent" if scaf_persistent else "t0",
                    seed_xy=seed_xy)
                if ww is not None:
                    import json as _json
                    with open(Path(snap_dir) / "weights.json", "w") as f:
                        _json.dump(ww, f)
                if init_logits is not None:
                    # persist the learned init board — without this the -init
                    # models cannot be reproduced from artifacts (phase-1 bug)
                    torch.save({"init_logits": init_logits.detach().cpu()},
                               str(Path(snap_dir) / "init.pth"))
                extra_log = {"word_stage": stages[stage]} if is_word else {}
                if hole_mask is not None:
                    extra_log["hole_mse"] = round(hole_mse, 4)
                meta.log(step, loss.item(), loss_rel=round(rel, 4), **extra_log)
        if snap_dir and (step % ckpt_every == 0 or step == steps - 1):
            save_checkpoint(snap_dir, step, model, opt, sched,
                            extra={"stage": stage, "stage_start": stage_start}
                            if is_word else None)

    print(f"Final loss {loss.item():.4f}")
    return model


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="static1",
                   choices=["static1", "dyn1", "multik", "sharedk", "full",
                            "aniso", "wave", "dynwave", "sphere"])
    p.add_argument("--target", default="dots",
                   help="dots|hex|tri|square, word:<TEXT> (letter curriculum) "
                        "or emoji:<CODEPOINT> (e.g. emoji:1f642)")
    p.add_argument("--channels", type=int, default=6)
    p.add_argument("--kernels", type=int, default=3)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--word-full", action="store_true",
                   help="train the whole word at once (no letter curriculum)")
    p.add_argument("--cond", default="none", choices=["none", "scaffold"],
                   help="scaffold: clamp a faint prepattern channel each step")
    p.add_argument("--train-init", action="store_true",
                   help="jointly learn the initial board (position in state)")
    p.add_argument("--word-scale", type=float, default=1.0)
    p.add_argument("--scaf-strength", type=float, default=0.5)
    p.add_argument("--scaf-holes", type=int, default=0)
    p.add_argument("--scaf-noise", type=float, default=0.0)
    p.add_argument("--scaf-persistent", action="store_true",
                   help="clamp the scaffold EVERY step (default: t0 only)")
    p.add_argument("--scaf-t0", action="store_true",
                   help="deprecated no-op: t0-only is now the default")
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--grok", action="store_true",
                   help="AdamW + weight decay + constant LR for grokking runs")
    p.add_argument("--rng-seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=150)
    p.add_argument("--snap-dir", default=None)
    a = p.parse_args()
    train(variant=a.variant, target=a.target, C=a.channels, K=a.kernels,
          steps=a.steps, word_full=a.word_full, grok=a.grok, cond=a.cond,
          train_init=a.train_init, word_scale=a.word_scale, size=a.size,
          scaf_strength=a.scaf_strength, scaf_holes=a.scaf_holes,
          scaf_noise=a.scaf_noise, scaf_persistent=a.scaf_persistent,
          rng_seed=a.rng_seed, log_every=a.log_every, snap_dir=a.snap_dir)
