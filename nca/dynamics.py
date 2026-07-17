"""Dynamical-systems analysis of trained NCAs.

Treats the deterministic update F (fire_rate=1) as a discrete dynamical
system on grid states and measures, per trained model:

- fixed-point residual  ||F(x*) - x*|| / ||x*||  after settling
- spectral radius rho of the Jacobian dF/dx at x* (power iteration with
  autograd JVPs; rho < 1 => linearly stable attractor)
- largest Lyapunov exponent (Benettin renormalization along a rollout;
  positive => chaotic sensitivity, negative => self-correcting)
- damage basin: largest circular zero-out radius from which the model
  still recovers to x* (bisection) — a direct 'healing capacity' number

Usage:
  python -m nca.dynamics --run base-hid-comp-noise-8k [--start seed|noise]
"""
import argparse
import io
import json
import urllib.request

import numpy as np
import torch

from nca.model import NCA

BUCKET = "https://storage.googleapis.com/recipe-lanes-nca-jobs"


def load_run(run):
    with urllib.request.urlopen(f"{BUCKET}/{run}/weights.json") as r:
        meta = json.load(r)
    with urllib.request.urlopen(f"{BUCKET}/{run}/latest.pth") as r:
        blob = r.read()
    sd = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=True)
    c_n = sd["fc1.weight"].shape[0]
    h_n = sd["fc1.weight"].shape[1]
    model = NCA(c_n, hidden_n=h_n)
    model.load_state_dict(sd)
    model.eval()
    H = meta.get("grid_h") or 40
    W = meta.get("grid_w") or 100
    return model, H, W, meta


def initial_state(model, H, W, start):
    x = torch.zeros(1, model.channel_n, H, W)
    if start == "noise":
        x = torch.rand(1, model.channel_n, H, W)
    else:
        x[:, 3:, H // 2, W // 2] = 1.0
    return x


def F(model, x):
    return model.step(x, fire_rate=1.0)


@torch.no_grad()
def settle(model, x, steps=400):
    for _ in range(steps):
        x = F(model, x)
    return x


@torch.no_grad()
def residual(model, x):
    nx = torch.linalg.vector_norm(x)
    if nx == 0:
        return float("nan")
    return float(torch.linalg.vector_norm(F(model, x) - x) / nx)


def spectral_radius(model, x_star, iters=40):
    """Dominant |eigenvalue| of dF/dx at x_star via JVP power iteration."""
    v = torch.randn_like(x_star)
    v /= torch.linalg.vector_norm(v)
    rho = float("nan")
    for _ in range(iters):
        _, jv = torch.autograd.functional.jvp(
            lambda y: F(model, y), (x_star,), (v,), create_graph=False)
        n = torch.linalg.vector_norm(jv)
        if float(n) == 0.0:
            return 0.0
        rho = float(n)
        v = (jv / n).detach()
    return rho


@torch.no_grad()
def lyapunov(model, x0, steps=200, eps=1e-4):
    """Benettin: renormalized separation growth along the trajectory."""
    x = x0.clone()
    d = torch.randn_like(x0)
    d = d / torch.linalg.vector_norm(d) * eps
    y = x0 + d
    logs = []
    for _ in range(steps):
        x = F(model, x)
        y = F(model, y)
        delta = y - x
        n = float(torch.linalg.vector_norm(delta))
        if n == 0.0:
            logs.append(-23.0)  # fully collapsed onto reference
            y = x + torch.randn_like(x) / torch.linalg.vector_norm(torch.randn_like(x)) * eps
            continue
        logs.append(np.log(n / eps))
        y = x + delta / n * eps
    return float(np.mean(logs))


@torch.no_grad()
def damage_basin(model, x_star, recover_steps=300, tol=0.15):
    """Largest circular zero-out radius the model heals back to x_star."""
    _, _, H, W = x_star.shape
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    ref = torch.linalg.vector_norm(x_star)

    def recovers(r):
        mask = (((yy - H / 2) ** 2 + (xx - W / 2) ** 2) > r * r).float()
        x = x_star * mask[None, None]
        x = settle(model, x, recover_steps)
        return float(torch.linalg.vector_norm(x - x_star) / ref) < tol

    lo, hi = 0.0, float(min(H, W)) / 2
    if not recovers(2.0):
        return 0.0
    lo = 2.0
    for _ in range(7):
        mid = (lo + hi) / 2
        if recovers(mid):
            lo = mid
        else:
            hi = mid
    return lo


def analyze(run, start="seed", settle_steps=400):
    model, H, W, meta = load_run(run)
    x0 = initial_state(model, H, W, start)
    x_star = settle(model, x0, settle_steps)
    alive = float((x_star[:, 3] > 0.1).float().mean())
    res = residual(model, x_star)
    rho = spectral_radius(model, x_star)
    lam = lyapunov(model, x_star)
    basin = damage_basin(model, x_star) if alive > 0.01 else 0.0
    return {
        "run": run, "start": start, "grid": f"{W}x{H}",
        "alive_frac": round(alive, 3),
        "fixed_point_residual": round(res, 5),
        "spectral_radius": round(rho, 4),
        "lyapunov": round(lam, 4),
        "damage_basin_radius": round(basin, 1),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--start", default="seed", choices=["seed", "noise"])
    p.add_argument("--settle-steps", type=int, default=400)
    a = p.parse_args()
    print(json.dumps(analyze(a.run, a.start, a.settle_steps), indent=2))
