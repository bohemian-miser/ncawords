"""Experiment 2: static rules, trained machinery.

We freeze the Game-of-Life surrogate learned in Experiment 1 (train_gol) —
that is now a fixed, differentiable physics. Nothing about the rules is
trained. Instead we train an INITIAL STATE: a soft board whose cells are
optimised so that, evolved under the frozen physics, it produces a desired
behaviour in a masked target region. The unmasked cells are free to become
whatever machinery (reflectors, emitters, still-lifes) the physics needs.

Targets:
  emit   - free board everywhere; a glider must appear in a target box at
           time T. 'Build a machine that emits a glider here.'
  return - a glider is GIVEN and fixed at the start; the free cells must be
           trained into machinery that makes a glider arrive in a return box
           at time T. 'Give it the glider, it builds the reflection stuff.'

Honesty guardrails (a continuous surrogate will happily cheat with
fractional 'half-alive' cells that are illegal in real Life):
  - a bimodality penalty pushes the trained board to {0,1};
  - every log step the board is ROUNDED to binary and run through the REAL
    discrete Life rule for T steps; the masked match against the target under
    true Life is the metric we actually believe.
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

from nca.train_gol import GoLNet, GLIDER, gol_step
from nca.runmeta import RunMeta

BUCKET = "https://storage.googleapis.com/recipe-lanes-nca-jobs"

_MOORE = torch.tensor([[1., 1, 1], [1, 0, 1], [1, 1, 1]]).view(1, 1, 3, 3)


def soft_life_step(b, k):
    """Analytic differentiable Life at temperature k. Same fixed rules — born
    at 3 neighbours, survive at 2 or 3 — rendered as smooth bumps so gradients
    flow. As k -> inf this approaches the exact discrete rule; small k is soft
    (well-conditioned for optimising the input). Toroidal."""
    kern = _MOORE.to(b.device)
    n = F.conv2d(F.pad(b, (1, 1, 1, 1), mode="circular"), kern)
    born = torch.exp(-k * (n - 3) ** 2)
    surv = torch.exp(-k * (n - 3) ** 2) + torch.exp(-k * (n - 2) ** 2)
    return (b * surv + (1 - b) * born).clamp(0, 1)


def load_frozen(surrogate, hidden_n, perception):
    """Load the Experiment-1 surrogate weights (frozen physics)."""
    model = GoLNet(hidden_n, perception, vis=1, hid=0)
    with urllib.request.urlopen(f"{BUCKET}/{surrogate}/latest.pth") as r:
        sd = torch.load(io.BytesIO(r.read()), map_location="cpu", weights_only=True)
    model.load_state_dict(sd)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    return model


def glider_stamp(H, W, r, c, phase=0):
    """A glider (its 4 phases via real Life) placed with top-left at (r,c)."""
    b = np.zeros((1, H, W), np.float32)
    b[0, r:r + 3, c:c + 3] = GLIDER
    for _ in range(phase):
        b = gol_step(b)
    return b[0]


def build_target(kind, H, W, T):
    """Return (target_board, mask, given_board, given_mask). given_* is the
    fixed part of the initial state (empty for 'emit')."""
    given = np.zeros((H, W), np.float32)
    given_mask = np.zeros((H, W), np.float32)
    tgt = np.zeros((H, W), np.float32)
    mask = np.zeros((H, W), np.float32)

    if kind == "grow":
        # trivially-reachable LOCAL sanity target: a 2x2 block (a still life)
        # in a central box at time T. The solution is 'a block there at t=0'.
        # If gradient can't even hold a still life, the pipeline is broken.
        cr, cc = H // 2, W // 2
        tgt[cr:cr + 2, cc:cc + 2] = 1.0
        mask[cr - 3:cr + 5, cc - 3:cc + 5] = 1.0
    elif kind == "emit":
        # target: a glider in a box near the bottom-right at time T
        br, bc = H - 8, W - 8
        tgt = glider_stamp(H, W, br, bc, phase=T % 4)
        mask[br - 1:br + 5, bc - 1:bc + 5] = 1.0
    elif kind == "return":
        # given: a glider fixed at top-left, heading down-right
        given = glider_stamp(H, W, 2, 2, phase=0)
        given_mask[1:7, 1:7] = 1.0
        # target: a glider back in a return box near the top-left at time T
        rr, rc = 3, 3
        tgt = glider_stamp(H, W, rr, rc, phase=T % 4)
        mask[rr - 2:rr + 5, rc - 2:rc + 5] = 1.0
    else:
        raise ValueError(kind)
    return tgt, mask, given, given_mask


def train(surrogate="gol-both", perception="both", hidden_n=64, kind="return",
          physics="soft", H=40, W=40, horizon=24, steps=4000, lr=0.05,
          bimodal_w=0.5, parsimony_w=0.02, k_lo=2.0, k_hi=14.0,
          init_bias=-2.0, curriculum=True, rng_seed=0, log_every=100, snap_dir=None):
    torch.manual_seed(100 + rng_seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device {device}, physics {physics}, target {kind}, T {horizon}")

    # 'soft': analytic temperature-annealed Life (good gradients). 'surrogate':
    # the frozen Experiment-1 net (faithful but saturated -> weak gradients).
    phys_net = load_frozen(surrogate, hidden_n, perception).to(device) \
        if physics == "surrogate" else None

    tgt_np, mask_np, given_np, gmask_np = build_target(kind, H, W, horizon)
    tgt = torch.from_numpy(tgt_np).to(device)[None, None]
    mask = torch.from_numpy(mask_np).to(device)[None, None]
    given = torch.from_numpy(given_np).to(device)[None, None]
    gmask = torch.from_numpy(gmask_np).to(device)[None, None]

    # the trainable machinery: a soft board (logits). Free cells only; the
    # given glider is stamped in each forward and not trained.
    logits = torch.zeros(1, 1, H, W, device=device, requires_grad=True)
    torch.nn.init.normal_(logits, init_bias, 0.8)   # init_bias<0 => mostly dead
    opt = torch.optim.Adam([logits], lr=lr)

    if snap_dir:
        Path(snap_dir).mkdir(parents=True, exist_ok=True)
        panel = np.concatenate([given_np, tgt_np, mask_np], axis=1)
        Image.fromarray(((1 - panel) * 255).astype(np.uint8)) \
            .resize((panel.shape[1] * 6, panel.shape[0] * 6), Image.NEAREST) \
            .save(Path(snap_dir) / "target.png")
    meta = RunMeta(snap_dir, "GOLMACH", "nca.train_gol_machine",
                   {"surrogate": surrogate, "kind": kind, "horizon": horizon,
                    "steps": steps, "lr": lr, "bimodal_w": bimodal_w,
                    "H": H, "W": W, "rng_seed": rng_seed},
                   1, hidden_n, "board", steps, device,
                   tags=["gol", "machine", kind])

    def make_board(train_mode=True):
        free = torch.sigmoid(logits)
        # stamp the fixed given-glider into its region (detached)
        return free * (1 - gmask) + given * gmask

    t0 = time.time()
    for step in range(steps):
        board = make_board()
        # anneal the soft-Life temperature soft -> sharp over training
        k = k_lo + (k_hi - k_lo) * min(1.0, step / (steps * 0.7))
        # horizon curriculum: grow the rollout 1 -> horizon so credit
        # assignment starts trivial. (Clean only for phase/position-invariant
        # targets like the 'grow' still-life; emit/return use fixed horizon.)
        eff_T = horizon
        if curriculum:
            eff_T = 1 + int((horizon - 1) * min(1.0, step / (steps * 0.5)))
        x = board
        for _ in range(eff_T):
            x = phys_net.step(x) if physics == "surrogate" else soft_life_step(x, k)
        end = x
        masked = F.binary_cross_entropy(
            (end * mask).clamp(1e-6, 1 - 1e-6), tgt * mask)
        bimodal = (make_board() * (1 - make_board())).mean()
        # parsimony: prefer few live free-cells (sparse machinery)
        pars = (torch.sigmoid(logits) * (1 - gmask)).mean()
        # warm the regularizers in: let machinery form first, sparsify/binarize
        # later — full pressure from step 0 collapses the board before any
        # reflector can appear.
        rw = min(1.0, step / (steps * 0.5))
        loss = masked + rw * bimodal_w * bimodal + rw * parsimony_w * pars

        opt.zero_grad(); loss.backward(); opt.step()

        if step % log_every == 0 or step == steps - 1:
            with torch.no_grad():
                # HONEST metric: round to binary, run REAL Life, score under mask
                b0 = (make_board() > 0.5).float().cpu().numpy()[0, 0]
                bb = b0[None].copy()
                for _ in range(horizon):
                    bb = gol_step(bb)
                real_end = bb[0]
                m = mask_np > 0.5
                real_match = float((( (real_end > 0.5) == (tgt_np > 0.5) )[m]).mean())
                live = int(b0.sum())
            print(f"[golmach-{kind}] step {step} bce {masked.item():.4f} "
                  f"bimod {bimodal.item():.3f} real_match {real_match:.3f} "
                  f"live {live} ({time.time()-t0:.1f}s)", flush=True)
            if snap_dir:
                with torch.no_grad():
                    # filmstrip of the REAL Life rollout from the rounded board
                    frames = [b0.copy()]
                    bb = b0[None].copy()
                    for _ in range(horizon):
                        bb = gol_step(bb); frames.append(bb[0].copy())
                    idx = np.linspace(0, horizon, 7).astype(int)
                    strip = np.concatenate([frames[i] for i in idx], axis=1)
                Image.fromarray(((1 - strip) * 255).astype(np.uint8)) \
                    .resize((strip.shape[1] * 5, strip.shape[0] * 5), Image.NEAREST) \
                    .save(Path(snap_dir) / f"GOL_{step:05d}.png")
                np.save(Path(snap_dir) / "board0.npy", b0)
                torch.save({"logits": logits.detach().cpu()},
                           str(Path(snap_dir) / "latest.pth"))
                meta.log(step, masked.item(), real_match=round(real_match, 3),
                         live=live)

    print(f"Final real_match {real_match:.3f}")
    return logits


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--surrogate", default="gol-both")
    p.add_argument("--perception", default="both")
    p.add_argument("--physics", default="soft", choices=["soft", "surrogate"])
    p.add_argument("--kind", default="return", choices=["emit", "return", "grow"])
    p.add_argument("--horizon", type=int, default=24)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--bimodal-w", type=float, default=0.5)
    p.add_argument("--init-bias", type=float, default=-2.0)
    p.add_argument("--no-curriculum", dest="curriculum", action="store_false")
    p.add_argument("--rng-seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--snap-dir", default=None)
    a = p.parse_args()
    train(surrogate=a.surrogate, perception=a.perception, physics=a.physics,
          kind=a.kind, horizon=a.horizon, steps=a.steps, lr=a.lr,
          bimodal_w=a.bimodal_w, init_bias=a.init_bias, curriculum=a.curriculum,
          rng_seed=a.rng_seed, log_every=a.log_every, snap_dir=a.snap_dir)
