"""Position-free emergence by negotiation.

Multiple partial candidates of the same word nucleate at random positions;
the local rule to learn is size-based consensus: the largest candidate
grows to completion, smaller ones yield and dissolve. Targets are built
at the input's own candidate positions, so the loss carries no absolute
position information — only the negotiation dynamic.

Batch mix: negotiation scenes (2..max candidates, distinct sizes; winner
advances by delta, losers shrink by delta), completion scenes (single
candidate advances), stability scenes (complete word stays put).
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from nca.model import NCA, to_rgba, to_rgb
from nca.train import FONT_PATH, char_color
from nca.checkpoint import save_checkpoint, try_resume
from nca.runmeta import RunMeta, export_run_weights
from nca.rollout import adaptive_rollout, fester

CANVAS = 64


def render_word(text, glyph=12):
    """Premultiplied RGBA [4,h,w] of the word, tightly cropped."""
    font = ImageFont.truetype(FONT_PATH, glyph)
    img = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    x = 2
    for ch in text:
        draw.text((x, 2), ch, font=font, fill=char_color(ch) + (255,))
        l, t, r, b = draw.textbbox((x, 2), ch, font=font)
        x = r + 2
    arr = np.asarray(img, dtype=np.float32) / 255.0
    ys, xs = np.where(arr[..., 3] > 0.05)
    arr = arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    arr[..., :3] *= arr[..., 3:]
    return arr.transpose(2, 0, 1)


def reveal_order(word):
    """Pixel indices of the word sorted by distance from its center —
    growth reveals nearest-first, shrink hides farthest-first."""
    a = word[3]
    ys, xs = np.where(a > 0.05)
    cy, cx = ys.mean(), xs.mean()
    d = (ys - cy) ** 2 + (xs - cx) ** 2
    order = np.argsort(d)
    return ys[order], xs[order]


def partial(word, ys, xs, frac):
    """Word revealed to the given fraction (nearest-first)."""
    out = np.zeros_like(word)
    k = int(len(ys) * np.clip(frac, 0.0, 1.0))
    if k > 0:
        out[:, ys[:k], xs[:k]] = word[:, ys[:k], xs[:k]]
    return out


def place(canvas, patch, y, x):
    _, h, w = patch.shape
    canvas[:, y:y + h, x:x + w] = np.maximum(canvas[:, y:y + h, x:x + w], patch)


def sample_scene(word, ys, xs, rng, max_cands=3, delta=0.15, nucleate_p=0.0):
    """Returns (input RGBA, target RGBA) as [4,CANVAS,CANVAS] numpy."""
    _, h, w = word.shape
    kind = rng.random()
    inp = np.zeros((4, CANVAS, CANVAS), np.float32)
    tgt = np.zeros((4, CANVAS, CANVAS), np.float32)

    if kind < nucleate_p:
        # nucleation: noise condenses into a few small candidates. Target
        # positions are random (not derivable from the input), so the model
        # can only learn the statistical rule: noise -> sparse proto-seeds.
        inp[:] = rng.random((4, CANVAS, CANVAS)) * rng.uniform(0.3, 0.8)
        for _ in range(int(rng.integers(2, 4))):
            f = rng.uniform(0.12, 0.3)
            y = int(rng.integers(0, CANVAS - h)); x = int(rng.integers(0, CANVAS - w))
            place(tgt, partial(word, ys, xs, f), y, x)
        return inp, tgt
    kind = (kind - nucleate_p) / max(1e-9, 1 - nucleate_p)

    def rand_pos(taken):
        for _ in range(40):
            y = rng.integers(0, CANVAS - h)
            x = rng.integers(0, CANVAS - w)
            if all(abs(y - ty) > h * 0.7 or abs(x - tx) > w * 0.7 for ty, tx in taken):
                return y, x
        return None

    if kind < 0.10:
        # stability: complete word holds
        y, x = rng.integers(0, CANVAS - h), rng.integers(0, CANVAS - w)
        place(inp, word, y, x)
        place(tgt, word, y, x)
    elif kind < 0.30:
        # completion: single partial candidate advances
        f = rng.uniform(0.2, 0.85)
        y, x = rng.integers(0, CANVAS - h), rng.integers(0, CANVAS - w)
        place(inp, partial(word, ys, xs, f), y, x)
        place(tgt, partial(word, ys, xs, min(1.0, f + delta)), y, x)
    else:
        # negotiation: winner (distinctly largest) advances, losers shrink
        n = int(rng.integers(2, max_cands + 1))
        fr = np.sort(rng.uniform(0.15, 0.6, n))
        fr[-1] = min(0.9, fr[-1] + 0.15)   # winner clearly larger
        taken = []
        for i in range(n):
            pos = rand_pos(taken)
            if pos is None:
                continue
            taken.append(pos)
            y, x = pos
            place(inp, partial(word, ys, xs, fr[i]), y, x)
            if i == n - 1:
                place(tgt, partial(word, ys, xs, min(1.0, fr[i] + delta)), y, x)
            else:
                place(tgt, partial(word, ys, xs, max(0.0, fr[i] - delta)), y, x)
    return inp, tgt


def build_state(rgba, channel_n, rng, hidden_noise=0.1):
    x = np.zeros((channel_n, CANVAS, CANVAS), np.float32)
    x[:4] = rgba
    x[4:] = rng.random((channel_n - 4, CANVAS, CANVAS)).astype(np.float32) * hidden_noise
    # hidden channels only where something is alive, so dead space stays dead
    x[4:] *= (rgba[3:4] > 0.05)
    return x


def make_self_batch(model, word, ys, xs, rng, batch, channel_n, delta, device):
    """Self-refereed scenes: run the model on noise, find its own strongest
    blob via 0-dim persistence, and build the target there — advance the
    winner the model actually made, dissolve everything else."""
    from nca.topology import persistence_0d
    _, h, w = word.shape
    with torch.no_grad():
        z = torch.rand(batch, channel_n, CANVAS, CANVAS, device=device)
        z = model(z, steps=int(rng.integers(24, 64)))
    tgts = np.zeros((batch, 4, CANVAS, CANVAS), np.float32)
    for i in range(batch):
        a = z[i, 3].clamp(0, 1).cpu().numpy()
        events = persistence_0d(a, min_persistence=0.05)
        if events:
            # prefer peaks that fit without clipping — clipped placement
            # taught the model edge-seeking (candidates migrated to the
            # canvas border and flared out there)
            fit = [e for e in events
                   if h // 2 <= e[2][0] <= CANVAS - h + h // 2
                   and w // 2 <= e[2][1] <= CANVAS - w + w // 2]
            pool_ev = fit if fit else events
            b_, d_, (py, px) = max(pool_ev, key=lambda e: e[0] - e[1])
            y0 = int(np.clip(py - h // 2, 0, CANVAS - h))
            x0 = int(np.clip(px - w // 2, 0, CANVAS - w))
            p = partial_np_local(word, ys, xs, 0.3 + delta)
            tgts[i, :, y0:y0 + h, x0:x0 + w] = p
    return z.detach(), torch.from_numpy(tgts).to(device)


def partial_np_local(word, ys, xs, frac):
    out = np.zeros_like(word)
    k = int(len(ys) * np.clip(frac, 0, 1))
    if k:
        out[:, ys[:k], xs[:k]] = word[:, ys[:k], xs[:k]]
    return out


def train(text="CO", steps=8000, glyph=14, channel_n=16, hidden_n=80,
          batch=16, lr=2e-3, ca_min=12, ca_max=24, max_cands=3, delta=0.15,
          nucleate_p=0.0, self_p=0.0, adaptive=False, fester_p=0.0,
          log_every=100, snap_dir=None, rng_seed=0):
    torch.manual_seed(sum(map(ord, text)) + 5)
    rng = np.random.default_rng(rng_seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")

    word = render_word(text, glyph)
    ys, xs = reveal_order(word)

    if snap_dir:
        Path(snap_dir).mkdir(parents=True, exist_ok=True)
        full = np.zeros((4, CANVAS, CANVAS), np.float32)
        place(full, word, (CANVAS - word.shape[1]) // 2, (CANVAS - word.shape[2]) // 2)
        vis = (1 - full[3] + full[:3]).clip(0, 1).transpose(1, 2, 0)
        Image.fromarray((vis * 255).astype(np.uint8)) \
            .resize((CANVAS * 6, CANVAS * 6), Image.NEAREST) \
            .save(Path(snap_dir) / "target.png")

    model = NCA(channel_n, hidden_n=hidden_n).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=[int(steps * 0.8)], gamma=0.1)

    start_step, _ = try_resume(snap_dir, model, opt, sched, device=device)

    meta = RunMeta(snap_dir, text, "nca.train_negotiate",
                   {"steps": steps, "glyph": glyph, "batch": batch, "lr": lr,
                    "max_cands": max_cands, "delta": delta,
                    "nucleate_p": nucleate_p, "self_p": self_p,
                    "rng_seed": rng_seed, "adaptive": adaptive,
                    "fester_p": fester_p},
                   channel_n, hidden_n, "negotiate", steps, device,
                   tags=["negotiate"] + (["adaptive"] if adaptive else []))

    t0 = time.time()
    for step in range(start_step, steps):
        if self_p > 0 and rng.random() < self_p:
            x, tgt = make_self_batch(model, word, ys, xs, rng, batch,
                                     channel_n, delta, device)
        else:
            inps, tgts = [], []
            for _ in range(batch):
                i, t = sample_scene(word, ys, xs, rng, max_cands, delta, nucleate_p)
                inps.append(build_state(i, channel_n, rng))
                tgts.append(t)
            x = torch.from_numpy(np.stack(inps)).to(device)
            tgt = torch.from_numpy(np.stack(tgts)).to(device)
        x_start = x[:1].detach().clone()

        if fester_p > 0 and torch.rand(1).item() < fester_p:
            x = fester(model, x, min_steps=100, max_steps=400)
        if adaptive:
            x, _used = adaptive_rollout(model, x, tgt, chunk=6, max_chunks=8)
        else:
            n_ca = int(torch.randint(ca_min, ca_max + 1, (1,)))
            x = model(x, steps=n_ca)
        loss = F.mse_loss(to_rgba(x), tgt)

        opt.zero_grad()
        loss.backward()
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    p.grad /= (p.grad.norm() + 1e-8)
        opt.step()
        sched.step()

        if step % log_every == 0 or step == steps - 1:
            print(f"[negotiate_{text}] step {step} loss {loss.item():.5f} "
                  f"({time.time() - t0:.1f}s)", flush=True)
            if snap_dir:
                s = f"{step:05d}"
                for tag, t in [("START", to_rgba(x_start)[0]),
                               ("COMP", to_rgba(x)[0]),
                               ("TARGET", tgt[0])]:
                    img = t.detach().cpu().clamp(0, 1)
                    vis = (1 - img[3:4] + img[:3]).clamp(0, 1).permute(1, 2, 0).numpy()
                    Image.fromarray((vis * 255).astype(np.uint8)) \
                        .resize((CANVAS * 6, CANVAS * 6), Image.NEAREST) \
                        .save(Path(snap_dir) / f"{tag}_{s}.png")
                torch.save(model.state_dict(), str(Path(snap_dir) / "latest.pth"))
                save_checkpoint(snap_dir, step, model, opt, sched)
                meta.log(step, loss.item())
                export_run_weights(model, snap_dir, text, glyph,
                                   grid_w=CANVAS, grid_h=CANVAS, seed_type="noise")

    print(f"Final loss for {text} (negotiate): {loss.item():.5f}")
    return model


def preview_scenes(out_dir, text="CO", glyph=14, n=8, max_cands=3, delta=0.15,
                   rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    word = render_word(text, glyph)
    ys, xs = reveal_order(word)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        inp, tgt = sample_scene(word, ys, xs, rng, max_cands, delta)
        pair = np.concatenate([inp, np.ones((4, CANVAS, 2), np.float32) * 0.5, tgt], axis=2)
        vis = (1 - pair[3] + pair[:3]).clip(0, 1).transpose(1, 2, 0)
        Image.fromarray((vis * 255).astype(np.uint8)) \
            .resize((vis.shape[1] * 5, vis.shape[0] * 5), Image.NEAREST) \
            .save(out / f"PAIR_{i:02d}.png")
    print(f"wrote {n} scene pairs to {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--text", default="CO")
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--max-cands", type=int, default=3)
    p.add_argument("--delta", type=float, default=0.15)
    p.add_argument("--nucleate-p", type=float, default=0.0,
                   help="Fraction of batches teaching noise -> proto-seeds")
    p.add_argument("--self-p", type=float, default=0.0,
                   help="Fraction of self-refereed batches (persistence-picked winner)")
    p.add_argument("--adaptive", action="store_true")
    p.add_argument("--fester-p", type=float, default=0.0)
    p.add_argument("--rng-seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--snap-dir", default=None)
    p.add_argument("--preview", action="store_true")
    a = p.parse_args()

    if a.preview:
        preview_scenes(a.snap_dir or "snaps_negotiate_preview", text=a.text,
                       max_cands=a.max_cands, delta=a.delta, rng_seed=a.rng_seed)
    else:
        train(a.text, steps=a.steps, max_cands=a.max_cands, delta=a.delta,
              nucleate_p=a.nucleate_p, self_p=a.self_p, adaptive=a.adaptive,
              fester_p=a.fester_p, rng_seed=a.rng_seed,
              log_every=a.log_every, snap_dir=a.snap_dir)
