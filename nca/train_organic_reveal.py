"""Organic exploration that reveals letters as it finds them.

A support structure grows outward from a seed with a rotating directional
bias. When it touches a character, that character fades in over a few
increments and its pixels join the growth frontier. Training is
state -> next-state over a persistent pool of trajectories: the model
receives its own state at increment k and is trained toward the target at
increment k+1, so it learns the growth process itself.

Loss: alpha (presence) everywhere; RGB only on letter pixels — the color
of the organic support is unconstrained ("something is there" is enough).
Each pool trajectory runs at a fixed 0/90/180/270 deg rotation.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from nca.model import NCA, to_rgb
from nca.train import FONT_PATH, char_color
from nca.checkpoint import save_checkpoint, try_resume
from nca.runmeta import RunMeta, export_run_weights
from nca.rollout import adaptive_rollout

CANVAS = 72          # square so 90-degree rotations are exact
SUPPORT_ALPHA = 0.6  # target presence level for support-only cells
REVEAL_STEPS = 6     # increments for a touched character to fade in


def render_chars(text, glyph=12):
    """Per-character premultiplied-RGBA layers centered on the square canvas."""
    font = ImageFont.truetype(FONT_PATH, glyph)
    pitch, n = 14, len(text)
    x0 = (CANVAS - pitch * n) // 2 + pitch // 2
    layers = []
    for i, ch in enumerate(text):
        img = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        l, t, r, b = draw.textbbox((0, 0), ch, font=font)
        x = x0 + pitch * i - (r - l) / 2 - l
        y = (CANVAS - (b - t)) / 2 - t
        draw.text((x, y), ch, font=font, fill=char_color(ch) + (255,))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr[..., :3] *= arr[..., 3:]
        layers.append(arr.transpose(2, 0, 1))  # [4, H, W]
    return layers


BACKBONE_ALPHA = 0.85  # letter-connecting network strengthens to this


def bfs_dist(net, sources, neigh_off):
    """Chebyshev BFS distance through net from source cells (inf elsewhere)."""
    dist = np.full(net.shape, np.inf, np.float32)
    frontier = sources & net
    dist[frontier] = 0
    d = 0
    while frontier.any():
        d += 1
        dil = np.zeros_like(frontier)
        for dy, dx in neigh_off:
            dil |= np.roll(np.roll(frontier, dy, 0), dx, 1)
        newly = dil & net & (dist > d)
        if not newly.any():
            break
        dist[newly] = d
        frontier = newly
    return dist


def letter_paths(net, char_pix_on, neigh_off):
    """Cells on shortest paths through net between consecutive letters."""
    paths = np.zeros(net.shape, bool)
    for a, b in zip(char_pix_on[:-1], char_pix_on[1:]):
        da = bfs_dist(net, a, neigh_off)
        db = bfs_dist(net, b, neigh_off)
        total = da + db
        if not np.isfinite(total).any():
            continue   # not connected yet
        paths |= total <= (total[np.isfinite(total)].min() + 1)
    return paths


def grow_frames(text, K=60, glyph=12, rng_seed=0, join_p=0.4, bias_gain=0.35,
                bias_turns=1.5, growth="bfs", step_len=3, branch_p=0.25,
                max_walkers=20, max_dist=None, lifespan=None):
    """Precompute the K target frames of the exploration process.

    growth='bfs': probabilistic frontier expansion (blob with ragged edges).
    growth='dfs': depth-first tendrils — walkers advance in a persistent
    direction, occasionally branching; revealed letters spawn new walkers
    from their border.

    Returns rgb [K,3,H,W], alpha [K,1,H,W], rgb_mask [1,H,W] (letter pixels).
    """
    rng = np.random.default_rng(rng_seed)
    H = W = CANVAS
    layers = render_chars(text, glyph)
    char_pix = [l[3] > 0.05 for l in layers]

    mask = np.zeros((H, W), bool)
    mask[H // 2, W // 2] = True
    birth = np.full((H, W), -1, np.int32)   # frame each support cell joined
    birth[H // 2, W // 2] = 0
    veins = np.zeros((H, W), bool)          # sticky letter-connecting network
    reveal_at = [None] * len(layers)   # increment when char was touched
    walkers = [(H / 2, W / 2, rng.uniform(0, 2 * np.pi))]

    frames_rgb, frames_a = [], []
    neigh_off = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    # Exploration is confined to within max_dist of the letters (default one
    # letter-height) so growth hugs the text and finds every character.
    if max_dist is None:
        letter_h = max(int(np.ptp(np.where(p.any(1))[0])) + 1 for p in char_pix)
        max_dist = letter_h
    all_letters = np.zeros((H, W), bool)
    for p in char_pix:
        all_letters |= p
    dist = np.full((H, W), 1e9, np.float32)
    dist[all_letters] = 0
    frontier = all_letters.copy()
    d = 0
    while frontier.any():
        d += 1
        dil = np.zeros_like(frontier)
        for dy, dx in neigh_off:
            dil |= np.roll(np.roll(frontier, dy, 0), dx, 1)
        newly = dil & (dist > d)
        if not newly.any():
            break
        dist[newly] = d
        frontier = newly
    allowed = dist <= max_dist

    for k in range(K):
        if growth == "dfs":
            # Tendril walkers: persistent heading + jitter, occasional branch.
            nxt = []
            for (y, x, ang) in walkers:
                for _ in range(step_len):
                    ang += rng.normal(0, 0.45)
                    ny = y + np.sin(ang)
                    nx_ = x + np.cos(ang)
                    if not (0 <= ny < H and 0 <= nx_ < W) \
                            or not allowed[int(ny), int(nx_)]:
                        ang += np.pi + rng.normal(0, 0.5)  # bounce off the band
                        continue
                    y, x = ny, nx_
                    mask[int(y), int(x)] = True
                nxt.append((y, x, ang))
                if rng.random() < branch_p and len(nxt) < max_walkers:
                    turn = rng.choice([-1, 1]) * rng.uniform(np.pi / 3, 2 * np.pi / 3)
                    nxt.append((y, x, ang + turn))
            walkers = nxt[:max_walkers]
        else:
            # Frontier expansion with a rotating directional bias.
            theta = 2 * np.pi * bias_turns * k / K
            bias = np.array([np.sin(theta), np.cos(theta)])  # (dy, dx)
            grow = np.zeros_like(mask)
            for dy, dx in neigh_off:
                cand = np.roll(np.roll(mask, dy, 0), dx, 1) & ~mask
                d = np.array([dy, dx]) / np.hypot(dy, dx)
                p = np.clip(join_p + bias_gain * float(d @ bias), 0.05, 0.95)
                grow |= cand & (rng.random((H, W)) < p)
            mask |= grow & allowed

        # Characters touched by the mask start fading in; fully revealed
        # characters join the mask so growth continues from their border.
        for ci, pix in enumerate(char_pix):
            if reveal_at[ci] is None and (mask & pix).any():
                reveal_at[ci] = k
            if reveal_at[ci] is not None and k - reveal_at[ci] == REVEAL_STEPS:
                mask |= pix
                if growth == "dfs":
                    ys, xs = np.where(pix)
                    for _ in range(2):
                        j = rng.integers(len(ys))
                        walkers.append((float(ys[j]), float(xs[j]),
                                        rng.uniform(0, 2 * np.pi)))
            elif reveal_at[ci] is not None and k - reveal_at[ci] > REVEAL_STEPS:
                mask |= pix

        newly_joined = mask & (birth < 0)
        birth[newly_joined] = k

        # Lifespan: scout cells die at age `lifespan` unless they lie on a
        # shortest path between adjacent letters — those become sticky veins
        # that persist and strengthen while everything else churns.
        if lifespan:
            letters_on = np.zeros_like(mask)
            on_list = []
            for ci, pix in enumerate(char_pix):
                if reveal_at[ci] is not None and k - reveal_at[ci] >= REVEAL_STEPS:
                    letters_on |= pix
                    on_list.append(pix)
            if len(on_list) >= 2:
                veins |= letter_paths(mask | letters_on, on_list, neigh_off) & mask
            dying = mask & ~letters_on & ~veins & (k - birth >= lifespan)
            mask &= ~dying
            birth[dying] = -1

        rgb = np.zeros((3, H, W), np.float32)
        alpha = np.zeros((H, W), np.float32)
        alpha[mask] = SUPPORT_ALPHA
        alpha[mask & veins] = BACKBONE_ALPHA
        for ci, layer in enumerate(layers):
            if reveal_at[ci] is None:
                continue
            rev = min(1.0, (k - reveal_at[ci] + 1) / REVEAL_STEPS)
            a = layer[3] * rev
            on = a > alpha
            for c in range(3):
                rgb[c] = np.where(on, layer[c] * rev, rgb[c])
            alpha = np.maximum(alpha, a)
        frames_rgb.append(rgb)
        frames_a.append(alpha[None])

    rgb_mask = np.zeros((1, H, W), np.float32)
    for pix in char_pix:
        rgb_mask[0][pix] = 1.0
    return (np.stack(frames_rgb), np.stack(frames_a), rgb_mask)


def frame_png(rgb, alpha, path, upscale=6):
    """Render a target frame: letters in color, support as gray presence."""
    a = alpha[0]
    img = np.ones((CANVAS, CANVAS, 3), np.float32)
    for c in range(3):
        img[..., c] = 1 - a + rgb[c]
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)) \
        .resize((CANVAS * upscale, CANVAS * upscale), Image.NEAREST).save(path)


def make_seed_state(channel_n, device):
    x = torch.zeros(channel_n, CANVAS, CANVAS, device=device)
    x[3:, CANVAS // 2, CANVAS // 2] = 1.0
    return x


def rotate_stack(t, deg):
    """Rotate a [N,C,H,W] tensor by deg degrees (bilinear, same canvas)."""
    rad = np.deg2rad(deg)
    c, s = float(np.cos(rad)), float(np.sin(rad))
    theta = torch.tensor([[c, -s, 0.0], [s, c, 0.0]], dtype=t.dtype,
                         device=t.device).expand(t.shape[0], 2, 3)
    grid = F.affine_grid(theta, t.shape, align_corners=False)
    return F.grid_sample(t, grid, align_corners=False)


def train(text, steps=8000, K=60, glyph=12, channel_n=16, hidden_n=80,
          batch=8, pool_size=64, lr=2e-3, ca_min=8, ca_max=16,
          log_every=100, snap_dir=None, rng_seed=0, growth="bfs",
          rot_mode="aug90", rot_at=1000, rot_deg=20.0, lifespan=None,
          letter_w=8.0, adaptive=False):
    torch.manual_seed(sum(map(ord, text)) + 31)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")

    rgb_np, a_np, mask_np = grow_frames(text, K, glyph, rng_seed, growth=growth,
                                        lifespan=lifespan)
    rgb_f = torch.from_numpy(rgb_np).to(device)      # [K,3,H,W]
    a_f = torch.from_numpy(a_np).to(device)          # [K,1,H,W]
    rgb_mask = torch.from_numpy(mask_np).to(device)  # [1,H,W]

    # rot_mode 'late': after rot_at steps the whole target world rotates by
    # rot_deg degrees — tests whether learned growth adapts to the shift.
    if rot_mode == "late":
        rgb_rot = rotate_stack(rgb_f, rot_deg)
        a_rot = rotate_stack(a_f, rot_deg)
        mask_rot = rotate_stack(rgb_mask[None], rot_deg)[0]

    if snap_dir:
        Path(snap_dir).mkdir(parents=True, exist_ok=True)
        frame_png(rgb_np[-1], a_np[-1], Path(snap_dir) / "target.png")

    model = NCA(channel_n, hidden_n=hidden_n).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=[int(steps * 0.8)], gamma=0.1)

    # Persistent trajectories: state, current frame index, fixed rotation.
    pool = torch.stack([make_seed_state(channel_n, device)] * pool_size)
    pool_k = torch.zeros(pool_size, dtype=torch.long)
    pool_rot = torch.arange(pool_size) % 4 if rot_mode == "aug90" \
        else torch.zeros(pool_size, dtype=torch.long)

    start_step, _ = try_resume(snap_dir, model, opt, sched, device=device)

    meta = RunMeta(snap_dir, text, "nca.train_organic_reveal",
                   {"steps": steps, "K": K, "glyph": glyph, "batch": batch,
                    "lr": lr, "rng_seed": rng_seed, "pool_size": pool_size,
                    "growth": growth, "rot_mode": rot_mode,
                    "rot_at": rot_at, "rot_deg": rot_deg, "lifespan": lifespan,
                    "letter_w": letter_w, "adaptive": adaptive},
                   channel_n, hidden_n, "single", steps, device,
                   tags=["organic"] + (["adaptive"] if adaptive else []))

    t0 = time.time()
    for step in range(start_step, steps):
        idx = torch.randperm(pool_size)[:batch]
        x = pool[idx].clone()
        ks = pool_k[idx]
        rots = pool_rot[idx]

        if rot_mode == "late" and step >= rot_at:
            src_rgb, src_a, src_mask = rgb_rot, a_rot, mask_rot
        else:
            src_rgb, src_a, src_mask = rgb_f, a_f, rgb_mask
        tgt_rgb = torch.stack([torch.rot90(src_rgb[min(k + 1, K - 1)], r.item(), (1, 2))
                               for k, r in zip(ks, rots)])
        tgt_a = torch.stack([torch.rot90(src_a[min(k + 1, K - 1)], r.item(), (1, 2))
                             for k, r in zip(ks, rots)])
        masks = torch.stack([torch.rot90(src_mask, r.item(), (1, 2)) for r in rots])

        if adaptive:
            tgt_full = torch.cat([tgt_rgb, tgt_a], dim=1)
            x, _used = adaptive_rollout(model, x, tgt_full, chunk=6, max_chunks=8)
        else:
            n_ca = int(torch.randint(ca_min, ca_max + 1, (1,)))
            x = model(x, steps=n_ca)

        # Letter pixels are ~4% of the canvas; without upweighting the model
        # learns blanket support and ignores the reveal entirely. The weight
        # ramps in over the first 2000 steps: full pressure from step 0
        # collapsed training into the absorbing all-dead state (lw8 run).
        lw_eff = 1.0 + (letter_w - 1.0) * min(1.0, step / 2000.0)
        pix_w = 1.0 + lw_eff * masks
        loss_a = ((x[:, 3:4] - tgt_a) ** 2 * pix_w).mean()
        # masks is [B,1,H,W]; broadcasting against [B,3,H,W] is correct as-is.
        # (A stray .unsqueeze(1) here used to create a [B,B,3,H,W] cross-batch
        # outer product — silently garbage gradients since v1.)
        loss_rgb = lw_eff * ((x[:, :3] - tgt_rgb) ** 2 * masks).sum() \
            / (masks.sum() * 3 * batch + 1e-8)
        loss = loss_a + loss_rgb

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
            for j, i in enumerate(idx):
                if ks[j] + 1 >= K - 1:   # trajectory done: restart it
                    pool[i] = make_seed_state(channel_n, device)
                    pool_k[i] = 0
                    if rot_mode == "aug90":
                        pool_rot[i] = torch.randint(0, 4, (1,))
                else:
                    pool_k[i] = ks[j] + 1

        if step % log_every == 0 or step == steps - 1:
            print(f"[organic_reveal_{text}] step {step} loss {loss.item():.5f} "
                  f"(a {loss_a.item():.5f} rgb {loss_rgb.item():.5f}) "
                  f"k_max {int(pool_k.max())} ({time.time() - t0:.1f}s)", flush=True)
            if snap_dir:
                s = f"{step:05d}"
                j = int(torch.argmax(ks).item())
                img = to_rgb(x)[j].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
                Image.fromarray((img * 255).astype(np.uint8)) \
                    .resize((CANVAS * 6, CANVAS * 6), Image.NEAREST) \
                    .save(Path(snap_dir) / f"COMP_{s}.png")
                kj = min(int(ks[j]) + 1, K - 1)
                rj = int(rots[j])
                snap_rgb, snap_a = (src_rgb, src_a) if rot_mode == "late" \
                    else (rgb_f, a_f)
                frame_png(np.rot90(snap_rgb[kj].cpu().numpy(), rj, (1, 2)).copy(),
                          np.rot90(snap_a[kj].cpu().numpy(), rj, (1, 2)).copy(),
                          Path(snap_dir) / f"TARGET_{s}.png")
                torch.save(model.state_dict(), str(Path(snap_dir) / "latest.pth"))
                save_checkpoint(snap_dir, step, model, opt, sched)
                meta.log(step, loss.item(), loss_alpha=round(loss_a.item(), 6),
                         loss_rgb=round(loss_rgb.item(), 6))
                export_run_weights(model, snap_dir, text, glyph,
                                   grid_w=CANVAS, grid_h=CANVAS)

    print(f"Final loss for {text} (organic reveal): {loss.item():.5f}")
    return model


def preview_targets(text, out_dir, K=60, glyph=12, rng_seed=0, every=4,
                    growth="bfs", lifespan=None):
    """Write the proposed target frames for LGTM before training."""
    rgb, a, _ = grow_frames(text, K, glyph, rng_seed, growth=growth,
                            lifespan=lifespan)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for k in range(0, K, every):
        frame_png(rgb[k], a[k], out / f"TARGET_{k:05d}.png")
    frame_png(rgb[-1], a[-1], out / f"TARGET_{K-1:05d}.png")
    print(f"Wrote {len(range(0, K, every)) + 1} proposed frames to {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--text", required=True)
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--frames", type=int, default=60)
    p.add_argument("--rng-seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--snap-dir", default=None)
    p.add_argument("--growth", default="bfs", choices=["bfs", "dfs"])
    p.add_argument("--rot-mode", default="aug90", choices=["aug90", "none", "late"],
                   help="aug90: per-trajectory 90-deg rotations; none: no rotation; "
                        "late: rotate the target world by --rot-deg after --rot-at steps")
    p.add_argument("--rot-at", type=int, default=1000)
    p.add_argument("--rot-deg", type=float, default=20.0)
    p.add_argument("--lifespan", type=int, default=None,
                   help="Support cells die after this many frames unless on "
                        "the letter-connecting backbone")
    p.add_argument("--letter-w", type=float, default=8.0,
                   help="Loss upweight on letter pixels")
    p.add_argument("--adaptive", action="store_true")
    p.add_argument("--preview", action="store_true",
                   help="Only generate proposed TARGET frames, no training")
    a = p.parse_args()

    if a.preview:
        preview_targets(a.text, a.snap_dir or "snaps_organic_reveal_preview",
                        K=a.frames, rng_seed=a.rng_seed, growth=a.growth,
                        lifespan=a.lifespan)
    else:
        train(a.text, steps=a.steps, K=a.frames, rng_seed=a.rng_seed,
              log_every=a.log_every, snap_dir=a.snap_dir, growth=a.growth,
              rot_mode=a.rot_mode, rot_at=a.rot_at, rot_deg=a.rot_deg,
              lifespan=a.lifespan, letter_w=a.letter_w, adaptive=a.adaptive)
