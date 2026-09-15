"""Single-Model Multi-Target NCA with Seed-Cue Conditioning & Pool Rotation Augmentation.

Features:
  1. Single NCA network trained to produce 2 different emojis (e.g. Heart vs Rocket).
  2. Conditioned purely on a colored seed cue:
       - Small Red circle seed -> differentiates into Emoji 0 (Heart).
       - Small Blue circle seed -> differentiates into Emoji 1 (Rocket).
  3. Field training: target is downsampled into a larger field (e.g. 28x28 in 56x56)
     with background noise to enforce active noise rejection and open boundaries.
  4. Pool blob rotation augmentation: living organism blobs are periodically
     cut out, rotated (90°, 180°, 270°), and trained against rotated targets so the
     model learns orientation stability.
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from nca.model import NCA, to_rgba
from nca.train_emoji_vanilla import emoji_rgba
from nca.train_lenia_pool import damage_mask_circle


def make_circle_seed(channel_n, H, W, color_rgb, radius=3, cy=None, cx=None):
    """Creates a small colored circular seed cue in a 16-channel zero tensor."""
    seed = torch.zeros(channel_n, H, W, dtype=torch.float32)
    if cy is None:
        cy = H // 2
    if cx is None:
        cx = W // 2
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    dist_sq = (yy - cy) ** 2 + (xx - cx) ** 2
    mask = (dist_sq <= radius ** 2)

    seed[0, mask] = color_rgb[0]
    seed[1, mask] = color_rgb[1]
    seed[2, mask] = color_rgb[2]
    seed[3, mask] = 1.0  # alpha viability
    return seed


def cut_and_rotate_blob(x, rot_k, pad=2):
    """Crops the living organism bounding box, rotates it by rot_k * 90 deg, and pastes it back."""
    C, H, W = x.shape
    alive = (x[3] > 0.1)
    if alive.sum() < 10:
        return torch.rot90(x, k=rot_k, dims=(-2, -1))

    rows = torch.where(alive.any(dim=1))[0]
    cols = torch.where(alive.any(dim=0))[0]
    r_min, r_max = rows[0].item(), rows[-1].item()
    c_min, c_max = cols[0].item(), cols[-1].item()

    cr = (r_min + r_max) // 2
    cc = (c_min + c_max) // 2
    rad = max((r_max - r_min) // 2, (c_max - c_min) // 2) + pad

    r0 = max(0, cr - rad)
    r1 = min(H, cr + rad + 1)
    c0 = max(0, cc - rad)
    c1 = min(W, cc + rad + 1)

    side = min(r1 - r0, c1 - c0)
    r1 = r0 + side
    c1 = c0 + side

    x_out = x.clone()
    patch = x_out[:, r0:r1, c0:c1]
    patch_rot = torch.rot90(patch, k=rot_k, dims=(-2, -1))

    x_out[:, r0:r1, c0:c1] = patch_rot
    return x_out


def save_mosaic(pool, pool_labels, pool_rots, H, W, out_path, grid_n=4):
    """Saves a grid_n x grid_n mosaic of RGBA boards from the pool."""
    with torch.no_grad():
        N = grid_n * grid_n
        chosen_idx = torch.randperm(pool.shape[0])[:N]
        samples = pool[chosen_idx]
        rgba = to_rgba(samples).clamp(0, 1).cpu()
        vis = (1.0 - rgba[:, 3:4] + rgba[:, :3]).clamp(0, 1)

        mosaic = torch.zeros(grid_n * H, grid_n * W, 3)
        for i in range(N):
            r = i // grid_n
            c = i % grid_n
            mosaic[r * H:(r + 1) * H, c * W:(c + 1) * W] = vis[i].permute(1, 2, 0)

        mosaic_np = (mosaic.numpy() * 255).astype(np.uint8)
        img = Image.fromarray(mosaic_np).resize((grid_n * W * 3, grid_n * H * 3), Image.NEAREST)
        img.save(out_path)


def train(
    emoji_a="2764",        # Target 0: Heart
    emoji_b="1f680",       # Target 1: Rocket
    H=56, W=56,            # Field size
    target_size=28,        # Downsampled emoji size
    channel_n=16,
    hidden_n=128,
    steps=8000,
    batch=16,
    pool_size=256,
    lr=2e-3,
    ca_min=32,
    ca_max=64,
    damage_p=0.3,
    rotate_p=0.4,          # Probability of cutting & rotating a pool sample
    noise_bg_p=0.5,        # Probability of injecting noise into background field
    snap_dir="nca_runs/multitarget_rot_demo",
    log_every=100,
    device=None,
):
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    print(f"Device: {device} | Dual Targets: {emoji_a} (Red seed) vs {emoji_b} (Blue seed) | Field: {H}x{W}")
    Path(snap_dir).mkdir(parents=True, exist_ok=True)

    # 1. Load base emoji targets (normalized RGBA in [0, 1])
    tgt_a_np = emoji_rgba(emoji_a, H=H, W=W, size=target_size)
    tgt_b_np = emoji_rgba(emoji_b, H=H, W=W, size=target_size)
    tgts_base = torch.stack([
        torch.from_numpy(tgt_a_np).to(device),
        torch.from_numpy(tgt_b_np).to(device)
    ])  # [2, 4, H, W]

    # Precompute 4 discrete 90-degree rotations for each target: [2, 4, 4, H, W]
    tgts = torch.zeros(2, 4, 4, H, W, device=device)
    for l in range(2):
        for k in range(4):
            tgts[l, k] = torch.rot90(tgts_base[l], k=k, dims=(-2, -1))

    # Save target references
    for l, name in enumerate([emoji_a, emoji_b]):
        for k in range(4):
            vis = tgts[l, k].cpu()
            vis = (1.0 - vis[3:4] + vis[:3]).clamp(0, 1).permute(1, 2, 0).numpy()
            Image.fromarray((vis * 255).astype(np.uint8)).resize((H * 4, W * 4), Image.NEAREST).save(
                Path(snap_dir) / f"target_{name}_rot{k * 90}.png"
            )

    # 2. Seeds: Red circle for Target A (0), Blue circle for Target B (1)
    seed_a = make_circle_seed(channel_n, H, W, color_rgb=(1.0, 0.0, 0.0), radius=3).to(device)
    seed_b = make_circle_seed(channel_n, H, W, color_rgb=(0.0, 0.0, 1.0), radius=3).to(device)
    seeds = torch.stack([seed_a, seed_b])  # [2, C, H, W]

    # 3. Model & Optimizer
    model = NCA(channel_n, fire_rate=0.5, hidden_n=hidden_n).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[int(steps * 0.85)], gamma=0.1)

    # 4. Sample Pool: stores state, target label (0 or 1), and rotation angle k (0..3)
    pool = torch.zeros(pool_size, channel_n, H, W, device=device)
    pool_labels = torch.zeros(pool_size, dtype=torch.long, device=device)
    pool_rots = torch.zeros(pool_size, dtype=torch.long, device=device)

    half = pool_size // 2
    pool[:half] = seed_a.unsqueeze(0).repeat(half, 1, 1, 1)
    pool_labels[:half] = 0
    pool[half:] = seed_b.unsqueeze(0).repeat(pool_size - half, 1, 1, 1)
    pool_labels[half:] = 1

    t0 = time.time()
    for step in range(steps):
        idx = torch.randperm(pool_size, device=device)[:batch]
        x = pool[idx]
        b_labels = pool_labels[idx]
        b_rots = pool_rots[idx]

        b_tgts = tgts[b_labels, b_rots]

        with torch.no_grad():
            sample_losses = F.mse_loss(to_rgba(x), b_tgts, reduction='none').mean(dim=(1, 2, 3))
            alive = (x[:, 3] > 0.1).float().sum(dim=(1, 2))
            sample_losses = sample_losses + (alive < 10).float() * 100.0
            rank = sample_losses.argsort(descending=True)

        x = x[rank]
        b_labels = b_labels[rank]
        b_rots = b_rots[rank]
        idx = idx[rank]

        # Worst board (rank 0) gets replaced with a fresh seed
        new_label = int(torch.randint(0, 2, (1,)))
        new_rot = int(torch.randint(0, 4, (1,))) if rotate_p > 0 else 0
        fresh_seed = seeds[new_label].clone()

        if noise_bg_p > 0 and torch.rand(1).item() < noise_bg_p:
            bg_noise = torch.rand(channel_n, H, W, device=device) * 0.2
            bg_mask = (fresh_seed[3:4] == 0.0).float()
            fresh_seed = fresh_seed + bg_noise * bg_mask

        x[0] = fresh_seed
        b_labels[0] = new_label
        b_rots[0] = new_rot

        # Pool blob cutout & rotation augmentation on surviving boards
        if rotate_p > 0 and step > 50:
            for b in range(1, batch):
                if torch.rand(1).item() < rotate_p:
                    k_delta = int(torch.randint(1, 4, (1,)))
                    x[b] = cut_and_rotate_blob(x[b], k_delta)
                    b_rots[b] = (b_rots[b] + k_delta) % 4

        # Occasional damage injection
        if damage_p > 0 and step > 100 and torch.rand(1).item() < damage_p:
            dmg = damage_mask_circle(1, H, W, device)
            x[-1] = x[-1] * dmg[0]

        b_tgts = tgts[b_labels, b_rots]

        T = int(torch.randint(ca_min, ca_max + 1, (1,)))
        x_rollout = x
        for _ in range(T):
            x_rollout = model(x_rollout, steps=1)

        loss = F.mse_loss(to_rgba(x_rollout), b_tgts)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        sched.step()

        with torch.no_grad():
            pool[idx] = x_rollout.detach()
            pool_labels[idx] = b_labels
            pool_rots[idx] = b_rots

        if step % log_every == 0 or step == steps - 1:
            dt = time.time() - t0
            print(f"step {step:5d} | loss {loss.item():.5f} | T={T} ({dt:.1f}s)", flush=True)
            save_mosaic(pool, pool_labels, pool_rots, H, W, Path(snap_dir) / f"pool_{step:05d}.png")
            torch.save(model.state_dict(), Path(snap_dir) / "latest.pth")

    print(f"Training complete! Artifacts saved to {snap_dir}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--H", type=int, default=56)
    parser.add_argument("--W", type=int, default=56)
    parser.add_argument("--snap_dir", type=str, default="nca_runs/multitarget_rot_demo")
    args = parser.parse_args()

    train(steps=args.steps, batch=args.batch, H=args.H, W=args.W, snap_dir=args.snap_dir)
