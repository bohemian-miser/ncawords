"""Simplified Continuous Cellular Automaton (Lenia-NCA Bridge).

Brings the continuous automaton formulation directly in line with the rest
of the codebase (Distill Neural Cellular Automata):
1. Multi-channel state with continuous Euler updates:
     x_{t+1} = clamp(x_t + dt * dx, 0, 1)
2. Spatial gradient perception (Sobel-x, Sobel-y, Identity) via depthwise conv.
3. Clean 2-layer 1x1 conv update rule (fc0 -> ReLU -> fc1).
4. Stochastic update masking (fire_rate=0.5) preventing catastrophic oscillation.
5. Persistent Distill sample pool (512 boards, loss-ranking seed replacement, damage injection).
6. Direct target MSE loss with spatial damage recovery.
7. Exports standard browser-compatible weights for live client-side execution in docs/lenia.html.
"""

import argparse
import io
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from nca.checkpoint import save_checkpoint, try_resume
from nca.kernel_nca import KernelNCA, export_kernel_nca_weights
from nca.model import to_rgba
from nca.runmeta import RunMeta
from nca.train_emoji_vanilla import emoji_rgba
from nca.train_staged import render_word_3_line_fan


def _lines(H, W, dirs, spacing, width):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    out = np.zeros((H, W), np.float32)
    for th in dirs:
        p = xx * np.cos(th) + yy * np.sin(th)
        d = np.abs(((p / spacing) % 1.0) - 0.5) * spacing
        out = np.maximum(out, (d < width).astype(np.float32))
    return out


def make_target(kind, H=64, W=64):
    if kind == 'dots':
        s = 14
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        best = np.full((H, W), 1e9, np.float32)
        for i in range(-1, H // int(s * 0.87) + 2):
            for j in range(-2, W // int(s) + 2):
                cy, cx = i * s * 0.866, j * s + (i % 2) * s / 2
                best = np.minimum(best, (yy - cy) ** 2 + (xx - cx) ** 2)
        return (best < 3.2 ** 2).astype(np.float32)
    if kind == 'hex':
        return _lines(H, W, [0, np.pi / 3, 2 * np.pi / 3], 11.0, 1.1)
    if kind == 'tri':
        return _lines(H, W, [np.pi / 6, np.pi / 2, 5 * np.pi / 6], 11.0, 1.1)
    if kind == 'square':
        return _lines(H, W, [0, np.pi / 2], 12.0, 1.2)
    raise ValueError(f'Unknown target: {kind}')


def get_target_rgba(target_spec, H=64, W=64):
    """Returns RGBA target numpy array [4, H, W] in [0, 1]."""
    if target_spec.startswith('emoji:'):
        code = target_spec[6:]
        return emoji_rgba(code, H, W, size=min(H, W) - 16)
    if target_spec.startswith('word:'):
        text = target_spec[5:]
        arr = render_word_3_line_fan(text, 12)
        # arr is [4, h, w]
        _, h, w = arr.shape
        out = np.zeros((4, H, W), np.float32)
        y0, x0 = max(0, (H - h) // 2), max(0, (W - w) // 2)
        ys, xs = min(h, H), min(w, W)
        out[:, y0:y0 + ys, x0:x0 + xs] = arr[:, :ys, :xs]
        return out
    
    # Texture target (single channel -> RGBA black pattern on transparent background)
    pattern = make_target(target_spec, H, W)
    rgba = np.zeros((4, H, W), np.float32)
    rgba[3] = pattern  # Alpha channel
    return rgba


def damage_mask_circle(B, H, W, device, min_r=4, max_r=12):
    mask = torch.ones(B, 1, H, W, device=device)
    yy, xx = torch.meshgrid(torch.arange(H, device=device),
                            torch.arange(W, device=device), indexing='ij')
    for b in range(B):
        cy = torch.randint(8, H - 8, (1,)).item()
        cx = torch.randint(8, W - 8, (1,)).item()
        rad = torch.randint(min_r, max_r + 1, (1,)).item()
        d2 = (yy - cy) ** 2 + (xx - cx) ** 2
        mask[b, 0, d2 < rad * rad] = 0.0
    return mask


def train(target='dots', channel_n=16, hidden_n=128, steps=6000, batch=8,
          pool_size=256, size=64, lr=2e-3, t_min=32, t_max=64,
          damage_p=0.35, rng_seed=0, log_every=100, ckpt_every=500,
          num_kernels=2, kernel_size=3, init_kernels='sobel',
          device=None, snap_dir=None):
    torch.manual_seed(100 + rng_seed)
    target_util = float(os.environ.get("NCA_GPU_UTIL", "0.30"))

    if device is None or device == 'auto':
        if torch.cuda.is_available():
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            free_gb = free_bytes / (1024 ** 3)
            # Require at least 6.0 GB free on GPU to touch CUDA, protecting main host workload
            if free_gb >= 6.0:
                device = 'cuda'
                try:
                    torch.cuda.set_per_process_memory_fraction(0.04)  # max ~1.9 GB out of 48 GB
                except Exception:
                    pass
            else:
                print(f'[RESOURCE GUARD] GPU free memory is tight ({free_gb:.2f} GiB free < 6.0 GiB).', flush=True)
                print(f'[RESOURCE GUARD] Running on 128-core CPU (16 threads) with 0 GPU VRAM to protect main workload.', flush=True)
                device = 'cpu'
                torch.set_num_threads(16)
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
            torch.set_num_threads(16)
    
    model = KernelNCA(
        channel_n=channel_n, fire_rate=0.5, hidden_n=hidden_n,
        num_kernels=num_kernels, kernel_size=kernel_size, init_kernels=init_kernels
    ).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f'Trainable-Kernel Lenia NCA on {device} (target GPU util: {int(target_util*100)}%, C={channel_n}, hidden={hidden_n}, {num_kernels}x{kernel_size}x{kernel_size} kernels), {n_par} parameters')

    tgt_np = get_target_rgba(target, size, size)
    tgt = torch.from_numpy(tgt_np).to(device)  # [4, H, W]
    tgt_b = tgt.unsqueeze(0).repeat(batch, 1, 1, 1)

    seed = torch.zeros(1, channel_n, size, size, device=device)
    # Seed live cell at center
    cy, cx = size // 2, size // 2
    seed[:, 3:, cy - 1:cy + 2, cx - 1:cx + 2] = 1.0

    print(f'Initializing persistent sample pool with {pool_size} boards...')
    pool = seed.repeat(pool_size, 1, 1, 1)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[int(steps * 0.85)], gamma=0.1)

    if snap_dir:
        Path(snap_dir).mkdir(parents=True, exist_ok=True)
        # Save reference target image
        tgt_vis = (1 - tgt_np[3:4] + tgt_np[:3]).clip(0, 1).transpose(1, 2, 0)
        Image.fromarray((tgt_vis * 255).astype(np.uint8)) \
            .resize((size * 5, size * 5), Image.NEAREST).save(Path(snap_dir) / 'target.png')

    meta = RunMeta(
        snap_dir, f'CA-{target}', 'nca.train_lenia_pool',
        {'target': target, 'channel_n': channel_n, 'hidden_n': hidden_n,
         'steps': steps, 'batch': batch, 'pool_size': pool_size,
         'lr': lr, 'size': size, 'params': n_par},
        channel_n, hidden_n, 'pool', steps, device, tags=['nca', 'pool', target]
    )

    start_step, _ = try_resume(snap_dir, model, opt, sched, device=device)
    if start_step >= steps:
        print(f'[{target}] already reached {start_step} >= {steps} steps, skipping.', flush=True)
        return model

    t0 = time.time()
    for step in range(start_step, steps):
        t_step = time.time()
        idx = torch.randperm(pool_size, device=device)[:batch]
        x = pool[idx]

        # Rank by loss against target RGBA
        with torch.no_grad():
            sample_losses = F.mse_loss(to_rgba(x), tgt_b, reduction='none').mean(dim=(1, 2, 3))
            rank = sample_losses.argsort(descending=True)
        x = x[rank]
        idx = idx[rank]

        # Replace worst sample with fresh seed
        x[:1] = seed

        # Occasional damage on lowest-loss samples
        if damage_p > 0 and step > 100 and torch.rand(1).item() < damage_p:
            dmg = damage_mask_circle(1, size, size, device)
            x[-1:] = x[-1:] * dmg

        x_start = x[-1:].detach().clone()

        T = int(torch.randint(t_min, t_max + 1, (1,)))
        x = model(x, steps=T)

        loss = F.mse_loss(to_rgba(x), tgt_b)

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

        # Duty-cycle throttle to target GPU utilization (e.g. 70%)
        if target_util < 1.0 and target_util > 0.0:
            active_s = time.time() - t_step
            idle_s = active_s * (1.0 - target_util) / target_util
            if idle_s > 0.001:
                time.sleep(idle_s)

        if step % log_every == 0 or step == steps - 1:
            print(f'[{target}] step {step:5d} | loss {loss.item():.5f} | T={T} ({time.time() - t0:.1f}s)', flush=True)
            if snap_dir:
                s = f'{step:05d}'
                for tag, t in [('COMP', to_rgba(x)[-1]), ('START', to_rgba(x_start)[0])]:
                    img_t = t.detach().cpu().clamp(0, 1)
                    vis = (1 - img_t[3:4] + img_t[:3]).clamp(0, 1).permute(1, 2, 0).numpy()
                    Image.fromarray((vis * 255).astype(np.uint8)) \
                        .resize((size * 5, size * 5), Image.NEAREST) \
                        .save(Path(snap_dir) / f'{tag}_{s}.png')
                raw_m = getattr(model, '_orig_mod', model)
                raw_m.save_kernel_image(Path(snap_dir) / f'KERNEL_{s}.png')
                torch.save(raw_m.state_dict(), str(Path(snap_dir) / 'latest.pth'))
                meta.log(step, loss.item(), ca_steps=T)
                export_kernel_nca_weights(raw_m, snap_dir, f'Lenia-{target}', grid_w=size, grid_h=size)

        if snap_dir and (step % ckpt_every == 0 or step == steps - 1):
            raw_m = getattr(model, '_orig_mod', model)
            save_checkpoint(snap_dir, step, raw_m, opt, sched)

    print(f'Final loss for {target}: {loss.item():.5f}')
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Lenia-like Continuous Automaton with Trainable Kernels')
    parser.add_argument('--target', default='dots', help='Target: dots, hex, emoji:1f525, word:COMP')
    parser.add_argument('--steps', type=int, default=6000)
    parser.add_argument('--channel-n', type=int, default=16)
    parser.add_argument('--hidden-n', type=int, default=128)
    parser.add_argument('--num-kernels', type=int, default=2)
    parser.add_argument('--kernel-size', type=int, default=3)
    parser.add_argument('--init-kernels', default='sobel', choices=['sobel', 'ring', 'random'])
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--pool-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--log-every', type=int, default=100)
    parser.add_argument('--snap-dir', default=None)
    args = parser.parse_args()

    train(
        target=args.target,
        channel_n=args.channel_n,
        hidden_n=args.hidden_n,
        steps=args.steps,
        batch=args.batch,
        pool_size=args.pool_size,
        lr=args.lr,
        log_every=args.log_every,
        num_kernels=args.num_kernels,
        kernel_size=args.kernel_size,
        init_kernels=args.init_kernels,
        snap_dir=args.snap_dir
    )
