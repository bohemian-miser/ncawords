"""Trainer for Dynamic / Layer-Modulated Kernel NCA.

Trains a continuous automaton where explicit weights and biases between the
state and hidden layers modulate the spatial convolution kernels on every step.
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
from nca.dynamic_kernel_nca import DynamicKernelNCA, export_dynamic_kernel_nca_weights
from nca.model import to_rgba
from nca.runmeta import RunMeta
from nca.train_lenia_pool import get_target_rgba, damage_mask_circle


def save_pool_mosaic(pool, tgt, size, out_path, grid_n=4):
    """Saves a grid_n x grid_n mosaic of RGBA boards from the pool."""
    with torch.no_grad():
        tgt_b = tgt.unsqueeze(0).repeat(pool.shape[0], 1, 1, 1)
        losses = F.mse_loss(to_rgba(pool), tgt_b, reduction='none').mean(dim=(1, 2, 3))
        ranked = losses.argsort()

        N = grid_n * grid_n
        idx_best = ranked[:grid_n]
        idx_med = ranked[len(ranked) // 3: len(ranked) // 3 + grid_n]
        idx_high = ranked[2 * len(ranked) // 3: 2 * len(ranked) // 3 + grid_n]
        idx_worst = ranked[-grid_n:]
        chosen_idx = torch.cat([idx_best, idx_med, idx_high, idx_worst])[:N]

        samples = pool[chosen_idx]
        rgba = to_rgba(samples).clamp(0, 1).cpu()
        vis = (1.0 - rgba[:, 3:4] + rgba[:, :3]).clamp(0, 1)

        mosaic = torch.zeros(grid_n * size, grid_n * size, 3)
        for i in range(N):
            r = i // grid_n
            c = i % grid_n
            mosaic[r * size:(r + 1) * size, c * size:(c + 1) * size] = vis[i].permute(1, 2, 0)

        mosaic_np = (mosaic.numpy() * 255).astype(np.uint8)
        img = Image.fromarray(mosaic_np).resize((grid_n * size * 2, grid_n * size * 2), Image.NEAREST)
        img.save(out_path)


def train(target='dots', channel_n=16, hidden_n=128, steps=10000, batch=8,
          pool_size=256, size=64, lr=2e-3, t_min=32, t_max=64,
          damage_p=0.35, rng_seed=0, log_every=100, ckpt_every=500,
          num_kernels=2, num_basis=4, kernel_size=5, init_kernels='sobel',
          device=None, snap_dir=None, pool_init='curriculum', force_reset=False,
          target_util=None):
    torch.manual_seed(200 + rng_seed)
    if target_util is None:
        target_util = float(os.environ.get("NCA_GPU_UTIL", "1.0"))

    if device is None or device == 'auto':
        if torch.cuda.is_available():
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            free_gb = free_bytes / (1024 ** 3)
            if free_gb >= 4.0:
                device = 'cuda'
                try:
                    if "NCA_GPU_MEM_FRAC" in os.environ:
                        torch.cuda.set_per_process_memory_fraction(float(os.environ["NCA_GPU_MEM_FRAC"]))
                    else:
                        torch.cuda.set_per_process_memory_fraction(0.45)
                except Exception:
                    pass
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cuda.matmul.allow_tf32 = True
            else:
                print(f'[RESOURCE GUARD] GPU free memory is tight ({free_gb:.2f} GiB free < 4.0 GiB).', flush=True)
                print(f'[RESOURCE GUARD] Running on 128-core CPU (16 threads) with 0 GPU VRAM to protect main workload.', flush=True)
                device = 'cpu'
                torch.set_num_threads(16)
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
            torch.set_num_threads(16)

    model = DynamicKernelNCA(
        channel_n=channel_n, fire_rate=0.5, hidden_n=hidden_n,
        num_kernels=num_kernels, num_basis=num_basis, kernel_size=kernel_size, init_kernels=init_kernels
    ).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f'Dynamic-Kernel NCA on {device} (pool_init: {pool_init}, target GPU util: {int(target_util*100)}%, C={channel_n}, hidden={hidden_n}, {num_kernels}x{num_basis}x{kernel_size}x{kernel_size} kernels), {n_par} parameters')

    tgt_np = get_target_rgba(target, size, size)
    tgt = torch.from_numpy(tgt_np).to(device)
    tgt_b = tgt.unsqueeze(0).repeat(batch, 1, 1, 1)

    seed = torch.zeros(1, channel_n, size, size, device=device)
    cy, cx = size // 2, size // 2
    if target in ('dots', 'hex', 'tri', 'square'):
        for sy in [size // 4, size // 2, 3 * size // 4]:
            for sx in [size // 4, size // 2, 3 * size // 4]:
                seed[:, 3:, sy - 1:sy + 2, sx - 1:sx + 2] = 1.0
    else:
        seed[:, 3:, cy - 1:cy + 2, cx - 1:cx + 2] = 1.0

    if pool_init == 'noise':
        print(f'Initializing 100% PURE NOISE sample pool with {pool_size} boards...')
        pool = torch.rand(pool_size, channel_n, size, size, device=device)
    else:
        print(f'Initializing noise-curriculum sample pool with {pool_size} boards...')
        pool = torch.zeros(pool_size, channel_n, size, size, device=device)
        n_seed = pool_size // 3
        n_noise = pool_size // 3
        pool[:n_seed] = seed.repeat(n_seed, 1, 1, 1)
        pool[n_seed:n_seed + n_noise] = torch.rand(n_noise, channel_n, size, size, device=device)
        bg_mask = (seed == 0.0).float()
        for i in range(n_seed + n_noise, pool_size):
            pool[i:i + 1] = seed + torch.rand(1, channel_n, size, size, device=device) * 0.5 * bg_mask

    mod_params = [
        model.basis_kernels,
        model.conv_surround.weight,
        model.conv_surround.bias,
        model.layer_x_to_k.weight,
        model.layer_x_to_k.bias,
        model.layer_h_to_k.weight,
        model.layer_h_to_k.bias,
    ]
    mlp_params = [model.fc0.weight, model.fc0.bias, model.fc1.weight]
    opt = torch.optim.Adam([
        {'params': mod_params, 'lr': lr * 2.5},
        {'params': mlp_params, 'lr': lr}
    ])
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[int(steps * 0.85)], gamma=0.1)

    if snap_dir:
        Path(snap_dir).mkdir(parents=True, exist_ok=True)
        tgt_vis = (1 - tgt[3:4] + tgt[:3]).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        Image.fromarray((tgt_vis * 255).astype(np.uint8)) \
            .resize((size * 5, size * 5), Image.NEAREST) \
            .save(Path(snap_dir) / 'target.png')

    meta = RunMeta(
        snap_dir, f'DynKernel-{target}', 'nca.train_dynamic_kernel',
        {'target': target, 'channel_n': channel_n, 'hidden_n': hidden_n,
         'steps': steps, 'batch': batch, 'pool_size': pool_size,
         'kernel_size': kernel_size, 'num_kernels': num_kernels,
         'init_kernels': init_kernels, 'lr': lr, 'size': size, 'params': n_par},
        channel_n, hidden_n, 'dynkernel', steps, device, tags=['nca', 'dynkernel', target]
    )

    if force_reset:
        start_step = 0
        print(f'[{target}] Force reset enabled: starting fresh from step 0 with cleared pool.', flush=True)
    else:
        start_step, _ = try_resume(snap_dir, model, opt, sched, device=device)
        if start_step >= steps:
            print(f'[{target}] already reached {start_step} >= {steps} steps, skipping.', flush=True)
            return model

    t0 = time.time()
    for step in range(start_step, steps):
        t_step = time.time()
        idx = torch.randperm(pool_size, device=device)[:batch]
        x = pool[idx]

        with torch.no_grad():
            sample_losses = F.mse_loss(to_rgba(x), tgt_b, reduction='none').mean(dim=(1, 2, 3))
            # Harsh penalty for dead boards (fewer than 15 alive cells) so they rank worst and get replaced
            alive_pix = (x[:, 3] > 0.1).float().sum(dim=(1, 2))
            sample_losses = sample_losses + (alive_pix < 15).float() * 100.0
            rank = sample_losses.argsort(descending=True)
        x = x[rank]
        idx = idx[rank]

        # Replacement on the worst boards:
        if pool_init == 'noise':
            # 100% pure random noise replacement: never inject center seeds
            x[0:1] = torch.rand(1, channel_n, size, size, device=device)
            if batch >= 16:
                x[1:2] = torch.rand(1, channel_n, size, size, device=device)
        else:
            # Diverse curriculum replacement on the worst boards:
            # Sample 0 gets a seed (50% clean seed, 50% noisy seed)
            if torch.rand(1).item() < 0.5:
                x[0:1] = seed
            else:
                x[0:1] = seed + torch.rand(1, channel_n, size, size, device=device) * 0.5 * bg_mask

            # Sample 1 gets pure random noise so the network constantly learns nucleation from scratch
            if batch > 1:
                x[1:2] = torch.rand(1, channel_n, size, size, device=device)

            # For larger batches (>= 16), also add another noisy seed to sample 2
            if batch >= 16:
                x[2:3] = seed + torch.rand(1, channel_n, size, size, device=device) * 0.5 * bg_mask

        # Mild noise injection on random surviving boards (curriculum robustness)
        n_corrupt = max(1, batch // 8)
        min_c = 3 if batch >= 16 else 2
        for _ in range(n_corrupt):
            if batch > min_c:
                c_idx = int(torch.randint(min_c, batch, (1,)))
                eta = torch.rand(1, device=device).item() * 0.4 + 0.1
                x[c_idx:c_idx + 1] = (1.0 - eta) * x[c_idx:c_idx + 1] + eta * torch.rand(1, channel_n, size, size, device=device)

        # Damage lowest-loss board(s)
        if damage_p > 0 and step > 100 and torch.rand(1).item() < damage_p:
            n_dmg = 2 if batch >= 16 else 1
            for d in range(1, n_dmg + 1):
                dmg = damage_mask_circle(1, size, size, device)
                x[-d:] = x[-d:] * dmg

        x_start = x[-1:].detach().clone()

        T = int(torch.randint(t_min, t_max + 1, (1,)))
        h = torch.zeros(batch, hidden_n, size, size, device=device)
        for _ in range(T):
            x, h = model.step(x, h)

        loss_mse = F.mse_loss(to_rgba(x), tgt_b)
        alpha = model.compute_local_alpha(x, h)

        # Measure spatial diversity exclusively over living cells of the organism:
        alive_mask = (x[:, 3:4] > 0.1).float()
        alive_count = alive_mask.sum(dim=(2, 3), keepdim=True).clamp(min=1.0)
        m_exp = alive_mask.unsqueeze(1)  # [B, 1, 1, H, W]
        mean_alive = (alpha * m_exp).sum(dim=(3, 4), keepdim=True) / alive_count.unsqueeze(1)
        var_alive = (((alpha - mean_alive) ** 2) * m_exp).sum(dim=(3, 4)) / alive_count.squeeze(-1)
        # Adding 1e-4 inside sqrt bounds gradients to <= 50, preventing exploding gradients / NaNs
        std_alive = torch.sqrt(var_alive + 1e-4).mean()

        # Smooth diversity bonus: Gini impurity sum_m alpha_m * (1 - alpha_m)
        # Bounded in [0, 0.75], completely smooth, zero at one-hot, maximum at uniform
        basis_spread = (alpha * (1.0 - alpha)).sum(dim=2).mean()

        # Extinction penalty: if alive cells in batch drop below 4%, penalize heavily
        alive_frac = alive_mask.mean()
        extinct_penalty = F.relu(0.04 - alive_frac) * 2.0

        loss = loss_mse - 0.08 * std_alive - 0.02 * basis_spread + extinct_penalty

        opt.zero_grad()
        if not (torch.isnan(loss) or torch.isinf(loss)):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            sched.step()
        else:
            print(f"[{target}] step {step}: NaN detected, skipping step", flush=True)

        with torch.no_grad():
            # Extinction safeguard: revive any board that collapsed to zero
            dead = (x[:, 3] > 0.1).float().sum(dim=(1, 2)) < 15
            if dead.any():
                for d_i in torch.where(dead)[0]:
                    if pool_init == 'noise':
                        x[d_i:d_i + 1] = torch.rand(1, channel_n, size, size, device=device)
                    else:
                        if torch.rand(1).item() < 0.5:
                            x[d_i:d_i + 1] = seed
                        else:
                            x[d_i:d_i + 1] = torch.rand(1, channel_n, size, size, device=device)
            pool[idx] = x.detach()

        # Duty-cycle throttle to target GPU utilization (e.g. 30%)
        if target_util < 1.0 and target_util > 0.0:
            if str(device).startswith('cuda'):
                torch.cuda.synchronize()
            active_s = time.time() - t_step
            idle_s = active_s * (1.0 - target_util) / target_util
            if idle_s > 0.001:
                time.sleep(idle_s)

        if step % log_every == 0 or step == steps - 1:
            print(f'[{target}] step {step:5d} | mse {loss_mse.item():.5f} | alive_std {std_alive.item():.3f} | spread {basis_spread.item():.3f} | T={T} ({time.time() - t0:.1f}s)', flush=True)
            if snap_dir:
                s = f'{step:05d}'
                for tag, t in [('COMP', to_rgba(x)[-1]), ('START', to_rgba(x_start)[0])]:
                    img_t = t.detach().cpu().clamp(0, 1)
                    vis = (1 - img_t[3:4] + img_t[:3]).clamp(0, 1).permute(1, 2, 0).numpy()
                    Image.fromarray((vis * 255).astype(np.uint8)) \
                        .resize((size * 5, size * 5), Image.NEAREST) \
                        .save(Path(snap_dir) / f'{tag}_{s}.png')
                raw_m = getattr(model, '_orig_mod', model)
                # Compute and save the dynamic kernel for the representative sample
                raw_m.save_kernel_image(Path(snap_dir) / f'KERNEL_{s}.png', x[-1:], h[-1:])
                # Save sample pool mosaic
                save_pool_mosaic(pool, tgt, size, Path(snap_dir) / f'POOL_{s}.png')
                torch.save(raw_m.state_dict(), str(Path(snap_dir) / 'latest.pth'))
                meta.log(step, loss.item(), ca_steps=T)
                export_dynamic_kernel_nca_weights(raw_m, snap_dir, f'DynKernel-{target}', grid_w=size, grid_h=size, seed_type='noise' if pool_init == 'noise' else 'seed')

        if snap_dir and (step % ckpt_every == 0 or step == steps - 1):
            raw_m = getattr(model, '_orig_mod', model)
            save_checkpoint(snap_dir, step, raw_m, opt, sched)

    print(f'Final loss for {target}: {loss.item():.5f}')
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Dynamic Layer-Modulated Kernel NCA')
    parser.add_argument('--target', default='dots')
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--channel-n', type=int, default=16)
    parser.add_argument('--hidden-n', type=int, default=128)
    parser.add_argument('--num-kernels', type=int, default=2)
    parser.add_argument('--num-basis', type=int, default=4)
    parser.add_argument('--kernel-size', type=int, default=5)
    parser.add_argument('--init-kernels', default='sobel', choices=['sobel', 'ring', 'random'])
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--pool-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--damage-p', type=float, default=0.35)
    parser.add_argument('--log-every', type=int, default=100)
    parser.add_argument('--ckpt-every', type=int, default=500)
    parser.add_argument('--snap-dir', default=None)
    parser.add_argument('--pool-init', default='curriculum', choices=['curriculum', 'noise'])
    parser.add_argument('--force-reset', action='store_true')
    parser.add_argument('--target-util', type=float, default=None)
    args = parser.parse_args()

    train(
        target=args.target,
        channel_n=args.channel_n,
        hidden_n=args.hidden_n,
        steps=args.steps,
        batch=args.batch,
        pool_size=args.pool_size,
        lr=args.lr,
        damage_p=args.damage_p,
        log_every=args.log_every,
        ckpt_every=args.ckpt_every,
        num_kernels=args.num_kernels,
        num_basis=args.num_basis,
        kernel_size=args.kernel_size,
        init_kernels=args.init_kernels,
        snap_dir=args.snap_dir,
        pool_init=args.pool_init,
        force_reset=args.force_reset,
        target_util=args.target_util
    )
