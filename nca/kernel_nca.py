"""Trainable-Kernel Neural Cellular Automata (Lenia-NCA Bridge).

Combines the continuous dynamics, stochastic gating, and sample pool of
Distill Neural Cellular Automata with TRAINABLE SPATIAL KERNELS (Lenia style).

Instead of hardcoded Sobel filters, the spatial convolution kernels K_1, K_2, ...
are learned directly by gradient descent alongside the 1x1 MLP weights.
"""

import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


class KernelNCA(nn.Module):
    def __init__(self, channel_n=16, fire_rate=0.5, hidden_n=128,
                 num_kernels=2, kernel_size=3, init_kernels="sobel"):
        super().__init__()
        self.channel_n = channel_n
        self.fire_rate = fire_rate
        self.num_kernels = num_kernels
        self.kernel_size = kernel_size
        ks = kernel_size
        assert ks % 2 == 1, "kernel_size must be odd"
        pad = ks // 2

        # 1. Identity kernel at center (fixed buffer so each cell perceives its own state)
        ident = torch.zeros(1, ks, ks)
        ident[0, pad, pad] = 1.0
        self.register_buffer("ident_kernel", ident)

        # 2. Trainable spatial kernels (e.g. num_kernels of size ks x ks)
        trainable = torch.zeros(num_kernels, ks, ks)
        coords = torch.arange(ks).float() - pad
        if init_kernels == "sobel":
            # Generalized Sobel for any odd kernel_size ks
            sigma = max(1.0, pad * 0.6)
            smooth = torch.exp(-0.5 * (coords / sigma) ** 2)
            smooth = smooth / smooth.sum()
            deriv = coords.clone()
            deriv = deriv / (deriv.abs().sum() + 1e-6)
            sobel_x = smooth[:, None] * deriv[None, :]
            sobel_y = sobel_x.T
            trainable[0] = sobel_x
            if num_kernels > 1:
                trainable[1] = sobel_y
            if num_kernels > 2:
                # Radial Laplacian / Mexican hat
                r2 = (coords[:, None] ** 2 + coords[None, :] ** 2) / (sigma ** 2)
                lap = (1.0 - 0.5 * r2) * torch.exp(-0.5 * r2)
                lap[pad, pad] = 0.0
                lap = lap - lap.mean()
                trainable[2] = lap / (lap.abs().sum() + 1e-6)
            for k in range(3, num_kernels):
                trainable[k] = torch.randn(ks, ks) * 0.05
        elif init_kernels == "ring":
            # Concentric Gaussian rings for any kernel size (Lenia style)
            y, x = torch.meshgrid(coords, coords, indexing="ij")
            r = (x ** 2 + y ** 2).sqrt() / max(1, pad)
            for k in range(num_kernels):
                mu = 0.25 + 0.6 * (k / max(1, num_kernels - 1))
                sigma = 0.25 / max(1, num_kernels)
                ring = torch.exp(-0.5 * ((r - mu) / sigma) ** 2)
                ring[pad, pad] = 0.0  # zero center
                trainable[k] = ring / (ring.sum() + 1e-6)
        else:
            # Random initialization
            trainable = torch.randn(num_kernels, ks, ks) * 0.05

        self.trainable_kernels = nn.Parameter(trainable)

        # Total perception features per channel = 1 (ident) + num_kernels
        in_channels = (1 + num_kernels) * channel_n
        self.fc0 = nn.Conv2d(in_channels, hidden_n, 1)
        self.fc1 = nn.Conv2d(hidden_n, channel_n, 1, bias=False)
        nn.init.normal_(self.fc1.weight, std=0.001)  # small initial delta update

    def get_all_kernels(self):
        """Returns [1 + num_kernels, ks, ks] containing ident + trainable kernels."""
        return torch.cat([self.ident_kernel, self.trainable_kernels], dim=0)

    def perceive(self, x):
        all_k = self.get_all_kernels()  # [1 + K, ks, ks]
        # Repeat for grouped depthwise conv across all channels
        # Shape: [(1+K)*C, 1, ks, ks]
        k_grouped = all_k.repeat(self.channel_n, 1, 1)[:, None, :, :]
        pad = self.kernel_size // 2
        return F.conv2d(x, k_grouped, padding=pad, groups=self.channel_n)

    def alive_mask(self, x):
        return F.max_pool2d(x[:, 3:4], 3, stride=1, padding=1) > 0.1

    def forward(self, x, fire_rate=None, steps=1):
        for _ in range(steps):
            x = self.step(x, fire_rate)
        return x

    def step(self, x, fire_rate=None):
        pre_life = self.alive_mask(x)
        dx = self.fc1(F.relu(self.fc0(self.perceive(x))))
        if fire_rate is None:
            fire_rate = self.fire_rate
        update_mask = (torch.rand(x.shape[0], 1, x.shape[2], x.shape[3],
                                  device=x.device) <= fire_rate).float()
        x = x + dx * update_mask
        post_life = self.alive_mask(x)
        life = (pre_life & post_life).float()
        return x * life

    def save_kernel_image(self, out_path):
        """Renders the learned kernels into a normalized tile strip for the KERNEL stream."""
        kimg = self.trainable_kernels.detach().cpu().numpy()  # [num_kernels, ks, ks]
        kmin = kimg.min(axis=(1, 2), keepdims=True)
        kmax = kimg.max(axis=(1, 2), keepdims=True)
        denom = np.where(kmax - kmin > 1e-6, kmax - kmin, 1.0)
        kn = (kimg - kmin) / denom

        # Add a 1-pixel border between kernels for visual distinction
        tiles = []
        for i in range(kn.shape[0]):
            tiles.append(kn[i])
            if i < kn.shape[0] - 1:
                tiles.append(np.full((kn.shape[1], 1), 0.5, dtype=np.float32))
        row = np.concatenate(tiles, axis=1)
        scale = max(16, 64 // self.kernel_size)
        img = Image.fromarray((row * 255).astype(np.uint8)) \
            .resize((row.shape[1] * scale, row.shape[0] * scale), Image.NEAREST)
        img.save(out_path)

def export_kernel_nca_weights(model, snap_dir, label, grid_w=64, grid_h=64):
    """Exports weights.json for live browser execution in docs/lenia.html."""
    out = Path(snap_dir) / "weights.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    c = model.channel_n
    num_k = 1 + model.num_kernels
    w0 = model.fc0.weight.detach().cpu().squeeze(-1).squeeze(-1).numpy()
    w0 = w0.reshape(-1, c, num_k).transpose(0, 2, 1).reshape(-1, num_k * c)
    w1 = model.fc1.weight.detach().cpu().squeeze(-1).squeeze(-1).numpy()
    b0 = model.fc0.bias.detach().cpu().numpy()
    kernels = model.trainable_kernels.detach().cpu().numpy()

    d = {
        "kind": "kernel_nca",
        "char": label,
        "label": label,
        "grid": grid_w,
        "grid_w": grid_w,
        "grid_h": grid_h,
        "channel_n": c,
        "hidden_n": model.fc0.out_channels,
        "fire_rate": model.fire_rate,
        "layout": "blocked",
        "kernel_size": model.kernel_size,
        "num_kernels": model.num_kernels,
        "kernels": kernels.round(5).tolist(),
        "seeds": [{"x": grid_w // 2, "y": grid_h // 2, "code": [], "char": label}],
        "fc0_w": w0.round(5).tolist(),
        "fc0_b": b0.round(5).tolist(),
        "fc1_w": w1.round(5).tolist(),
    }
    out.write_text(json.dumps(d))
    # Also copy to docs/weights if docs/weights exists
    docs_w = Path("docs/weights") / f"{Path(snap_dir).name}.json"
    if Path("docs/weights").exists():
        docs_w.write_text(json.dumps(d))
