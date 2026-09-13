"""Dynamic / Surrounding-Pixel-Modulated Kernel Neural Cellular Automata.

Per-pixel dynamic kernels:
Every cell (y, x) has its spatial perception kernel modulated by its
surrounding pixels (local 3x3 neighborhood) as well as global network layers
(state x, hidden h):
  alpha(y, x) = Conv2d(x, W_surround, 3x3)(y, x) + (W_x * x_bar + b_x) + (W_h * h_bar + b_h)
  K_eff(y, x) = sum_m alpha_m(y, x) * B_m

This allows the organism to use entirely different spatial filters at its
boundary (edge detection / boundary preservation), inside its body (diffusion /
reinforcement), and in empty surrounding space, while maintaining O(1) memory
footprint during training.
"""

import json
from pathlib import Path
import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


class DynamicKernelNCA(nn.Module):
    def __init__(self, channel_n=16, fire_rate=0.5, hidden_n=128,
                 num_kernels=2, kernel_size=5, num_basis=4, init_kernels="sobel"):
        super().__init__()
        self.channel_n = channel_n
        self.fire_rate = fire_rate
        self.hidden_n = hidden_n
        self.num_kernels = num_kernels
        self.num_basis = num_basis
        self.kernel_size = kernel_size
        ks = kernel_size
        assert ks % 2 == 1, "kernel_size must be odd"
        pad = ks // 2
        self.pad = pad

        # 1. Spatial basis filters: [num_kernels, num_basis, ks, ks]
        basis = torch.zeros(num_kernels, num_basis, ks, ks)
        coords = torch.arange(ks).float() - pad
        sigma = max(1.0, pad * 0.6)
        smooth = torch.exp(-0.5 * (coords / sigma) ** 2)
        smooth = smooth / (smooth.sum() + 1e-6)
        deriv = coords.clone()
        deriv = deriv / (deriv.abs().sum() + 1e-6)
        sobel_x = smooth[:, None] * deriv[None, :]
        sobel_y = sobel_x.T

        y_grid, x_grid = torch.meshgrid(coords, coords, indexing="ij")
        r_grid = (x_grid ** 2 + y_grid ** 2).sqrt() / max(1, pad)
        r2 = (coords[:, None] ** 2 + coords[None, :] ** 2) / (sigma ** 2)
        lap = (1.0 - 0.5 * r2) * torch.exp(-0.5 * r2)
        lap[pad, pad] = 0.0
        lap = lap - lap.mean()
        lap = lap / (lap.abs().sum() + 1e-6)

        ring = torch.exp(-0.5 * ((r_grid - 0.55) / 0.25) ** 2)
        ring[pad, pad] = 0.0
        ring = ring / (ring.sum() + 1e-6)

        primitives = [
            (sobel_x if init_kernels == "sobel" else ring),
            lap,
            ring,
            (sobel_y if init_kernels == "sobel" else lap)
        ]
        for k in range(num_kernels):
            for m in range(num_basis):
                if k == 0 and m == 0:
                    basis[k, m] = sobel_x if init_kernels == "sobel" else ring
                elif k == 1 and m == 0:
                    basis[k, m] = sobel_y if init_kernels == "sobel" else ring
                elif m == 1:
                    basis[k, m] = lap
                elif m == 2:
                    basis[k, m] = ring
                else:
                    basis[k, m] = primitives[(k + m) % len(primitives)]

        self.basis_kernels = nn.Parameter(basis)

        # 2. Local Surrounding Pixels Convolution: looks at surrounding 3x3 patch
        # around each pixel to determine local mixing weights alpha(y, x)
        self.conv_surround = nn.Conv2d(channel_n, num_kernels * num_basis, kernel_size=3, padding=1, bias=True)
        # Initialize surrounding weights with healthy variance so initial modulation is distinct
        nn.init.normal_(self.conv_surround.weight, std=0.05)
        with torch.no_grad():
            self.conv_surround.bias.zero_()
            for k in range(num_kernels):
                self.conv_surround.bias[k * num_basis] = 1.0

        # 3. Global network layer modulation: weights and biases connecting layers to kernels
        self.layer_x_to_k = nn.Linear(channel_n, num_kernels * num_basis, bias=True)
        self.layer_h_to_k = nn.Linear(hidden_n, num_kernels * num_basis, bias=True)
        nn.init.zeros_(self.layer_x_to_k.weight)
        nn.init.zeros_(self.layer_x_to_k.bias)
        nn.init.zeros_(self.layer_h_to_k.weight)
        nn.init.zeros_(self.layer_h_to_k.bias)

        # Scale factor 1/sqrt(C*9) prevents logit saturation into hard one-hot switches
        self.scale = 1.0 / math.sqrt(channel_n * 9)
        self.temperature = 1.0

        # 4. Dense MLP layers
        in_channels = (1 + num_kernels) * channel_n
        self.fc0 = nn.Conv2d(in_channels, hidden_n, 1)
        self.fc1 = nn.Conv2d(hidden_n, channel_n, 1, bias=False)
        nn.init.normal_(self.fc1.weight, std=0.001)

    def compute_local_alpha(self, x, h):
        """Computes per-pixel basis mixture weights alpha of shape [B, Nk, M, H, W] via softmax gating."""
        B, C, H, W = x.shape
        Nk = self.num_kernels
        M = self.num_basis

        # Local surrounding pixels modulation scaled to prevent saturation: [B, Nk * M, H, W]
        alpha_local = self.conv_surround(x) * self.scale

        # Global layer modulation (State x and Hidden h): [B, Nk * M, 1, 1]
        x_bar = x.mean(dim=(2, 3))
        h_bar = h.mean(dim=(2, 3))
        alpha_layer = (self.layer_x_to_k(x_bar) + self.layer_h_to_k(h_bar)).unsqueeze(-1).unsqueeze(-1) * self.scale

        logits = (alpha_local + alpha_layer).view(B, Nk, M, H, W)
        # Softmax over basis primitives so each cell actively selects its spatial perception mode
        alpha_tot = F.softmax(logits / self.temperature, dim=2)
        return alpha_tot

    def compute_effective_kernel_at(self, x, h, y, x_coord, b=0):
        """Returns the effective [Nk, ks, ks] kernel matrix at a specific pixel (y, x)."""
        with torch.no_grad():
            alpha = self.compute_local_alpha(x[b:b+1], h[b:b+1])  # [1, Nk, M, H, W]
            alpha_pix = alpha[0, :, :, y, x_coord]  # [Nk, M]
            basis = self.basis_kernels.detach()  # [Nk, M, ks, ks]
            k_eff = torch.einsum('km,kmhw->khw', alpha_pix, basis)
            return k_eff.cpu().numpy()

    def perceive(self, x, h):
        """Grouped depthwise perception where each pixel is convolved with its surrounding-pixel kernel."""
        B, C, H, W = x.shape
        Nk = self.num_kernels
        M = self.num_basis
        ks = self.kernel_size
        pad = self.pad

        # 1. Depthwise convolution with all basis filters
        # Expand basis kernels across C channels: [Nk * M * C, 1, ks, ks]
        k_depth = self.basis_kernels.unsqueeze(2).expand(Nk, M, C, ks, ks).contiguous().view(Nk * M * C, 1, ks, ks)
        x_exp = x.unsqueeze(1).expand(B, Nk * M, C, H, W).contiguous().view(B, Nk * M * C, H, W)
        r = F.conv2d(x_exp, k_depth, padding=pad, groups=Nk * M * C).view(B, Nk, M, C, H, W)

        # 2. Local surrounding pixel modulation
        alpha = self.compute_local_alpha(x, h).unsqueeze(3)  # [B, Nk, M, 1, H, W]

        # 3. Dynamic perception response per kernel
        p_dyn = (alpha * r).sum(dim=2).view(B, Nk * C, H, W)

        # 4. Concatenate center identity channel
        p = torch.cat([x, p_dyn], dim=1)  # [B, (1 + Nk) * C, H, W]
        return p

    def alive_mask(self, x):
        return F.max_pool2d(x[:, 3:4], 3, stride=1, padding=1) > 0.1

    def step(self, x, h=None, fire_rate=None):
        if h is None:
            h = torch.zeros(x.shape[0], self.hidden_n, x.shape[2], x.shape[3], device=x.device)

        pre_life = self.alive_mask(x)
        p = self.perceive(x, h)
        h_new = F.relu(self.fc0(p))
        dx = self.fc1(h_new)

        if fire_rate is None:
            fire_rate = self.fire_rate
        update_mask = (torch.rand(x.shape[0], 1, x.shape[2], x.shape[3],
                                  device=x.device) <= fire_rate).float()
        x_new = x + dx * update_mask

        post_life = self.alive_mask(x_new)
        life = (pre_life & post_life).float()
        x_new = x_new * life
        h_new = h_new * life
        return x_new, h_new

    def forward(self, x, fire_rate=None, steps=1):
        h = torch.zeros(x.shape[0], self.hidden_n, x.shape[2], x.shape[3], device=x.device)
        for _ in range(steps):
            x, h = self.step(x, h, fire_rate)
        return x

    def save_kernel_image(self, out_path, x=None, h=None):
        """Saves visual image of the effective learned kernels across representative coordinates."""
        device = next(self.parameters()).device
        if x is None:
            x = torch.zeros(1, self.channel_n, 64, 64, device=device)
        if h is None:
            h = torch.zeros(1, self.hidden_n, 64, 64, device=device)

        with torch.no_grad():
            alpha = self.compute_local_alpha(x[:1], h[:1])  # [1, Nk, M, H, W]
            alpha_mean = alpha.mean(dim=(3, 4))[0]  # [Nk, M]
            basis = self.basis_kernels.detach()
            k_eff = torch.einsum('km,kmhw->khw', alpha_mean, basis).cpu().numpy()

        Nk, ks, _ = k_eff.shape
        strips = []
        for k in range(Nk):
            ker = k_eff[k]
            vmax = max(abs(ker.min()), abs(ker.max()), 1e-5)
            norm = np.clip((ker / vmax + 1.0) * 0.5 * 255.0, 0, 255).astype(np.uint8)
            img = Image.fromarray(norm).resize((64, 64), Image.NEAREST)
            strips.append(np.array(img))
        combined = np.concatenate(strips, axis=1)
        Image.fromarray(combined).save(out_path)


def export_dynamic_kernel_nca_weights(model, out_dir, name="DynamicKernelNCA", grid_w=64, grid_h=64, seed_type="seed"):
    """Exports model weights including surrounding-pixel convolution and basis kernels."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fc0 = model.fc0
    fc1 = model.fc1
    w0 = fc0.weight.detach().cpu().squeeze(-1).squeeze(-1).numpy().T
    b0 = fc0.bias.detach().cpu().numpy()
    w1 = fc1.weight.detach().cpu().squeeze(-1).squeeze(-1).numpy().T

    basis_k = model.basis_kernels.detach().cpu().numpy()
    scale = getattr(model, 'scale', 1.0)
    w_surround = (model.conv_surround.weight * scale).detach().cpu().numpy()
    b_surround = (model.conv_surround.bias * scale).detach().cpu().numpy()
    wx = (model.layer_x_to_k.weight * scale).detach().cpu().numpy()
    bx = (model.layer_x_to_k.bias * scale).detach().cpu().numpy()
    wh = (model.layer_h_to_k.weight * scale).detach().cpu().numpy()
    bh = (model.layer_h_to_k.bias * scale).detach().cpu().numpy()

    # Precompute base/average kernels for backward compatibility with standard viewers
    alpha_logits = b_surround.reshape(model.num_kernels, model.num_basis)
    exp_logits = np.exp((alpha_logits - alpha_logits.max(axis=1, keepdims=True)) / model.temperature)
    alpha_base = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    base_k = np.einsum('km,kmhw->khw', alpha_base, basis_k)

    data = {
        "name": name,
        "kind": "dynamic_kernel_nca",
        "type": "dynamic_kernel_nca",
        "seedType": seed_type,
        "channel_n": model.channel_n,
        "hidden_n": model.hidden_n,
        "fire_rate": model.fire_rate,
        "kernel_size": model.kernel_size,
        "num_kernels": model.num_kernels,
        "num_basis": model.num_basis,
        "temperature": model.temperature,
        "layout": "blocked",
        "kernels": base_k.tolist(),
        "fc0_w": w0.T.tolist(),
        "fc0_b": b0.tolist(),
        "fc1_w": w1.T.tolist(),
        "w0": w0.tolist(),
        "b0": b0.tolist(),
        "w1": w1.tolist(),
        "basis_kernels": basis_k.tolist(),
        "w_surround": w_surround.tolist(),
        "b_surround": b_surround.tolist(),
        "w_x_to_k": wx.tolist(),
        "b_x_to_k": bx.tolist(),
        "w_h_to_k": wh.tolist(),
        "b_h_to_k": bh.tolist(),
        "grid_w": grid_w,
        "grid_h": grid_h
    }

    out_file = out_dir / "weights.json"
    with open(out_file, "w") as f:
        json.dump(data, f)
    return out_file
