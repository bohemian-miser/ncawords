"""Neural Cellular Automata model — PyTorch port of the Growing NCA
architecture from distill.pub/2020/growing-ca (Mordvintsev et al., 2020).

Grid state: [B, C, H, W] with C channels. Channels 0-2 = RGB, 3 = alpha
(alive), 4+ = hidden. A cell is "alive" if any cell in its 3x3
neighborhood has alpha > 0.1.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NCA(nn.Module):
    def __init__(self, channel_n=16, fire_rate=0.5, hidden_n=128):
        super().__init__()
        self.channel_n = channel_n
        self.fire_rate = fire_rate

        # Perception: fixed identity + Sobel-x + Sobel-y depthwise filters.
        ident = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / 8.0
        sobel_y = sobel_x.T
        kernels = torch.stack([ident, sobel_x, sobel_y])  # [3,3,3]
        # One copy of each kernel per channel -> [3*C, 1, 3, 3] for grouped conv
        kernels = kernels.repeat(channel_n, 1, 1)[:, None, :, :]
        self.register_buffer("perception_kernels", kernels)

        self.fc0 = nn.Conv2d(channel_n * 3, hidden_n, 1)
        self.fc1 = nn.Conv2d(hidden_n, channel_n, 1, bias=False)
        nn.init.zeros_(self.fc1.weight)  # do-nothing initial behavior

    def perceive(self, x):
        # Depthwise conv: each channel convolved with identity/sobel_x/sobel_y
        y = F.conv2d(x, self.perception_kernels, padding=1, groups=self.channel_n)
        return y

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


def make_seed(size, channel_n=16, n=1):
    """Single live cell in the center, hidden channels + alpha set to 1."""
    x = torch.zeros(n, channel_n, size, size)
    x[:, 3:, size // 2, size // 2] = 1.0
    return x


def to_rgba(x):
    return x[:, :4]


def to_rgb(x):
    # Premultiplied-alpha RGBA over white background
    rgb, a = x[:, :3], x[:, 3:4].clamp(0, 1)
    return 1.0 - a + rgb
