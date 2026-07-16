import os
import numpy as np
from PIL import Image
from pathlib import Path
"""Train ONE NCA that grows a whole multi-character string from a SINGLE seed.
Method 4: Proximity Lattice / Procedural Webbing.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
torch.set_num_threads(1)
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from nca.model import NCA, to_rgba, to_rgb
from nca.train import FONT_PATH, char_color, SamplePool


PITCH = 14          # Lower pitch for closer letters
MARGIN = 6          # Lower margin
GRID_H = 20         # Lower height for lower res

def word_geometry(text):
    w = MARGIN * 2 + PITCH * len(text)
    return w, GRID_H

def render_word(text, glyph=12, font_path=FONT_PATH):
    """Whole string on a transparent canvas, per-char color, with a procedural proximity lattice."""
    w, h = word_geometry(text)
    font = ImageFont.truetype(font_path, glyph)
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # 1. Render text to find letter masks.
    text_img = Image.new("L", (w, h), 0)
    text_draw = ImageDraw.Draw(text_img)
    for i, ch in enumerate(text):
        x_center = MARGIN + PITCH * i + PITCH // 2
        l, t, r, b = text_draw.textbbox((0, 0), ch, font=font)
        text_draw.text((x_center - (r - l) / 2 - l, (h - (b - t)) / 2 - t), ch,
                      font=font, fill=255)
    
    mask = np.asarray(text_img) > 0 # True for letters

    # Simple contour detection
    contour_mask = mask & (
        (np.roll(mask, 1, axis=0) != mask) |
        (np.roll(mask, -1, axis=0) != mask) |
        (np.roll(mask, 1, axis=1) != mask) |
        (np.roll(mask, -1, axis=1) != mask)
    )

    # 2. Sample N anchor points on letter contours.
    letter_y, letter_x = np.where(contour_mask)
    n_anchors = min(len(letter_y), 50) # Sample at most 50 anchors
    rng = np.random.RandomState(sum(map(ord, text)))
    if len(letter_y) > 0:
        anchor_indices = rng.choice(len(letter_y), n_anchors, replace=False)
        anchors = list(zip(letter_x[anchor_indices], letter_y[anchor_indices]))
    else:
        anchors = []

    # 3. Sample M free points in empty space.
    empty_y, empty_x = np.where(~mask)
    n_free = min(len(empty_y), 30) # Sample at most 30 free points
    if len(empty_y) > 0:
        free_indices = rng.choice(len(empty_y), n_free, replace=False)
        free_points = list(zip(empty_x[free_indices], empty_y[free_indices]))
    else:
        free_points = []

    all_points = anchors + free_points
    
    # 4. Connect points based on distance (Proximity Graph).
    threshold = 8.0 # Max distance to connect
    if len(all_points) > 0:
        pts = np.array(all_points)
        # Pairwise differences
        diff = pts[:, None, :] - pts[None, :, :] # [N, N, 2]
        dists = np.linalg.norm(diff, axis=2) # [N, N]
        
        # Find pairs with dist < threshold and i < j
        i_indices, j_indices = np.where((dists < threshold) & (np.triu(np.ones_like(dists), 1) > 0))
        
        for i, j in zip(i_indices, j_indices):
            dist = dists[i, j]
            alpha = int(96 * (1 - dist / threshold))
            # Fix thickness: ensure it can be 1 or 2
            width = 2 if dist < threshold / 2 else 1
            p1 = all_points[i]
            p2 = all_points[j]
            draw.line((p1[0], p1[1], p2[0], p2[1]), fill=(128, 128, 128, alpha), width=width)

    # 5. Draw the text letters over the lattice
    for i, ch in enumerate(text):
        x_center = MARGIN + PITCH * i + PITCH // 2
        l, t, r, b = draw.textbbox((0, 0), ch, font=font)
        draw.text((x_center - (r - l) / 2 - l, (h - (b - t)) / 2 - t), ch,
                  font=font, fill=char_color(ch) + (255,))
        
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr[..., :3] *= arr[..., 3:]
    return arr.transpose(2, 0, 1)  # [4, H, W]

def make_single_seed(text, channel_n=16, n=1, tgt=None):
    w, h = word_geometry(text)
    x = torch.zeros(n, channel_n, h, w)
    
    cy, cx = h // 2, w // 2
    if tgt is not None:
        y_ids, x_ids = np.where(tgt[3] > 0.5)
        if len(y_ids) > 0:
            distances = (y_ids - cy)**2 + (x_ids - cx)**2
            best_idx = np.argmin(distances)
            cy, cx = y_ids[best_idx], x_ids[best_idx]
            
    import os
    if os.getenv("NOISE_START") == "1":
        x = torch.rand_like(x)
    else:
        x[:, 3:, cy, cx] = 1.0
    return x

def make_noise_seed(text, channel_n=16, n=1):
    w, h = word_geometry(text)
    x = torch.randn(n, channel_n, h, w) * 0.1
    x[:, 3:, h // 2, w // 2] = 1.0 
def damage_mask_rect(n, h, w, device):
    """1 outside a random circle, 0 inside (multiply to damage).
    Uniformly distributed across rectangular grid (h, w)."""
    short_axis = min(h, w)
    y = torch.linspace(-h/short_axis, h/short_axis, h, device=device)
    x = torch.linspace(-w/short_axis, w/short_axis, w, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    
    # Centers in middle half of corresponding dimension
    cy = (torch.rand(n, 1, 1, device=device) - 0.5) * (h/short_axis)
    cx = (torch.rand(n, 1, 1, device=device) - 0.5) * (w/short_axis)
    
    r = torch.rand(n, 1, 1, device=device) * 0.3 + 0.1
    
    mask = ((xx[None] - cx) ** 2 + (yy[None] - cy) ** 2 >= r ** 2).float()
    return mask[:, None]

def train(text, steps=4000, glyph=12, channel_n=32, hidden_n=128,
          batch=8, pool_size=256, lr=2e-3, damage_n=1, ca_min=64, ca_max=96,
          log_every=100, out=None, snap_dir=None, seed_type="single"):
    torch.manual_seed(sum(map(ord, text)) + 99)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")
    
    tgt = render_word(text, glyph)
    target = torch.from_numpy(tgt)[None].repeat(batch, 1, 1, 1).to(device)

    # Save target image for verification
    if snap_dir:
        Path(snap_dir).mkdir(parents=True, exist_ok=True)
        tgt_img = (tgt.transpose(1, 2, 0) * 255).astype(np.uint8)
        Image.fromarray(tgt_img).save(Path(snap_dir) / "target.png")

    model = NCA(channel_n, hidden_n=hidden_n).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=[int(steps * 0.8)], gamma=0.1)

    if seed_type == "noise":
        seed = make_noise_seed(text, channel_n)
    else:
        seed = make_single_seed(text, channel_n, tgt=tgt)
    
    pool = SamplePool(seed, pool_size)
    h, w = seed.shape[2], seed.shape[3]

    t0 = time.time()
    noise_idx = 0.60
    recent_losses = []
    for step in range(steps):
        idx, x = pool.sample(batch)
        x = x.to(device)
        with torch.no_grad():
            loss_rank = F.mse_loss(to_rgba(x), target, reduction="none") \
                .mean(dim=(1, 2, 3)).argsort(descending=True)
        x = x[loss_rank]
        x[:1] = seed.to(device) 
        if damage_n:
            m = damage_mask_rect(damage_n, h, w, device)
            x[-damage_n:] *= m

        n_ca = int(torch.randint(ca_min, ca_max + 1, (1,)))
        x = model(x, steps=n_ca)
        if noise_idx > 0:
            target_noisy = target * (1.0 - noise_idx) + torch.rand_like(target) * noise_idx
            loss = F.mse_loss(to_rgba(x), target_noisy)
        else:
            loss = F.mse_loss(to_rgba(x), target)
            if 'target_noisy' in locals():
                del target_noisy
        

        opt.zero_grad()
        loss.backward()
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    p.grad /= (p.grad.norm() + 1e-8)
        opt.step()
        sched.step()
        pool.commit(idx, x.cpu())

        recent_losses.append(loss.item())
        if len(recent_losses) > 100:
            recent_losses.pop(0)
            
        if len(recent_losses) == 100:
            avg_loss = sum(recent_losses) / 100.0
            if avg_loss < 0.035:
                noise_idx = max(0.0, noise_idx - 0.05)
                recent_losses.clear()
            elif avg_loss > 0.045:
                noise_idx = min(0.60, noise_idx + 0.01)
                recent_losses.clear()

        if step % log_every == 0 or step == steps - 1:
            if snap_dir:
                try:
                    
                    
                    
                    tgt_t = target_noisy if 'target_noisy' in locals() else target
                    a = tgt_t[0, 3:4].cpu()
                    rgb = tgt_t[0, :3].cpu()
                    tgt_img_arr = (1.0 - a + rgb).clamp(0,1).permute(1,2,0).numpy()
                    Image.fromarray((tgt_img_arr * 255).astype(np.uint8)).resize((target.shape[3] * 8, target.shape[2] * 8), getattr(Image, 'Resampling', Image).NEAREST).save(Path(snap_dir) / f'TARGET_{step:05d}.png')
                except Exception as e:
                    print(f'Fail target: {e}')
            print(f"[train_web_method4_{text}] step {step} loss {loss.item():.5f} "
                  f"({(time.time() - t0):.1f}s)", flush=True)
            if snap_dir:
                torch.save(model.state_dict(), str(Path(snap_dir) / 'latest.pth'))
                save_word_png(model, text, channel_n,
                              Path(snap_dir) / f"{text}_{step:05d}.png",
                              seed_type, device)

    if out:
        # Placeholder for export, since we don't need it yet
        pass
    print(f"Final loss for {text} (seed={seed_type}): {loss.item():.5f}")
    return model

def save_word_png(model, text, channel_n, path, seed_type, device, n_steps=120):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Upscale by 8 for visibility
    grow_word_image(model, text, channel_n, n_steps, upscale=8, seed_type=seed_type, device=device).save(path)

def grow_word_image(model, text, channel_n, n_steps=120, upscale=8, seed_type="single", device="cpu"):
    with torch.no_grad():
        if seed_type == "noise":
            x = make_noise_seed(text, channel_n)
        else:
            tgt = render_word(text, 12)
            x = make_single_seed(text, channel_n, tgt=tgt)
        x = model(x.to(device), steps=n_steps)
    img = to_rgb(x)[0].cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    im = Image.fromarray((img * 255).astype(np.uint8))
    rez_method = getattr(Image, "Resampling", Image).NEAREST # Use NEAREST for pixel art style
    return im.resize((im.width * upscale, im.height * upscale), rez_method)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--text", required=True)
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--out", default=None)
    p.add_argument("--snap-dir", default=None)
    p.add_argument("--seed-type", default="single")
    a = p.parse_args()
    
    train(a.text, steps=a.steps, log_every=a.log_every, out=a.out, snap_dir=a.snap_dir, seed_type=a.seed_type)
