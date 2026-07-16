import os
import numpy as np
from PIL import Image
from pathlib import Path
"""Train ONE NCA that grows a whole multi-character string from a SINGLE seed.
Method 5: Distance Field Gravitational Conduits.
Organic-looking conduits dynamically link letters.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from nca.model import NCA, to_rgba, to_rgb
from nca.train import FONT_PATH, char_color, damage_mask, SamplePool

PITCH = 14          # Lower pitch for closer letters
MARGIN = 6          # Lower margin
GRID_H = 20         # Lower height for lower res

def word_geometry(text):
    w = MARGIN * 2 + PITCH * len(text)
    return w, GRID_H

def simple_edt(mask):
    """Simple Euclidean Distance Transform in pure NumPy."""
    H, W = mask.shape
    y, x = np.mgrid[0:H, 0:W]
    y_on, x_on = np.where(mask > 0.5)
    
    if len(y_on) == 0:
        return np.full((H, W), 1e6)
    
    # Vectorized compute distance to all 'on' pixels
    dists = np.sqrt((y[:, :, None] - y_on[None, None, :])**2 + 
                    (x[:, :, None] - x_on[None, None, :])**2)
    
    return np.min(dists, axis=2)

def get_organic_conduit_paths(text, w, h, font, glyph):
    """Generate organic conduit paths using potential field derived from EDT."""
    letter_masks = []
    for i, ch in enumerate(text):
        img = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(img)
        x_center = MARGIN + PITCH * i + PITCH // 2
        l, t, r, b = draw.textbbox((0, 0), ch, font=font)
        draw.text((x_center - (r - l) / 2 - l, (h - (b - t)) / 2 - t), ch,
                  font=font, fill=255)
        letter_masks.append(np.asarray(img, dtype=np.float32) / 255.0)
        
    edts = [simple_edt(mask) for mask in letter_masks]
    
    epsilon = 1.0
    potential = np.zeros((w, h), dtype=np.float32).T # Match [H, W]
    for edt in edts:
        potential += 1.0 / (edt + epsilon)
        
    Gy, Gx = np.gradient(potential)
    
    all_paths = []
    
    for i in range(len(text) - 1):
        x_mid = MARGIN + PITCH * (i + 1)
        y_mid = h // 2
        
        # Perturb left
        path_left = [[y_mid, x_mid - 0.1]]
        for _ in range(100):
            y_curr, x_curr = int(path_left[-1][0]), int(path_left[-1][1])
            if y_curr < 0 or y_curr >= h or x_curr < 0 or x_curr >= w:
                break
            if letter_masks[i][y_curr, x_curr] > 0.5: # Hit left letter
                break
            
            gy = Gy[y_curr, x_curr]
            gx = Gx[y_curr, x_curr]
            norm = np.sqrt(gy**2 + gx**2) + 1e-6
            step_y = gy / norm
            step_x = gx / norm
            
            # Step UP gradient
            next_y = path_left[-1][0] + step_y * 0.5
            next_x = path_left[-1][1] + step_x * 0.5
            path_left.append([next_y, next_x])
        all_paths.append(path_left)
            
        # Perturb right
        path_right = [[y_mid, x_mid + 0.1]]
        for _ in range(100):
            y_curr, x_curr = int(path_right[-1][0]), int(path_right[-1][1])
            if y_curr < 0 or y_curr >= h or x_curr < 0 or x_curr >= w:
                break
            if letter_masks[i+1][y_curr, x_curr] > 0.5: # Hit right letter
                break
            
            gy = Gy[y_curr, x_curr]
            gx = Gx[y_curr, x_curr]
            norm = np.sqrt(gy**2 + gx**2) + 1e-6
            step_y = gy / norm
            step_x = gx / norm
            
            # Step UP gradient
            next_y = path_right[-1][0] + step_y * 0.5
            next_x = path_right[-1][1] + step_x * 0.5
            path_right.append([next_y, next_x])
        all_paths.append(path_right)
        
    return all_paths

def render_word(text, glyph=12, font_path=FONT_PATH):
    """Whole string on a transparent canvas, per-char color, with organic conduits."""
    w, h = word_geometry(text)
    font = ImageFont.truetype(font_path, glyph)
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Generate conduits
    paths = get_organic_conduit_paths(text, w, h, font, glyph)
    
    # Draw conduits with faint alpha
    for path in paths:
        if len(path) > 1:
            # Convert to (x, y) tuples
            path_tuples = [(p[1], p[0]) for p in path]
            draw.line(path_tuples, fill=(128, 128, 128, 64), width=1)

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
    return x

def train(text, steps=4000, glyph=12, channel_n=32, hidden_n=128,
          batch=4, pool_size=64, lr=2e-3, damage_n=1, ca_min=8, ca_max=16,
          log_every=1, out=None, snap_dir=None, seed_type="single"):
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
            m = damage_mask(damage_n, max(h, w), device)[:, :, :h, :w]
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
            print(f"[train_web_method5_{text}] step {step} loss {loss.item():.5f} "
                  f"({(time.time() - t0):.1f}s)", flush=True)
            if snap_dir:
                torch.save(model.state_dict(), str(Path(snap_dir) / 'latest.pth'))
                save_word_png(model, text, channel_n,
                              Path(snap_dir) / f"{text}_{step:05d}.png",
                              seed_type, device)


    if out:
        pass
    print(f"Final loss for {text} (seed={seed_type}): {loss.item():.5f}")
    return model

def save_word_png(model, text, channel_n, path, seed_type, device, n_steps=120):
    path.parent.mkdir(parents=True, exist_ok=True)
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
    rez_method = getattr(Image, "Resampling", Image).NEAREST
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
