import os
import numpy as np
from PIL import Image
from pathlib import Path
"""Train ONE NCA that grows a whole multi-character string from a SINGLE seed.
V2: Closer letters, organic tree supports, lower resolution (pixel art style).
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

def draw_branch(draw, x, y, length, angle, depth, fill, rng):
    """Draw a recursive branching structure (L-system style)."""
    if depth == 0 or length < 1:
        return
    
    # Angle has some noise
    angle = angle + rng.uniform(-10, 10)
    
    x_end = x + length * np.cos(np.deg2rad(angle))
    y_end = y + length * np.sin(np.deg2rad(angle))
    
    # Draw current segment
    draw.line((x, y, x_end, y_end), fill=fill, width=1)
    
    # Branching
    new_length = length * rng.uniform(0.6, 0.8)
    angle_spread = rng.uniform(20, 45)
    
    # Primary branch continues somewhat in same direction
    draw_branch(draw, x_end, y_end, new_length, angle + rng.uniform(-15, 15), depth - 1, fill, rng)
    
    # Secondary branches
    if rng.rand() < 0.7: # 70% chance to branch left
        draw_branch(draw, x_end, y_end, new_length * 0.8, angle - angle_spread, depth - 1, fill, rng)
    if rng.rand() < 0.7: # 70% chance to branch right
        draw_branch(draw, x_end, y_end, new_length * 0.8, angle + angle_spread, depth - 1, fill, rng)

def render_word(text, glyph=12, font_path=FONT_PATH):
    """Whole string on a transparent canvas, per-char color, with organic tree supports."""
    w, h = word_geometry(text)
    font = ImageFont.truetype(font_path, glyph)
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Deterministic RNG based on text
    seed = sum(map(ord, text))
    rng = np.random.RandomState(seed)
    
    # Faint color for supports
    support_fill = (128, 128, 128, 64) # Faint grey
    
    # Draw organic tree supports linking adjacent letters
    for i in range(len(text) - 1):
        x1 = MARGIN + PITCH * i + PITCH // 2
        x2 = MARGIN + PITCH * (i + 1) + PITCH // 2
        y1 = h // 2
        y2 = h // 2
        
        # Link horizontally
        dx = x2 - x1
        dy = y2 - y1
        angle = np.rad2deg(np.arctan2(dy, dx))
        
        # Originate from center/boundary of s1 or some points between
        num_tendrils = 3
        for _ in range(num_tendrils):
            # Start slightly offset from s1 towards s2
            x_start = x1 + rng.uniform(0, 10)
            y_start = y1 + rng.uniform(-5, 5)
            
            # Grow towards s2
            length = dx * rng.uniform(0.4, 0.6)
            draw_branch(draw, x_start, y_start, length, angle, depth=5, fill=support_fill, rng=rng)
            
            # Also grow from s2 towards s1
            x_start_rev = x2 - rng.uniform(0, 10)
            y_start_rev = y2 + rng.uniform(-5, 5)
            draw_branch(draw, x_start_rev, y_start_rev, length, angle + 180, depth=5, fill=support_fill, rng=rng)

    # Secondary supports from bottom
    for i in range(len(text)):
        x_center = MARGIN + PITCH * i + PITCH // 2
        if rng.rand() < 0.5: # 50% chance per letter
            x_start = x_center + rng.uniform(-5, 5)
            y_start = h - 1 # bottom
            # Grow upwards
            length = rng.uniform(5, 12)
            draw_branch(draw, x_start, y_start, length, -90, depth=4, fill=support_fill, rng=rng)

    # Draw letters over supports
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

def train(text, steps=4000, glyph=12, channel_n=16, hidden_n=80,
          batch=4, pool_size=256, lr=2e-3, damage_n=1, ca_min=64, ca_max=96,
          log_every=100, out=None, snap_dir=None, seed_type="single"):
    torch.manual_seed(sum(map(ord, text)) + 99)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")
    
    print(f"Rendering word...", flush=True)
    tgt = render_word(text, glyph)
    print(f"Word rendered. Target shape: {tgt.shape}", flush=True)
    target = torch.from_numpy(tgt)[None].repeat(batch, 1, 1, 1).to(device)

    # Save target image for verification
    if snap_dir:
        Path(snap_dir).mkdir(parents=True, exist_ok=True)
        tgt_img = (tgt.transpose(1, 2, 0) * 255).astype(np.uint8)
        Image.fromarray(tgt_img).save(Path(snap_dir) / "target.png")
        print(f"Saved target image to {snap_dir}/target.png", flush=True)

    print(f"Initializing NCA model...", flush=True)
    model = NCA(channel_n, hidden_n=hidden_n).to(device)
    print(f"NCA model initialized.", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=[int(steps * 0.8)], gamma=0.1)

    print(f"Creating seed...", flush=True)
    if seed_type == "noise":
        seed = make_noise_seed(text, channel_n)
    else:
        seed = make_single_seed(text, channel_n, tgt=tgt)
    print(f"Seed created. Seed shape: {seed.shape}", flush=True)
    
    pool = SamplePool(seed, pool_size)
    h, w = seed.shape[2], seed.shape[3]

    print(f"Starting training loop...", flush=True)
    t0 = time.time()
    noise_idx = 0.60
    recent_losses = []
    for step in range(steps):
        if step == 0: print(f"Step 0: Sampling pool...", flush=True)
        idx, x = pool.sample(batch)
        if step == 0: print(f"Step 0: Moving to device...", flush=True)
        x = x.to(device)
        if step == 0: print(f"Step 0: Ranking loss...", flush=True)
        with torch.no_grad():
            loss_rank = F.mse_loss(to_rgba(x), target, reduction="none") \
                .mean(dim=(1, 2, 3)).argsort(descending=True)
        x = x[loss_rank]
        x[:1] = seed.to(device) 
        if damage_n:
            if step == 0: print(f"Step 0: Applying damage...", flush=True)
            m = damage_mask(damage_n, max(h, w), device)[:, :, :h, :w]
            x[-damage_n:] *= m

        if step == 0: print(f"Step 0: Running NCA model...", flush=True)
        n_ca = int(torch.randint(ca_min, ca_max + 1, (1,)))
        x = model(x, steps=n_ca)
        if step == 0: print(f"Step 0: Computing loss...", flush=True)
        if noise_idx > 0:
            target_noisy = target * (1.0 - noise_idx) + torch.rand_like(target) * noise_idx
            loss = F.mse_loss(to_rgba(x), target_noisy)
        else:
            loss = F.mse_loss(to_rgba(x), target)
            if 'target_noisy' in locals():
                del target_noisy
        

        if step == 0: print(f"Step 0: Backward pass...", flush=True)
        opt.zero_grad()
        loss.backward()
        if step == 0: print(f"Step 0: Clipping gradients...", flush=True)
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    p.grad /= (p.grad.norm() + 1e-8)
        if step == 0: print(f"Step 0: Optimizer step...", flush=True)
        opt.step()
        sched.step()
        if step == 0: print(f"Step 0: Committing pool...", flush=True)
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
            print(f"[train_method2_{text}] step {step} loss {loss.item():.5f} "
                  f"({(time.time() - t0):.1f}s)", flush=True)
            if snap_dir:
                print(f"Saving snapshot to {snap_dir}...", flush=True)
                torch.save(model.state_dict(), str(Path(snap_dir) / 'latest.pth'))
                save_word_png(model, text, channel_n,
                              Path(snap_dir) / f"{text}_{step:05d}.png",
                              seed_type, device)
                print(f"Saved snapshot.", flush=True)


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
