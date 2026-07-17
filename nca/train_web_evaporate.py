import os
import numpy as np
from PIL import Image
from pathlib import Path
"""Train ONE NCA that grows a whole multi-character string from a SINGLE seed.
Method: 9-line scaffold that EVAPORATES over training.
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
from nca.train import FONT_PATH, char_color, SamplePool
from nca.checkpoint import save_checkpoint, try_resume
from nca.runmeta import RunMeta, export_run_weights

PITCH = 14          # Lower pitch for closer letters
MARGIN = 6          # Lower margin
GRID_H = 20         # Lower height for lower res

def word_geometry(text):
    w = MARGIN * 2 + PITCH * len(text)
    return w, GRID_H

def damage_mask_rect(n, h, w, device):
    """1 outside a random circle, 0 inside (multiply to damage).
    Correctly handles rectangular grids without coordinate bias."""
    y = torch.linspace(-1, 1, h, device=device)
    x = torch.linspace(-1, 1, w, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    
    # Expand for batch dimension
    xx = xx.unsqueeze(0).repeat(n, 1, 1)
    yy = yy.unsqueeze(0).repeat(n, 1, 1)
    
    # Center in middle half of grid
    cx = (torch.rand(n, 1, 1, device=device) - 0.5) # [-0.5, 0.5]
    cy = (torch.rand(n, 1, 1, device=device) - 0.5) # [-0.5, 0.5]
    
    # Radius 0.1-0.4 of grid scale (use min dimension to keep circle shape or max depending on intent)
    # Keeping it simple and symmetric in coordinate space
    r = torch.rand(n, 1, 1, device=device) * 0.3 + 0.1
    
    mask = ((xx - cx)**2 + (yy - cy)**2) > r**2
    return mask.unsqueeze(1) # [N, 1, H, W]

def render_word_9_line(text, glyph=12, font_path=FONT_PATH, strand_alpha=64):
    """Whole string on a transparent canvas, per-char color, with 9-line BB bridges."""
    w, h = word_geometry(text)
    font = ImageFont.truetype(font_path, glyph)
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # 1. Find bounding boxes for each character
    char_boxes = []
    for i, ch in enumerate(text):
        x_center = MARGIN + PITCH * i + PITCH // 2
        l, t, r, b = draw.textbbox((0, 0), ch, font=font)
        x_pos = x_center - (r - l) / 2 - l
        y_pos = (h - (b - t)) / 2 - t
        
        # Absolute boxes in image coordinates
        L = x_pos + l
        T = y_pos + t
        R = x_pos + r
        B = y_pos + b
        
        char_boxes.append({
            "char": ch,
            "L": L, "T": T, "R": R, "B": B,
            "x_pos": x_pos, "y_pos": y_pos
        })

    # 2. Draw 9 connecting strands per pair
    for i in range(len(char_boxes) - 1):
        boxA = char_boxes[i]
        boxB = char_boxes[i+1]
        
        # Individual Ys for A
        Y_top_A = boxA["T"]
        Y_cen_A = (boxA["T"] + boxA["B"]) / 2
        Y_bot_A = boxA["B"]
        
        # Individual Ys for B
        Y_top_B = boxB["T"]
        Y_cen_B = (boxB["T"] + boxB["B"]) / 2
        Y_bot_B = boxB["B"]
        
        # Edges
        R_A = boxA["R"]
        L_B = boxB["L"]
        
        # Low alpha strands
        strand_color = (128, 128, 128, strand_alpha)
        
        # 9 connections
        draw.line((R_A, Y_top_A, L_B, Y_top_B), fill=strand_color)
        draw.line((R_A, Y_top_A, L_B, Y_cen_B), fill=strand_color)
        draw.line((R_A, Y_top_A, L_B, Y_bot_B), fill=strand_color)
        
        draw.line((R_A, Y_cen_A, L_B, Y_top_B), fill=strand_color)
        draw.line((R_A, Y_cen_A, L_B, Y_cen_B), fill=strand_color)
        draw.line((R_A, Y_cen_A, L_B, Y_bot_B), fill=strand_color)
        
        draw.line((R_A, Y_bot_A, L_B, Y_top_B), fill=strand_color)
        draw.line((R_A, Y_bot_A, L_B, Y_cen_B), fill=strand_color)
        draw.line((R_A, Y_bot_A, L_B, Y_bot_B), fill=strand_color)

    # 3. Draw characters
    for box in char_boxes:
        draw.text((box["x_pos"], box["y_pos"]), box["char"],
                  font=font, fill=char_color(box["char"]) + (255,))
        
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

def train(text, steps=4000, glyph=12, channel_n=16, hidden_n=80,
          batch=8, pool_size=256, lr=2e-3, damage_n=1, ca_min=64, ca_max=96,
          log_every=100, out=None, snap_dir=None, seed_type="single"):
    torch.manual_seed(sum(map(ord, text)) + 99)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")
    
    tgt_start = render_word_9_line(text, glyph, strand_alpha=64)
    target_start = torch.from_numpy(tgt_start)[None].repeat(batch, 1, 1, 1).to(device)

    tgt_end = render_word_9_line(text, glyph, strand_alpha=0)
    target_end = torch.from_numpy(tgt_end)[None].repeat(batch, 1, 1, 1).to(device)

    # Save target image for verification
    if snap_dir:
        Path(snap_dir).mkdir(parents=True, exist_ok=True)
        tgt_img = (tgt_start.transpose(1, 2, 0) * 255).astype(np.uint8)
        Image.fromarray(tgt_img).save(Path(snap_dir) / "target_start.png")
        tgt_img2 = (tgt_end.transpose(1, 2, 0) * 255).astype(np.uint8)
        Image.fromarray(tgt_img2).save(Path(snap_dir) / "target_end.png")

    model = NCA(channel_n, hidden_n=hidden_n).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=[int(steps * 0.8)], gamma=0.1)

    seed = make_single_seed(text, channel_n, tgt=tgt_start)
    pool = SamplePool(seed, pool_size)
    h, w = seed.shape[2], seed.shape[3]

    t0 = time.time()
    noise_idx = 0.60
    start_step, ckpt_extra = try_resume(snap_dir, model, opt, sched, pool, device)
    noise_idx = ckpt_extra.get("noise_idx", noise_idx)
    meta = RunMeta(snap_dir, text, "nca.train_web_evaporate",
                   {"steps": steps, "glyph": glyph, "batch": batch, "lr": lr,
                    "damage_n": damage_n},
                   channel_n, hidden_n, seed_type, steps, device)
    recent_losses = []

    fade_start_step = int(steps * 0.6)
    fade_end_step = int(steps * 0.9)

    for step in range(start_step, steps):
        if step <= fade_start_step:
            ratio = 0.0
        elif step >= fade_end_step:
            ratio = 1.0
        else:
            ratio = (step - fade_start_step) / (fade_end_step - fade_start_step)
            
        target = target_start * (1.0 - ratio) + target_end * ratio
        
        idx, x = pool.sample(batch)
        x = x.to(device)
        with torch.no_grad():
            loss_rank = F.mse_loss(to_rgba(x), target, reduction="none") \
                .mean(dim=(1, 2, 3)).argsort(descending=True)
        x = x[loss_rank]
        x[:1] = seed.to(device) 
        if damage_n:
            # Use the corrected rectangular damage mask
            m = damage_mask_rect(damage_n, h, w, device)
            x[-damage_n:] *= m

        x_start = x[-1:].detach().clone()   # damaged input, for START snapshots
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
            print(f"[train_web_9_line_{text}] step {step} loss {loss.item():.5f} "
                  f"({(time.time() - t0):.1f}s)", flush=True)
            if snap_dir:
                torch.save(model.state_dict(), str(Path(snap_dir) / 'latest.pth'))
                save_checkpoint(snap_dir, step, model, opt, sched, pool,
                                extra={"noise_idx": noise_idx})
                meta.log(step, loss.item(), noise_idx=noise_idx)
                export_run_weights(model, snap_dir, text, glyph)
                for tag, t in [("START", x_start[0]), ("RECOV", x[-1])]:
                    img = to_rgba(t[None])[0].detach().cpu().clamp(0, 1)
                    vis = (1 - img[3:4] + img[:3]).clamp(0, 1).permute(1, 2, 0).numpy()
                    Image.fromarray((vis * 255).astype(np.uint8)) \
                        .resize((w * 8, h * 8), Image.NEAREST) \
                        .save(Path(snap_dir) / f"{tag}_{step:05d}.png")
                save_word_png(model, text, channel_n,
                              Path(snap_dir) / f"{text}_{step:05d}.png",
                              seed_type, device)

    print(f"Final loss for {text} (seed={seed_type}): {loss.item():.5f}")
    return model

def save_word_png(model, text, channel_n, path, seed_type, device, n_steps=120):
    path.parent.mkdir(parents=True, exist_ok=True)
    grow_word_image(model, text, channel_n, n_steps, upscale=8, seed_type=seed_type, device=device).save(path)

def grow_word_image(model, text, channel_n, n_steps=120, upscale=8, seed_type="single", device="cpu"):
    with torch.no_grad():
        tgt = render_word_9_line(text, 12)
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
