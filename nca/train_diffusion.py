import os
import numpy as np
from PIL import Image
from pathlib import Path
"""Train ONE NCA that grows a whole multi-character string from a SINGLE seed.
Method: 9-line web between bounding boxes of adjacent letters.
Includes the damage_mask_rect bugfix.
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

def render_word_9_line(text, glyph=12, font_path=FONT_PATH):
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
        strand_color = (128, 128, 128, 64)
        
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

def get_curriculum_pair(step, total_steps):
    pair_idx = int((step / total_steps) * 55)
    pair_idx = min(pair_idx, 54)
    count = 0
    for gap_idx in range(1, 11):
        gap = gap_idx / 10.0
        num_pairs_in_gap = 11 - gap_idx
        if count + num_pairs_in_gap > pair_idx:
            offset = pair_idx - count
            start_x = 1.0 - (offset * 0.1)
            target_y = start_x - gap
            return round(start_x, 2), round(target_y, 2)
        count += num_pairs_in_gap
    return 1.0, 0.0

def build_state(tgt_tensor, noise_level, channel_n, batch, device):
    state = torch.rand(batch, channel_n, tgt_tensor.shape[1], tgt_tensor.shape[2], device=device)
    state[:, :4] = (1.0 - noise_level) * tgt_tensor + noise_level * state[:, :4]
    state[:, 4:] = noise_level * state[:, 4:]
    return state

def train(text, steps=4000, glyph=12, channel_n=16, hidden_n=80,
          batch=8, pool_size=256, lr=2e-3, damage_n=1, ca_min=64, ca_max=96,
          log_every=100, out=None, snap_dir=None, seed_type="single"):
    torch.manual_seed(sum(map(ord, text)) + 99)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")
    
    tgt = render_word_9_line(text, glyph)
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

    tgt_single = torch.from_numpy(tgt).to(device) # [4, h, w]
    
    t0 = time.time()
    for step in range(steps):
        # 1. Ask curriculum for noise brackets
        input_noise, target_noise = get_curriculum_pair(step, steps)
        
        # 2. Build input state with X noise
        with torch.no_grad():
            x0 = build_state(tgt_single, input_noise, channel_n, batch, device)
            # 3. Build target state with Y noise
            x_target = build_state(tgt_single, target_noise, channel_n, batch, device)
            
        # Optional: slight random variations to steps
        n_steps = int(np.random.randint(48, 64))
        
        # 4. Forward
        x = model(x0, steps=n_steps)
        
        # 5. Loss: compare RGBA output against the Y noise target
        loss = F.mse_loss(x[:, :4], x_target[:, :4])

        opt.zero_grad()
        loss.backward()
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    p.grad /= (p.grad.norm() + 1e-8)
        opt.step()
        sched.step()

        if step % log_every == 0 or step == steps - 1:
            print(f"[train_diffusion_{text}] step {step} loss {loss.item():.4f} X: {input_noise:.2f} -> Y: {target_noise:.2f} ({time.time() - t0:.1f}s)", flush=True)
            if snap_dir:
                img = to_rgb(x0)[0].detach().cpu().numpy()
                Image.fromarray((img.transpose(1, 2, 0) * 255).astype(np.uint8)).save(Path(snap_dir) / f"{text}_{step:05d}_in.png")
                
                img = to_rgb(x_target)[0].detach().cpu().numpy()
                Image.fromarray((img.transpose(1, 2, 0) * 255).astype(np.uint8)).save(Path(snap_dir) / f"{text}_{step:05d}_tgt.png")
                
                img_out = to_rgb(x)[0].detach().cpu().numpy()
                Image.fromarray((img_out.transpose(1, 2, 0) * 255).astype(np.uint8)).save(Path(snap_dir) / f"{text}_{step:05d}_out.png")
                
                with open(Path(snap_dir) / "loss.txt", "w") as f:
                    f.write(str(loss.item()))
                
                import subprocess
                try:
                    subprocess.run(["python", "update_dashboard.py"])
                except Exception as e:
                    pass
                
                torch.save(model.state_dict(), Path(snap_dir) / "latest.pth")

    print(f"Final loss for {text} (diffusion): {loss.item():.5f}")
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
