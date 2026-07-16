import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont

from nca.model import NCA, to_rgba, to_rgb
from nca.train import FONT_PATH, char_color, damage_mask, SamplePool

PITCH = 14          
MARGIN = 6          
GRID_H = 20         

def word_geometry(text):
    w = MARGIN * 2 + PITCH * len(text)
    return w, GRID_H

def render_word(text, glyph=12, font_path=FONT_PATH):
    w, h = word_geometry(text)
    font = ImageFont.truetype(font_path, glyph)
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    for i, ch in enumerate(text):
        x_center = MARGIN + PITCH * i + PITCH // 2
        l, t, r, b = draw.textbbox((0, 0), ch, font=font)
        x_pos = x_center - (r - l) / 2 - l
        y_pos = (h - (b - t)) / 2 - t
        draw.text((x_pos, y_pos), ch, font=font, fill=char_color(ch) + (255,))
        
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr[..., :3] *= arr[..., 3:]
    return arr.transpose(2, 0, 1)  # [4, H, W]

def save_word_png(model, text, channel_n, step, snap_dir, x_idx, y_idx, device, target_np, n_steps=60):
    Path(snap_dir).mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        x_init = torch.zeros(1, channel_n, target_np.shape[1], target_np.shape[2]).to(device)
        x_init[:, :4] = torch.from_numpy(target_np).unsqueeze(0)
        x_init[:, :4, :, :] = x_init[:, :4, :, :] * (1.0 - x_idx) + torch.rand_like(x_init[:, :4, :, :]) * x_idx
        x_init[:, 4:, :, :] = torch.randn_like(x_init[:, 4:, :, :]) * 0.05
        x_final = model(x_init.clone(), steps=n_steps)
        
    img_init = to_rgba(x_init)[0, :3].cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    img_final = to_rgba(x_final)[0, :3].cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    
    img_combined = np.concatenate([img_init, img_final], axis=1) # [H, 2W, 3]
    im = Image.fromarray((img_combined * 255).astype(np.uint8))
    rez_method = getattr(Image, "Resampling", Image).NEAREST
    im = im.resize((im.width * 8, im.height * 8), rez_method)
    path = str(Path(snap_dir) / f"COMP_{step:05d}.png")
    im.save(path)

def train(text, steps=16000, glyph=12, channel_n=16, hidden_n=80,
          batch=8, pool_size=1024, lr=2e-3, damage_n=1, ca_min=64, ca_max=96,
          log_every=100, out=None, snap_dir="snaps_adaptive_cloud"):
    torch.manual_seed(sum(map(ord, text)) + 99)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}", flush=True)
    
    tgt = render_word(text, glyph)
    target = torch.from_numpy(tgt)[None].repeat(batch, 1, 1, 1).to(device)

    model = NCA(channel_n, hidden_n=hidden_n).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=[int(steps * 0.8)], gamma=0.1)

    dummy_seed = torch.zeros(1, channel_n, tgt.shape[1], tgt.shape[2])
    pool = SamplePool(dummy_seed, pool_size)
    h, w = tgt.shape[1], tgt.shape[2]

    # Pre-populate pool with target image content + noise
    with torch.no_grad():
        for i in range(0, pool_size, 32):
            b = min(32, pool_size - i)
            base = torch.zeros(b, channel_n, tgt.shape[1], tgt.shape[2]).cpu()
            base[:, :4] = torch.from_numpy(tgt).unsqueeze(0)
            base[:, 4:] = torch.randn_like(base[:, 4:]) * 0.05
            pool.pool[i:i+b] = base

    # Curriculum generation
    curriculum = []
    for gap in range(10, 101, 10):
        min_x = 20 if gap == 10 else gap
        for x in range(100, min_x - 1, -10):
            curriculum.append((x / 100.0, (x - gap) / 100.0))
    curriculum.append((1.0, 0.0)) # Ensure 100->0 is final
    
    start_step = 0
    curr_idx = 0
    recent_losses = []

    # Resumability Logic
    if snap_dir:
        Path(snap_dir).mkdir(parents=True, exist_ok=True)
        latest_pth = Path(snap_dir) / 'latest.pth'
        if latest_pth.exists():
            try:
                ckpt = torch.load(latest_pth, map_location=device)
                if 'model_state_dict' in ckpt:
                    model.load_state_dict(ckpt['model_state_dict'])
                    opt.load_state_dict(ckpt['optimizer_state_dict'])
                    start_step = ckpt['step'] + 1
                    curr_idx = ckpt.get('curr_idx', 0)
                    print(f"Resumed from step {start_step}, curriculum index {curr_idx}")
                else:
                    model.load_state_dict(ckpt)
                    print("Resumed from legacy model dict")
            except Exception as e:
                print(f"Failed to resume: {e}")

    t0 = time.time()
    for step in range(start_step, steps):
        x_idx, y_idx = curriculum[min(curr_idx, len(curriculum) - 1)]
        
        idx, x = pool.sample(batch)
        x = x.to(device)
        
        with torch.no_grad():
            target_noisy_for_rank = target * (1.0 - y_idx) + torch.rand_like(target) * y_idx
            loss_rank = F.mse_loss(to_rgba(x), target_noisy_for_rank, reduction="none") \
                .mean(dim=(1, 2, 3)).argsort(descending=True)
        x = x[loss_rank]
        
        # Fresh seed replacing worst sample
        x_new = torch.zeros(1, channel_n, h, w).to(device)
        tgt_slice = target[:1]
        x_new[:, :4] = tgt_slice * (1.0 - x_idx) + torch.rand_like(tgt_slice) * x_idx
        x_new[:, 4:] = torch.randn_like(x_new[:, 4:]) * 0.05
        x[:1] = x_new
        
        if damage_n:
            m = damage_mask(damage_n, max(h, w), device)[:, :, :h, :w]
            x[-damage_n:] *= m

        n_ca = int(torch.randint(ca_min, ca_max + 1, (1,)))
        x = model(x, steps=n_ca)
        
        target_noisy = target * (1.0 - y_idx) + torch.rand_like(target) * y_idx
        loss = F.mse_loss(to_rgba(x), target_noisy)
        
        opt.zero_grad()
        loss.backward()
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    p.grad /= (p.grad.norm() + 1e-8)
        opt.step()
        sched.step()
        pool.commit(idx, x.cpu())

        # Adaptive progression
        recent_losses.append(loss.item())
        if len(recent_losses) > 100:
            recent_losses.pop(0)
        
        if len(recent_losses) == 100:
            avg_loss = sum(recent_losses) / 100.0
            if avg_loss < 0.035:
                # Progress to next curriculum stage
                curr_idx = min(curr_idx + 1, len(curriculum) - 1)
                recent_losses.clear()
            elif avg_loss > 0.06:
                # Revert a bit if struggling hard
                curr_idx = max(0, curr_idx - 1)
                recent_losses.clear()

        if step % log_every == 0 or step == steps - 1:
            if snap_dir:
                try:
                    ckpt = {
                        'step': step,
                        'curr_idx': curr_idx,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': opt.state_dict()
                    }
                    torch.save(ckpt, str(Path(snap_dir) / 'latest.pth'))
                    save_word_png(model, text, channel_n, step, snap_dir, x_idx, y_idx, device, tgt)
                    
                    # Save TARGET image
                    tgt_t = target_noisy[0].cpu()
                    a = tgt_t[3:4]
                    rgb = tgt_t[:3]
                    tgt_img_arr = (1.0 - a + rgb).clamp(0,1).permute(1,2,0).numpy()
                    im_tgt = Image.fromarray((tgt_img_arr * 255).astype(np.uint8))
                    im_tgt = im_tgt.resize((im_tgt.width * 8, im_tgt.height * 8), getattr(Image, 'Resampling', Image).NEAREST)
                    
                    # Annotate the image with percentages
                    draw = ImageDraw.Draw(im_tgt)
                    font_target = ImageFont.truetype(FONT_PATH, 16)
                    info_str = f"Given: {int(x_idx*100)}% | Target: {int(y_idx*100)}%"
                    
                    # Draw a black rect as background for text
                    draw.rectangle([0, 0, im_tgt.width, 24], fill=(0,0,0,180))
                    draw.text((4, 2), info_str, fill=(255, 255, 255), font=font_target)
                    
                    im_tgt.save(Path(snap_dir) / f"TARGET_{step:05d}.png")
                except Exception as e:
                    print(f"Failed to snap: {e}")
                    
                import subprocess
                subprocess.Popen(["venv/bin/python", "update_dashboard.py"])
                
            print(f"[adaptive_cloud] step {step} loss {loss.item():.5f} X:{x_idx:.2f}->Y:{y_idx:.2f} (idx:{curr_idx}/{len(curriculum)}) {(time.time() - t0):.1f}s", flush=True)

    print(f"Final loss for {text}: {loss.item():.5f}")
    if out:
        torch.save(model.state_dict(), out)
    return model

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--text", default="COMP")
    p.add_argument("--steps", type=int, default=16000)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--out", default=None)
    p.add_argument("--snap-dir", default="snaps_adaptive_cloud")
    a = p.parse_args()
    
    train(a.text, steps=a.steps, log_every=a.log_every, out=a.out, snap_dir=a.snap_dir)
