import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
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
    return arr.transpose(2, 0, 1)

def save_word_png(model, text, channel_n, step, snap_dir, x_idx, y_idx, device, target_np, n_steps=60):
    Path(snap_dir).mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        x_init = torch.zeros(1, channel_n, target_np.shape[1], target_np.shape[2]).to(device)
        x_init[:, :4] = torch.from_numpy(target_np).unsqueeze(0)
        x_init[:, :4, :, :] = x_init[:, :4, :, :] * (1.0 - x_idx) + torch.rand_like(x_init[:, :4, :, :]) * x_idx
        # Local stage field channel 4 initialized to initial noise magnitude
        x_init[:, 4:5, :, :] = x_idx
        x_init[:, 5:, :, :] = torch.randn_like(x_init[:, 5:, :, :]) * 0.05
        x_final = model(x_init.clone(), steps=n_steps)
        
    img_init = to_rgba(x_init)[0, :3].cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    img_final = to_rgba(x_final)[0, :3].cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    
    img_combined = np.concatenate([img_init, img_final], axis=1)
    im = Image.fromarray((img_combined * 255).astype(np.uint8))
    rez_method = getattr(Image, "Resampling", Image).NEAREST
    im = im.resize((im.width * 8, im.height * 8), rez_method)
    path = str(Path(snap_dir) / f"COMP_{step:05d}.png")
    im.save(path)

def train(text, steps=16000, glyph=12, channel_n=16, hidden_n=80,
          batch=8, pool_size=1024, lr=2e-3, damage_n=1, ca_min=64, ca_max=96,
          log_every=100, out=None, snap_dir="snaps_stage_local"):
    torch.manual_seed(sum(map(ord, text)) + 101)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Stage-Local R&D] Training on device: {device}", flush=True)
    
    tgt = render_word(text, glyph)
    target = torch.from_numpy(tgt)[None].repeat(batch, 1, 1, 1).to(device)

    model = NCA(channel_n, hidden_n=hidden_n).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=[int(steps * 0.8)], gamma=0.1)

    dummy_seed = torch.zeros(1, channel_n, tgt.shape[1], tgt.shape[2])
    pool = SamplePool(dummy_seed, pool_size)
    h, w = tgt.shape[1], tgt.shape[2]

    with torch.no_grad():
        for i in range(0, pool_size, 32):
            b = min(32, pool_size - i)
            base = torch.zeros(b, channel_n, tgt.shape[1], tgt.shape[2]).cpu()
            base[:, :4] = torch.from_numpy(tgt).unsqueeze(0)
            base[:, 4:5] = 0.0  # target stage is 0 when fully complete
            base[:, 5:] = torch.randn_like(base[:, 5:]) * 0.05
            pool.pool[i:i+b] = base

    curriculum = []
    for gap in range(10, 101, 10):
        min_x = 20 if gap == 10 else gap
        for x in range(100, min_x - 1, -10):
            curriculum.append((x / 100.0, (x - gap) / 100.0))
    curriculum.append((1.0, 0.0))
    
    start_step = 0
    curr_idx = 0

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
        # Channel 4 = local stage indicator (1.0 = high noise/raw stage)
        x_new[:, 4:5] = x_idx 
        x_new[:, 5:] = torch.randn_like(x_new[:, 5:]) * 0.05
        x[:1] = x_new
        
        if damage_n:
            m = damage_mask(damage_n, max(h, w), device)[:, :, :h, :w]
            x[-damage_n:] *= m
            # Spiking stage layer in damaged pixels to 1.0 so neighbors know local repair is required!
            x[-damage_n:, 4:5] = torch.where(m[:, :1] < 0.5, torch.ones_like(x[-damage_n:, 4:5]), x[-damage_n:, 4:5])

        n_ca = int(torch.randint(ca_min, ca_max + 1, (1,)))
        x = model(x, steps=n_ca)
        
        target_noisy = target * (1.0 - y_idx) + torch.rand_like(target) * y_idx
        loss_rgba = F.mse_loss(to_rgba(x), target_noisy)
        
        # Local stage field target: y_idx for background/noisy areas, 0.0 for fully resolved pixels
        local_target_stage = torch.full_like(x[:, 4:5, :, :], y_idx)
        loss_stage = F.mse_loss(x[:, 4:5, :, :], local_target_stage)
        
        loss = loss_rgba + 0.5 * loss_stage
        
        opt.zero_grad()
        loss.backward()
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    p.grad /= (p.grad.norm() + 1e-8)
        opt.step()
        sched.step()
        pool.commit(idx, x.cpu())

        curr_idx = min(step // 100, len(curriculum) - 1)

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
                except Exception as e:
                    print(f"Failed to snap: {e}")
                    
            print(f"[stage_local] step {step} loss_rgba {loss_rgba.item():.5f} loss_stage {loss_stage.item():.5f} X:{x_idx:.2f}->Y:{y_idx:.2f} ({(time.time() - t0):.1f}s)", flush=True)

    if out:
        torch.save(model.state_dict(), out)
    return model

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--text", default="COMP")
    p.add_argument("--steps", type=int, default=16000)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--out", default=None)
    p.add_argument("--snap-dir", default="snaps_stage_local")
    a = p.parse_args()
    
    train(a.text, steps=a.steps, log_every=a.log_every, out=a.out, snap_dir=a.snap_dir)
