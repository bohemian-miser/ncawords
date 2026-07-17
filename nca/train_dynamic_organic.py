import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import time
import argparse

from nca.model import NCA, to_rgba, to_rgb
from nca.train import FONT_PATH, char_color, damage_mask, SamplePool
from nca.experiment import Experiment
from nca.checkpoint import save_checkpoint, try_resume

def word_geometry(text):
    PITCH = 14
    MARGIN = 6
    GRID_H = 20
    w = MARGIN * 2 + PITCH * len(text)
    return w, GRID_H

def draw_branch(draw, x, y, length, angle, depth, fill, rng):
    if depth == 0 or length < 1:
        return
    angle = angle + rng.uniform(-10, 10)
    x_end = x + length * np.cos(np.deg2rad(angle))
    y_end = y + length * np.sin(np.deg2rad(angle))
    draw.line((x, y, x_end, y_end), fill=fill, width=1)
    new_length = length * rng.uniform(0.6, 0.8)
    angle_spread = rng.uniform(20, 45)
    draw_branch(draw, x_end, y_end, new_length, angle + rng.uniform(-15, 15), depth - 1, fill, rng)
    if rng.rand() < 0.7:
        draw_branch(draw, x_end, y_end, new_length * 0.8, angle - angle_spread, depth - 1, fill, rng)
    if rng.rand() < 0.7:
        draw_branch(draw, x_end, y_end, new_length * 0.8, angle + angle_spread, depth - 1, fill, rng)

def render_word(text, glyph=12, font_path=FONT_PATH, organic_seed=None, support_vol=1.0):
    PITCH = 14
    MARGIN = 6
    w, h = word_geometry(text)
    font = ImageFont.truetype(font_path, glyph)
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    seed = sum(map(ord, text)) if organic_seed is None else organic_seed
    rng = np.random.RandomState(seed)
    
    support_fill = (128, 128, 128, 64)
    
    for i in range(len(text) - 1):
        x1 = MARGIN + PITCH * i + PITCH // 2
        x2 = MARGIN + PITCH * (i + 1) + PITCH // 2
        y1 = h // 2
        y2 = h // 2
        dx = x2 - x1
        dy = y2 - y1
        angle = np.rad2deg(np.arctan2(dy, dx))
        
        num_tendrils = 3
        for _ in range(num_tendrils):
            if rng.rand() > support_vol: continue
            x_start = x1 + rng.uniform(0, 10)
            y_start = y1 + rng.uniform(-5, 5)
            length = dx * rng.uniform(0.4, 0.6)
            draw_branch(draw, x_start, y_start, length, angle, depth=5, fill=support_fill, rng=rng)
            
            x_start_rev = x2 - rng.uniform(0, 10)
            y_start_rev = y2 + rng.uniform(-5, 5)
            draw_branch(draw, x_start_rev, y_start_rev, length, angle + 180, depth=5, fill=support_fill, rng=rng)

    for i in range(len(text)):
        x_center = MARGIN + PITCH * i + PITCH // 2
        if rng.rand() < 0.5 * support_vol:
            x_start = x_center + rng.uniform(-5, 5)
            y_start = h - 1
            length = rng.uniform(5, 12)
            draw_branch(draw, x_start, y_start, length, -90, depth=4, fill=support_fill, rng=rng)

    for i, ch in enumerate(text):
        x_center = MARGIN + PITCH * i + PITCH // 2
        l, t, r, b = draw.textbbox((0, 0), ch, font=font)
        draw.text((x_center - (r - l) / 2 - l, (h - (b - t)) / 2 - t), ch,
                  font=font, fill=char_color(ch) + (255,))
        
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr[..., :3] *= arr[..., 3:]
    return arr.transpose(2, 0, 1)

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

class DynamicOrganicExperiment(Experiment):
    ID = "dynamic_organic"
    TITLE = "Dynamic Organic Branches"
    DESCRIPTION = "Dynamically generates organic branches/tendrils as a target."
    SEED_TYPE = "single"

    def __init__(self, base_dir=".", text="COMP", **kwargs):
        super().__init__(base_dir)
        self.text = text
        self.glyph = kwargs.get("glyph", 12)
        self.channel_n = kwargs.get("channel_n", 16)
        self.hidden_n = kwargs.get("hidden_n", 80)
        self.batch = kwargs.get("batch", 4)
        self.pool_size = kwargs.get("pool_size", 256)
        self.lr = kwargs.get("lr", 2e-3)
        self.damage_n = kwargs.get("damage_n", 1)
        self.ca_min = kwargs.get("ca_min", 64)
        self.ca_max = kwargs.get("ca_max", 96)
        self.log_every = kwargs.get("log_every", 100)
        self.update_every = kwargs.get("update_every", 1)
        self.no_noise = kwargs.get("no_noise", False)
        self.support_vol = kwargs.get("support_vol", 1.0)
        self.seed_type = kwargs.get("seed_type", "single")
        self.SEED_TYPE = self.seed_type
    
    def generate_proposed_targets(self, total_steps: int = 8000):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        tgt = render_word(self.text, self.glyph, organic_seed=0, support_vol=self.support_vol)
        organic_seed = 0
        for step in range(total_steps):
            if step % self.update_every == 0:
                organic_seed += 1
                tgt = render_word(self.text, self.glyph, organic_seed=organic_seed, support_vol=self.support_vol)
            
            # just save a few for preview
            if step % (total_steps // 10 if total_steps >= 10 else 1) == 0 or step == total_steps - 1:
                t = torch.from_numpy(tgt)
                a = t[3:4]
                rgb = t[:3]
                tgt_img_arr = (1.0 - a + rgb).clamp(0,1).permute(1,2,0).numpy()
                im_tgt = Image.fromarray((tgt_img_arr * 255).astype(np.uint8))
                im_tgt = im_tgt.resize((im_tgt.width * 8, im_tgt.height * 8), getattr(Image, 'Resampling', Image).NEAREST)
                im_tgt.save(self.output_dir / f"TARGET_{step:05d}.png")
                
    def grow_word_image(self, model, n_steps=120, upscale=8, device="cpu"):
        with torch.no_grad():
            if self.seed_type == "noise":
                x = make_noise_seed(self.text, self.channel_n)
            else:
                tgt = render_word(self.text, 12, organic_seed=0)
                x = make_single_seed(self.text, self.channel_n, tgt=tgt)
            x = model(x.to(device), steps=n_steps)
        img = to_rgb(x)[0].cpu().clamp(0, 1).permute(1, 2, 0).numpy()
        im = Image.fromarray((img * 255).astype(np.uint8))
        rez_method = getattr(Image, "Resampling", Image).NEAREST
        return im.resize((im.width * upscale, im.height * upscale), rez_method)

    def save_word_png(self, model, path, device, n_steps=120):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.grow_word_image(model, n_steps, upscale=8, device=device).save(path)

    def train(self, total_steps: int = 8000):
        torch.manual_seed(sum(map(ord, self.text)) + 99)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Training on device: {device}")
        
        def get_target(rnd_seed):
            tgt = render_word(self.text, self.glyph, organic_seed=rnd_seed, support_vol=self.support_vol)
            return tgt, torch.from_numpy(tgt)[None].repeat(self.batch, 1, 1, 1).to(device)

        tgt, target = get_target(0)
        
        print(f"Initializing NCA model...")
        model = NCA(self.channel_n, hidden_n=self.hidden_n).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[int(total_steps * 0.8)], gamma=0.1)

        print(f"Creating seed...")
        if self.seed_type == "noise":
            seed = make_noise_seed(self.text, self.channel_n)
        else:
            seed = make_single_seed(self.text, self.channel_n, tgt=tgt)
        
        pool = SamplePool(seed, self.pool_size)
        h, w = seed.shape[2], seed.shape[3]

        print(f"Starting training loop...")
        t0 = time.time()
        noise_idx = 0.0 if self.no_noise else 0.60
        recent_losses = []

        organic_seed = 0
        start_step, ckpt_extra = try_resume(self.output_dir, model, opt, sched, pool, device)
        if start_step > 0:
            noise_idx = ckpt_extra.get("noise_idx", noise_idx)
            organic_seed = ckpt_extra.get("organic_seed", organic_seed)
            tgt, target = get_target(organic_seed)
        for step in range(start_step, total_steps):
            if step % self.update_every == 0:
                organic_seed += 1
                tgt, target = get_target(organic_seed)
                
            idx, x = pool.sample(self.batch)
            x = x.to(device)
            with torch.no_grad():
                loss_rank = F.mse_loss(to_rgba(x), target, reduction="none") \
                    .mean(dim=(1, 2, 3)).argsort(descending=True)
            x = x[loss_rank]
            x[:1] = seed.to(device) 
            if self.damage_n:
                m = damage_mask(self.damage_n, max(h, w), device)[:, :, :h, :w]
                x[-self.damage_n:] *= m

            n_ca = int(torch.randint(self.ca_min, self.ca_max + 1, (1,)))
            x = model(x, steps=n_ca)
            if noise_idx > 0:
                target_noisy = target * (1.0 - noise_idx) + torch.rand_like(target) * noise_idx
                loss = F.mse_loss(to_rgba(x), target_noisy)
            else:
                loss = F.mse_loss(to_rgba(x), target)

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
                if not self.no_noise:
                    if avg_loss < 0.035:
                        noise_idx = max(0.0, noise_idx - 0.05)
                    elif avg_loss > 0.045:
                        noise_idx = min(0.60, noise_idx + 0.01)
                recent_losses.clear()

            if step % self.log_every == 0 or step == total_steps - 1:
                os.makedirs(self.output_dir, exist_ok=True)
                try:
                    tgt_t = target_noisy if 'target_noisy' in locals() else target
                    a = tgt_t[0, 3:4].cpu()
                    rgb = tgt_t[0, :3].cpu()
                    tgt_img_arr = (1.0 - a + rgb).clamp(0,1).permute(1,2,0).numpy()
                    Image.fromarray((tgt_img_arr * 255).astype(np.uint8)).resize((target.shape[3] * 8, target.shape[2] * 8), getattr(Image, 'Resampling', Image).NEAREST).save(self.output_dir / f'TARGET_{step:05d}.png')
                except Exception as e:
                    print(f'Fail target: {e}')
                    
                print(f"[train_dynamic_organic_{self.text}] step {step} loss {loss.item():.5f} noise_idx {noise_idx:.2f} ({time.time() - t0:.1f}s)", flush=True)
                torch.save(model.state_dict(), str(self.output_dir / 'latest.pth'))
                save_checkpoint(self.output_dir, step, model, opt, sched, pool,
                                extra={"noise_idx": noise_idx, "organic_seed": organic_seed})
                # Output COMP_{step:05d}.png so it runs in UI alongside targets
                self.save_word_png(model, self.output_dir / f"COMP_{step:05d}.png", device)

        print(f"Final loss for {self.text} (seed={self.seed_type}): {loss.item():.5f}")
        return model

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--text", required=True)
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--snap-dir", default="snaps_dynamic_organic")
    p.add_argument("--seed-type", default="single")
    p.add_argument("--update-every", type=int, default=1)
    p.add_argument("--no-noise", action="store_true")
    p.add_argument("--support-vol", type=float, default=1.0)
    a = p.parse_args()
    
    exp = DynamicOrganicExperiment(base_dir=".", text=a.text, seed_type=a.seed_type, 
                                   update_every=a.update_every, no_noise=a.no_noise, 
                                   support_vol=a.support_vol)
    exp.output_dir = Path(a.snap_dir)
    exp.log_every = a.log_every
    exp.train(total_steps=a.steps)
