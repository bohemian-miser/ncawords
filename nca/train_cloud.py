import argparse
import time
from pathlib import Path
import os

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont

from nca.model import NCA, to_rgba, to_rgb
from nca.train import FONT_PATH, char_color, damage_mask, SamplePool
from nca.experiment import Experiment

def word_geometry(text):
    PITCH = 14
    MARGIN = 6
    GRID_H = 20
    w = MARGIN * 2 + PITCH * len(text)
    return w, GRID_H

def render_word(text, glyph=12, font_path=FONT_PATH):
    PITCH = 14
    MARGIN = 6
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

def make_cloud_seed(tgt, channel_n, n, device="cpu"):
    # tgt: [4, H, W] numpy array
    x = torch.zeros(n, channel_n, tgt.shape[1], tgt.shape[2], device=device)
    tgt_t = torch.from_numpy(tgt).unsqueeze(0).to(device)  # [1, 4, H, W]
    
    for i in range(n):
        angle = (torch.rand(1).item() - 0.5) * 20.0
        max_trans_h = tgt.shape[1] * 0.1
        max_trans_w = tgt.shape[2] * 0.1
        translate = [int((torch.rand(1).item() - 0.5) * max_trans_w), int((torch.rand(1).item() - 0.5) * max_trans_h)]
        scale = 1.0 + torch.rand(1).item() * 0.4
        shear_val = (torch.rand(1).item() - 0.5) * 15.0
        
        dist = TF.affine(tgt_t, angle=angle, translate=translate, scale=scale, shear=[shear_val], 
                         interpolation=TF.InterpolationMode.BILINEAR)
        
        k = int(torch.randint(5, 11, (1,)).item())
        if k % 2 == 0: k += 1
        sig = torch.rand(1).item() * 2.0 + 1.5
        dist = TF.gaussian_blur(dist, kernel_size=[k, k], sigma=[sig, sig])
        
        noise = torch.randn_like(dist) * 0.2
        dist = torch.clamp(dist + noise, 0.0, 1.0)
        
        x[i, :4, :, :] = dist[0]
        
    x[:, 4:, :, :] = torch.randn_like(x[:, 4:, :, :]) * 0.05
    return x

class CloudExperiment(Experiment):
    ID = "cloud"
    TITLE = "Cloud Target (Noise & Transforms)"
    DESCRIPTION = "Uses randomized cloud-like affine transformed seeds as constraints."
    SEED_TYPE = "noise"

    def __init__(self, base_dir=".", text="COMP", **kwargs):
        super().__init__(base_dir)
        self.text = text
        self.glyph = kwargs.get("glyph", 12)
        self.channel_n = kwargs.get("channel_n", 32)
        self.hidden_n = kwargs.get("hidden_n", 128)
        self.batch = kwargs.get("batch", 8)
        self.pool_size = kwargs.get("pool_size", 1024)
        self.lr = kwargs.get("lr", 2e-3)
        self.damage_n = kwargs.get("damage_n", 0)
        self.ca_min = kwargs.get("ca_min", 64)
        self.ca_max = kwargs.get("ca_max", 96)
        self.log_every = kwargs.get("log_every", 100)
    
    def generate_proposed_targets(self, total_steps: int = 1000):
        tgt = render_word(self.text, self.glyph, FONT_PATH)
        tgt_img = (tgt.transpose(1, 2, 0) * 255).astype(np.uint8)
        img = Image.fromarray(tgt_img)
        img = img.resize((img.width * 8, img.height * 8), getattr(Image, 'Resampling', Image).NEAREST)
        # We can just write out TARGET_00000.png and duplicate
        for i in [0, total_steps//2, total_steps-1]:
            img.save(self.output_dir / f"TARGET_{i:05d}.png")
            
    def save_word_png(self, model, step, device, n_steps=60):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            tgt = render_word(self.text, self.glyph, FONT_PATH)
            torch.manual_seed(42 + step)
            x_init = make_cloud_seed(tgt, self.channel_n, n=1, device=device)
            x_final = model(x_init.clone(), steps=n_steps)
            
        img_init = to_rgba(x_init)[0, :3].cpu().clamp(0, 1).permute(1, 2, 0).numpy()
        img_final = to_rgba(x_final)[0, :3].cpu().clamp(0, 1).permute(1, 2, 0).numpy()
        
        img_combined = np.concatenate([img_init, img_final], axis=1) # [H, 2W, 3]
        im = Image.fromarray((img_combined * 255).astype(np.uint8))
        rez_method = getattr(Image, "Resampling", Image).NEAREST
        im = im.resize((im.width * 8, im.height * 8), rez_method)
        path = str(self.output_dir / f"COMP_{step:05d}.png")
        im.save(path)

    def train(self, total_steps: int = 1000):
        torch.manual_seed(sum(map(ord, self.text)) + 99)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Training on device: {device}", flush=True)
        
        tgt = render_word(self.text, self.glyph)
        target = torch.from_numpy(tgt)[None].repeat(self.batch, 1, 1, 1).to(device)

        model = NCA(self.channel_n, hidden_n=self.hidden_n).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.MultiStepLR(
            opt, milestones=[int(total_steps * 0.8)], gamma=0.1)

        dummy_seed = make_cloud_seed(tgt, self.channel_n, 1, device=device)
        pool = SamplePool(dummy_seed.cpu(), self.pool_size)
        h, w = tgt.shape[1], tgt.shape[2]

        with torch.no_grad():
            for i in range(0, self.pool_size, 32):
                batch_size = min(32, self.pool_size - i)
                pool_clouds = make_cloud_seed(tgt, self.channel_n, batch_size, device="cpu")
                pool.pool[i:i+batch_size] = pool_clouds

        t0 = time.time()
        for step in range(total_steps):
            idx, x = pool.sample(self.batch)
            x = x.to(device)
            
            with torch.no_grad():
                loss_rank = F.mse_loss(to_rgba(x), target, reduction="none") \
                    .mean(dim=(1, 2, 3)).argsort(descending=True)
            x = x[loss_rank]
            
            x[:1] = make_cloud_seed(tgt, self.channel_n, 1, device=device)
            
            if self.damage_n:
                m = damage_mask(self.damage_n, max(h, w), device)[:, :, :h, :w]
                x[-self.damage_n:] *= m

            n_ca = int(torch.randint(self.ca_min, self.ca_max + 1, (1,)))
            x = model(x, steps=n_ca)
            
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

            if step % self.log_every == 0 or step == total_steps - 1:
                print(f"[train_cloud_{self.text}] step {step} loss {loss.item():.5f} "
                      f"({(time.time() - t0):.1f}s)", flush=True)
                try:
                    torch.save(model.state_dict(), str(self.output_dir / 'latest.pth'))
                    self.save_word_png(model, step, device=device)
                    
                    tgt_t = target[0].cpu()
                    a = tgt_t[3:4]
                    rgb = tgt_t[:3]
                    tgt_img_arr = (1.0 - a + rgb).clamp(0,1).permute(1,2,0).numpy()
                    im_tgt = Image.fromarray((tgt_img_arr * 255).astype(np.uint8))
                    im_tgt = im_tgt.resize((im_tgt.width * 8, im_tgt.height * 8), getattr(Image, 'Resampling', Image).NEAREST)
                    im_tgt.save(self.output_dir / f"TARGET_{step:05d}.png")
                except Exception as e:
                    print(f"Failed to save snap: {e}")

        print(f"Final loss for {self.text}: {loss.item():.5f}")
        return model

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--text", default="COMP")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--out", default=None)
    p.add_argument("--snap-dir", default="snaps_cloud")
    a = p.parse_args()
    
    exp = CloudExperiment(base_dir=".", text=a.text)
    exp.output_dir = Path(a.snap_dir)
    exp.log_every = a.log_every
    model = exp.train(total_steps=a.steps)
    if a.out:
        torch.save(model.state_dict(), a.out)
