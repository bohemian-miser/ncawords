import argparse
import time
from pathlib import Path
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from nca.model import NCA, to_rgba, make_seed
from nca.train import damage_mask, SamplePool
from nca.experiment import Experiment

class GuidedExperiment(Experiment):
    ID = "guided"
    TITLE = "Guided Target Changes"
    DESCRIPTION = "Loads a pre-generated sequence of external constraints / targets."
    SEED_TYPE = "single"

    def __init__(self, base_dir=".", text="COMP", **kwargs):
        super().__init__(base_dir)
        self.text = text
        self.channel_n = kwargs.get("channel_n", 32)
        self.hidden_n = kwargs.get("hidden_n", 128)
        self.batch = kwargs.get("batch", 8)
        self.pool_size = kwargs.get("pool_size", 1024)
        self.lr = kwargs.get("lr", 2e-3)
        self.damage_n = kwargs.get("damage_n", 0)
        self.ca_min = kwargs.get("ca_min", 64)
        self.ca_max = kwargs.get("ca_max", 96)
        self.log_every = kwargs.get("log_every", 100)

    def generate_proposed_targets(self, total_steps: int = 4000):
        # The targets are already generated in 'snaps_proposed_targets' externally, 
        # or we just load them here for preview.
        # But to be compliant, we can just save the ones we expect.
        preview_dir = Path(self.output_dir)
        preview_dir.mkdir(parents=True, exist_ok=True)
        # Assuming gen_guided_targets.py was used to create snaps_proposed_targets
        # We can just copy them to our output_dir for UI preview if they exist.
        for step in [0, total_steps//2, total_steps-1]:
            idx = min(step, 4000 - 1)
            path = f"snaps_proposed_targets/TARGET_{idx:05d}.png"
            if not os.path.exists(path):
                path = f"snaps_proposed_targets/TARGET_03999.png"
            if os.path.exists(path):
                img = Image.open(path).convert("RGBA")
                img.save(preview_dir / f"TARGET_{step:05d}.png")

    def load_target(self, step, device="cpu"):
        idx = min(step, 4000 - 1)
        path = f"snaps_proposed_targets/TARGET_{idx:05d}.png"
        if not os.path.exists(path):
            path = f"snaps_proposed_targets/TARGET_03999.png"
        if not os.path.exists(path):
            # dummy target if not found
            return torch.zeros((1, 4, 40, 40), device=device)
        img = Image.open(path).convert("RGBA")
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr[..., :3] *= arr[..., 3:]
        tensor = torch.from_numpy(arr.transpose(2, 0, 1)).to(device)
        return tensor.unsqueeze(0)

    def save_word_png(self, model, step, device, n_steps=60):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            x_init = make_seed(40, self.channel_n).to(device)
            x_final = model(x_init, steps=n_steps)
            
        img_final = to_rgba(x_final)[0, :3].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
        im = Image.fromarray((img_final * 255).astype(np.uint8))
        rez_method = getattr(Image, "Resampling", Image).NEAREST
        im = im.resize((im.width * 8, im.height * 8), rez_method)
        im.save(self.output_dir / f"COMP_{step:05d}.png")

    def train(self, total_steps: int = 4000):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Training on device: {device}", flush=True)

        model = NCA(self.channel_n, hidden_n=self.hidden_n).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.MultiStepLR(
            opt, milestones=[int(total_steps * 0.8)], gamma=0.1)

        seed = make_seed(40, self.channel_n)
        pool = SamplePool(seed, self.pool_size)
        h, w = 40, 40

        t0 = time.time()
        for step in range(total_steps):
            target = self.load_target(step, device).repeat(self.batch, 1, 1, 1)
            
            idx, x = pool.sample(self.batch)
            x = x.to(device)
            
            with torch.no_grad():
                loss_rank = F.mse_loss(to_rgba(x), target, reduction="none") \
                    .mean(dim=(1, 2, 3)).argsort(descending=True)
            x = x[loss_rank]
            
            x[:1] = make_seed(40, self.channel_n).to(device)
            
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
                print(f"[train_guided] step {step} loss {loss.item():.5f} "
                      f"({(time.time() - t0):.1f}s)", flush=True)
                try:
                    torch.save(model.state_dict(), str(self.output_dir / 'latest.pth'))
                    self.save_word_png(model, step, device=device)
                    # For visualization logic: UI wants TARGET_{step}.png too
                    tgt_path_out = self.output_dir / f"TARGET_{step:05d}.png"
                    if not tgt_path_out.exists():
                         src_idx = min(step, 4000-1)
                         src_path = f"snaps_proposed_targets/TARGET_{src_idx:05d}.png"
                         if os.path.exists(src_path):
                             import shutil
                             shutil.copy(src_path, tgt_path_out)
                except Exception as e:
                    print(f"Failed to save snap: {e}")

        print(f"Final loss: {loss.item():.5f}")
        return model

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--out", default="weights_guided.pth")
    p.add_argument("--snap-dir", default="snaps_guided")
    a = p.parse_args()
    
    exp = GuidedExperiment(base_dir=".", text="COMP")
    exp.output_dir = Path(a.snap_dir) # override for compatibility
    exp.log_every = a.log_every
    exp.train(total_steps=a.steps)
