import argparse
import io
import time
from pathlib import Path
import json

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from nca.model import NCA, to_rgba
from nca.train_web_hidden import damage_mask_rect
from nca.checkpoint import save_checkpoint, try_resume
from nca.runmeta import RunMeta, export_run_weights
from nca.rollout import fester
from nca.train_emoji_vanilla import emoji_rgba

def _mixed_corrupt(x, model, step, steps, device, ceil_noise, max_fester):
    B, C, h, w = x.shape
    # Scale corruption intensity with training progress (starting after 25% of training)
    frac = max(0.0, (step - steps * 0.25) / (steps * 0.75))
    
    current_ceil = 0.05 + ceil_noise * frac
    max_dmg = 1 + int(4 * frac)
    max_fest = int(5 + max_fester * frac)

    # State Noise
    a = (torch.rand(B, 1, 1, 1, device=device) ** 1.5) * current_ceil
    x = (1 - a) * x + a * torch.rand_like(x)

    # Damage
    for b in range(B):
        if torch.rand(1).item() < 0.6:
            for _ in range(int(torch.randint(1, max_dmg + 1, (1,)))):
                bh = int(torch.randint(3, max(4, h // 2), (1,)))
                bw = int(torch.randint(3, max(4, w // 2), (1,)))
                y0 = int(torch.randint(0, max(1, h - bh), (1,)))
                x0 = int(torch.randint(0, max(1, w - bw), (1,)))
                x[b, :, y0:y0 + bh, x0:x0 + bw] = 0.0

    # Fester
    fest_n = 0
    if max_fest > 5 and torch.rand(1).item() < 0.5:
        fest_n = int(torch.randint(5, max_fest + 1, (1,)).item())
        sub = x.clone()
        with torch.no_grad():
            for t in range(fest_n):
                sub = model(sub, steps=1)
                # occasional damage during festering
                if t and t % 30 == 0 and torch.rand(1).item() < 0.5:
                    sub = sub * damage_mask_rect(B, h, w, device)
        x = x.clone()
        # Only apply festered subset to half the batch
        idx = torch.randperm(B, device=device)[:B//2]
        x[idx] = sub[idx].detach()
        
    return x, {"noise": round(float(a.mean()), 3), "fest": fest_n, "dmg": max_dmg}

def train(emoji="1f642", steps=8000, channel_n=16, hidden_n=96,
          batch=16, pool_size=256, lr=2e-3, 
          baseline_damage_p=0.4, ceil_noise=0.9, max_fester=200, apply_mix=True,
          rng_seed=0, snap_dir=None):
    
    label = f"grid_{emoji}_h{hidden_n}_n{ceil_noise}_f{max_fester}"
    torch.manual_seed( sum(map(ord, emoji)) + rng_seed )
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device {device}, starting {label}")

    tgt_np = emoji_rgba(emoji)
    _, h, w = tgt_np.shape
    target = torch.from_numpy(tgt_np)[None].repeat(batch, 1, 1, 1).to(device)

    model = NCA(channel_n, fire_rate=0.5, hidden_n=hidden_n).to(device)
    try:
        import torch._dynamo
        torch._dynamo.config.suppress_errors = True
        model = torch.compile(model)
    except:
        pass
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[int(steps * 0.85)], gamma=0.1)

    seed = torch.zeros(1, channel_n, h, w, device=device)
    seed[:, 3:, h // 2, w // 2] = 1.0
    pool = seed.repeat(pool_size, 1, 1, 1)

    start_step, _ = try_resume(snap_dir, model, opt, sched, device=device)

    if snap_dir:
        Path(snap_dir).mkdir(parents=True, exist_ok=True)
        img = (1 - tgt_np[3:4] + tgt_np[:3]).clip(0, 1).transpose(1, 2, 0)
        Image.fromarray((img * 255).astype(np.uint8)) \
            .resize((w * 6, h * 6), Image.NEAREST).save(Path(snap_dir) / "target.png")
            
    meta = RunMeta(snap_dir, label.upper(), "nca.train_emoji_grid",
                   {"emoji": emoji, "steps": steps, "batch": batch, "lr": lr,
                    "hidden_n": hidden_n, "ceil_noise": ceil_noise, "max_fester": max_fester},
                   channel_n, hidden_n, "single", steps, device, tags=["emoji", "experiment"])

    t0 = time.time()
    for step in range(start_step, steps):
        idx = torch.randperm(pool_size, device=device)[:batch]
        x = pool[idx]
        with torch.no_grad():
            rank = F.mse_loss(to_rgba(x), target, reduction="none").mean(dim=(1, 2, 3)).argsort(descending=True)
        x = x[rank]; idx = idx[rank]
        x[:1] = seed
        
        # Apply baseline damage occasionally 
        if torch.rand(1).item() < baseline_damage_p:
            nd = int(torch.randint(1, 4, (1,)))
            m = damage_mask_rect(nd, h, w, device)
            x[-nd:] = x[-nd:] * m

        # Mix of noise + fester + damage after initial clean warm-up
        if apply_mix and step > steps * 0.2:
            x, _info = _mixed_corrupt(x, model, step, steps, device, ceil_noise, max_fester)
        
        x_start = x[-1:].detach().clone()
        
        n_ca = int(torch.randint(64, 96 + 1, (1,)))
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
        with torch.no_grad():
            pool[idx] = x.detach()

        if step % 200 == 0 or step == steps - 1:
            print(f"[{label}] step {step} loss {loss.item():.5f} ({time.time() - t0:.1f}s)", flush=True)
            if snap_dir:
                s = f"{step:05d}"
                for tag, t in [("COMP", to_rgba(x)[-1]), ("START", to_rgba(x_start)[0])]:
                    img_t = t.detach().cpu().clamp(0, 1)
                    vis = (1 - img_t[3:4] + img_t[:3]).clamp(0, 1).permute(1, 2, 0).numpy()
                    Image.fromarray((vis * 255).astype(np.uint8)) \
                        .resize((w * 6, h * 6), Image.NEAREST) \
                        .save(Path(snap_dir) / f"{tag}_{s}.png")
                torch.save(model.state_dict(), str(Path(snap_dir) / "latest.pth"))
                meta.log(step, loss.item(), ca_steps=n_ca)
                export_run_weights(model, snap_dir, label, 12, grid_w=w, grid_h=h)
        if snap_dir and (step % 200 == 0 or step == steps - 1):
            save_checkpoint(snap_dir, step, model, opt, sched)

    final_loss = loss.item()
    print(f"Final loss for {label}: {final_loss:.5f}")
    if snap_dir:
        # Extra save to easily pull the final loss by parser script later
        with open(Path(snap_dir) / "grid_result.json", "w") as f:
            json.dump({"loss": final_loss, "hidden_n": hidden_n, "ceil_noise": ceil_noise, "max_fester": max_fester}, f)
            
    return model

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--emoji", default="1f600")
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--hidden-n", type=int, default=96)
    p.add_argument("--ceil-noise", type=float, default=0.9)
    p.add_argument("--max-fester", type=int, default=200)
    p.add_argument("--snap-dir", default=None)
    a = p.parse_args()
    
    train(emoji=a.emoji, steps=a.steps, hidden_n=a.hidden_n, 
          ceil_noise=a.ceil_noise, max_fester=a.max_fester, 
          snap_dir=a.snap_dir)
