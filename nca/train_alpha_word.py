"""COMP growth supervised on activations only.

Target: word + 3-line scaffold, but the loss sees ONLY the alpha channel
— where something should be alive. RGB and hidden channels are entirely
unconstrained: the model chooses its own palette and machinery. Snapshots
save both the alpha field (what's supervised) and the RGB view (what it
invented).

Defaults fold in the HP star findings: lr 1e-3, hidden 128.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from nca.model import NCA, to_rgba
from nca.train_staged import render_word_3_line, make_seed
from nca.train_web_hidden import damage_mask_rect
from nca.checkpoint import save_checkpoint, try_resume
from nca.runmeta import RunMeta, export_run_weights
from nca.rollout import fester


def train(text="COMP", steps=12000, glyph=12, channel_n=16, hidden_n=128,
          batch=16, pool_size=256, lr=1e-3, ca_min=64, ca_max=96,
          seed_type="single", damage_p=0.3, fester_p=0.25,
          log_every=200, ckpt_every=500, snap_dir=None):
    torch.manual_seed(sum(map(ord, text)) + 11)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}, seed_type {seed_type}")

    tgt_np = render_word_3_line(text, glyph)
    _, h, w = tgt_np.shape
    tgt_a = torch.from_numpy(tgt_np[3:4])[None] \
        .repeat(batch, 1, 1, 1).to(device)   # alpha only — all we supervise

    if snap_dir:
        Path(snap_dir).mkdir(parents=True, exist_ok=True)
        Image.fromarray(((1 - tgt_np[3]) * 255).astype(np.uint8)) \
            .resize((w * 8, h * 8), Image.NEAREST).save(Path(snap_dir) / "target.png")

    model = NCA(channel_n, hidden_n=hidden_n).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=[int(steps * 0.85)], gamma=0.1)

    if seed_type == "noise":
        seed = torch.rand(1, channel_n, h, w, device=device)
    else:
        seed = make_seed(tgt_np, channel_n).to(device)
    pool = seed.repeat(pool_size, 1, 1, 1)

    start_step, _ = try_resume(snap_dir, model, opt, sched, device=device)

    meta = RunMeta(snap_dir, text, "nca.train_alpha_word",
                   {"steps": steps, "batch": batch, "lr": lr,
                    "hidden_n": hidden_n, "seed_type": seed_type,
                    "damage_p": damage_p, "fester_p": fester_p},
                   channel_n, hidden_n, seed_type, steps, device,
                   tags=["alpha-only", seed_type])

    t0 = time.time()
    for step in range(start_step, steps):
        idx = torch.randperm(pool_size, device=device)[:batch]
        x = pool[idx]
        with torch.no_grad():
            rank = F.mse_loss(x[:, 3:4], tgt_a, reduction="none") \
                .mean(dim=(1, 2, 3)).argsort(descending=True)
        x = x[rank]; idx = idx[rank]
        if seed_type == "noise":
            x[:1] = torch.rand_like(x[:1])
        else:
            x[:1] = seed
        if torch.rand(1).item() < damage_p:
            m = damage_mask_rect(2, h, w, device)
            x[-2:] = x[-2:] * m
        if fester_p > 0 and torch.rand(1).item() < fester_p:
            x = fester(model, x,
                       damage_fn=lambda z: z * damage_mask_rect(2, h, w, device))

        n_ca = int(torch.randint(ca_min, ca_max + 1, (1,)))
        x = model(x, steps=n_ca)
        loss = F.mse_loss(x[:, 3:4], tgt_a)

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

        if step % log_every == 0 or step == steps - 1:
            print(f"[alpha_{text}-{seed_type}] step {step} loss {loss.item():.5f} "
                  f"({time.time() - t0:.1f}s)", flush=True)
            if snap_dir:
                s = f"{step:05d}"
                img = to_rgba(x)[0].detach().cpu().clamp(0, 1)
                vis = (1 - img[3:4] + img[:3]).clamp(0, 1).permute(1, 2, 0).numpy()
                Image.fromarray((vis * 255).astype(np.uint8)) \
                    .resize((w * 8, h * 8), Image.NEAREST) \
                    .save(Path(snap_dir) / f"COMP_{s}.png")   # invented palette
                a = img[3, :, :].numpy()
                Image.fromarray(((1 - a) * 255).astype(np.uint8)) \
                    .resize((w * 8, h * 8), Image.NEAREST) \
                    .save(Path(snap_dir) / f"ALPHA_{s}.png")  # what's supervised
                torch.save(model.state_dict(), str(Path(snap_dir) / "latest.pth"))
                meta.log(step, loss.item())
                export_run_weights(model, snap_dir, text, glyph,
                                   grid_w=w + 20, grid_h=h + 10,
                                   seed_type="noise" if seed_type == "noise" else None)
        if snap_dir and (step % ckpt_every == 0 or step == steps - 1):
            save_checkpoint(snap_dir, step, model, opt, sched)

    print(f"Final loss: {loss.item():.5f}")
    return model


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--text", default="COMP")
    p.add_argument("--steps", type=int, default=12000)
    p.add_argument("--seed-type", default="single", choices=["single", "noise"])
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--snap-dir", default=None)
    a = p.parse_args()
    train(a.text, steps=a.steps, seed_type=a.seed_type,
          log_every=a.log_every, snap_dir=a.snap_dir)
