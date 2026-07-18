"""Train an NCA to recreate Physarum (slime mold) dynamics from snapshots.

A classic agent-based Physarum simulation (Jones 2010: sense-ahead, steer
toward trail, deposit, diffuse, evaporate) generates a sequence of trail-
field frames. The NCA never sees the agents — it is trained state -> next
-state on the field snapshots over a persistent pool of trajectories, so
it must learn dynamics that reproduce vein-network formation with only
local perception. Trail intensity lives in the alpha channel; RGB and the
hidden channels are unconstrained (the model can 'do a lot of different
stuff' — only the presence field is scored).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from nca.model import NCA, to_rgb
from nca.checkpoint import save_checkpoint, try_resume
from nca.runmeta import RunMeta, export_run_weights

CANVAS = 72


def letter_food(text, glyph=26):
    """Food field from letter shapes (blurred so agents can smell it)."""
    from PIL import ImageFont, ImageDraw, Image as PILImage
    from nca.train import FONT_PATH
    font = ImageFont.truetype(FONT_PATH, glyph)
    img = PILImage.new("L", (CANVAS, CANVAS), 0)
    draw = ImageDraw.Draw(img)
    l, t, r, b = draw.textbbox((0, 0), text, font=font)
    draw.text(((CANVAS - (r - l)) / 2 - l, (CANVAS - (b - t)) / 2 - t),
              text, font=font, fill=255)
    f = np.asarray(img, np.float32) / 255.0
    for _ in range(6):   # diffuse so the gradient reaches out
        f = (f + np.roll(f, 1, 0) + np.roll(f, -1, 0)
             + np.roll(f, 1, 1) + np.roll(f, -1, 1)) / 5.0
    return (f / max(f.max(), 1e-6)).astype(np.float32)


def physarum_frames(K=120, n_agents=4000, substeps=3, sensor_d=5.0,
                    sensor_a=np.pi / 8, turn=np.pi / 6, speed=1.0,
                    deposit=0.35, evap=0.12, rng_seed=0,
                    food=None, food_w=0.0):
    """Agent-based Physarum on a torus; returns trail frames [K, H, W] in 0..1.

    With a food field and food_w > 0, agents steer toward food as well as
    trail — the network condenses onto the food (letters)."""
    rng = np.random.default_rng(rng_seed)
    H = W = CANVAS
    pos = rng.random((n_agents, 2)) * (H, W)
    ang = rng.random(n_agents) * 2 * np.pi
    trail = np.zeros((H, W), np.float32)

    def sense(offset):
        sy = ((pos[:, 0] + np.sin(ang + offset) * sensor_d) % H).astype(int)
        sx = ((pos[:, 1] + np.cos(ang + offset) * sensor_d) % W).astype(int)
        s = trail[sy, sx]
        if food is not None and food_w > 0:
            # food is both smellable and (below) a constant trail source
            s = s + food_w * 2.0 * food[sy, sx]
        return s

    frames = []
    for _ in range(K):
        for _ in range(substeps):
            f, l, r = sense(0), sense(sensor_a), sense(-sensor_a)
            steer = np.where((f >= l) & (f >= r), 0.0,
                             np.where(l > r, turn, -turn))
            ang = ang + steer + rng.normal(0, 0.05, n_agents)
            reorient = rng.random(n_agents) < 0.02
            ang = np.where(reorient, rng.random(n_agents) * 2 * np.pi, ang)
            pos[:, 0] = (pos[:, 0] + np.sin(ang) * speed) % H
            pos[:, 1] = (pos[:, 1] + np.cos(ang) * speed) % W
            np.add.at(trail, (pos[:, 0].astype(int), pos[:, 1].astype(int)), deposit)
            t = trail
            t = (t
                 + np.roll(t, 1, 0) + np.roll(t, -1, 0)
                 + np.roll(t, 1, 1) + np.roll(t, -1, 1)
                 + np.roll(np.roll(t, 1, 0), 1, 1) + np.roll(np.roll(t, 1, 0), -1, 1)
                 + np.roll(np.roll(t, -1, 0), 1, 1) + np.roll(np.roll(t, -1, 0), -1, 1)) / 9.0
            trail = (t * (1.0 - evap)).astype(np.float32)
            if food is not None and food_w > 0:
                trail = trail + food_w * deposit * food
        frames.append(np.clip(trail, 0, 1).copy())
    return np.stack(frames)  # [K, H, W]


def frame_png(a, path, upscale=6):
    img = (1.0 - np.clip(a, 0, 1))  # trail dark on white
    Image.fromarray((img * 255).astype(np.uint8)) \
        .resize((CANVAS * upscale, CANVAS * upscale), Image.NEAREST).save(path)


def make_state_from_frame(frame, channel_n, device, hidden_noise=0.1, rng=None,
                          food=None):
    """Visible alpha = trail field; hidden channels get faint noise so the
    model can break symmetry the way the invisible agents did. If a food
    field is given it occupies channel 4 as fixed conditioning — at rollout
    you can paint any shape there and the learned dynamics chase it."""
    x = torch.zeros(channel_n, CANVAS, CANVAS, device=device)
    x[3] = torch.from_numpy(frame).to(device)
    h0 = 4
    if food is not None:
        x[4] = torch.from_numpy(food).to(device)
        h0 = 5
    if rng is None:
        x[h0:] = torch.rand(channel_n - h0, CANVAS, CANVAS, device=device) * hidden_noise
    else:
        noise = rng.random((channel_n - h0, CANVAS, CANVAS)).astype(np.float32)
        x[h0:] = torch.from_numpy(noise).to(device) * hidden_noise
    return x


def train(text="SLIME", steps=8000, K=120, channel_n=16, hidden_n=80,
          batch=8, pool_size=64, lr=2e-3, ca_min=8, ca_max=16,
          log_every=100, snap_dir=None, rng_seed=0, sim_kwargs=None,
          food_text=None, food_w=0.0):
    torch.manual_seed(1337)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")

    print("Running Physarum simulation to generate training frames...")
    food = letter_food(food_text) if food_text else None
    frames_np = physarum_frames(K=K, rng_seed=rng_seed, food=food,
                                food_w=food_w, **(sim_kwargs or {}))
    frames = torch.from_numpy(frames_np).to(device)  # [K, H, W]
    print(f"Simulated {K} frames, mean trail {frames_np.mean():.3f}")

    if snap_dir:
        Path(snap_dir).mkdir(parents=True, exist_ok=True)
        frame_png(frames_np[-1], Path(snap_dir) / "target.png")

    model = NCA(channel_n, hidden_n=hidden_n).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=[int(steps * 0.8)], gamma=0.1)

    pool = torch.stack([make_state_from_frame(frames_np[0], channel_n, device,
                                              food=food)
                        for _ in range(pool_size)])
    pool_k = torch.zeros(pool_size, dtype=torch.long)

    start_step, _ = try_resume(snap_dir, model, opt, sched, device=device)

    meta = RunMeta(snap_dir, text, "nca.train_slime",
                   {"steps": steps, "K": K, "batch": batch, "lr": lr,
                    "rng_seed": rng_seed, "pool_size": pool_size,
                    "food_text": food_text, "food_w": food_w},
                   channel_n, hidden_n, "noise", steps, device)

    t0 = time.time()
    for step in range(start_step, steps):
        idx = torch.randperm(pool_size)[:batch]
        x = pool[idx].clone()
        ks = pool_k[idx]

        tgt_a = torch.stack([frames[min(int(k) + 1, K - 1)] for k in ks]).unsqueeze(1)

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
            for j, i in enumerate(idx):
                if ks[j] + 1 >= K - 1:
                    pool[i] = make_state_from_frame(frames_np[0], channel_n, device,
                                                    food=food)
                    pool_k[i] = 0
                else:
                    pool_k[i] = ks[j] + 1

        if step % log_every == 0 or step == steps - 1:
            print(f"[slime_{text}] step {step} loss {loss.item():.5f} "
                  f"k_max {int(pool_k.max())} ({time.time() - t0:.1f}s)", flush=True)
            if snap_dir:
                s = f"{step:05d}"
                j = int(torch.argmax(ks).item())
                a_out = x[j, 3].detach().cpu().clamp(0, 1).numpy()
                frame_png(a_out, Path(snap_dir) / f"COMP_{s}.png")
                frame_png(frames_np[min(int(ks[j]) + 1, K - 1)],
                          Path(snap_dir) / f"TARGET_{s}.png")
                torch.save(model.state_dict(), str(Path(snap_dir) / "latest.pth"))
                save_checkpoint(snap_dir, step, model, opt, sched)
                meta.log(step, loss.item())
                export_run_weights(model, snap_dir, text,
                                   grid_w=CANVAS, grid_h=CANVAS, seed_type="noise")

    print(f"Final loss for {text} (slime): {loss.item():.5f}")
    return model


def preview_targets(out_dir, K=120, rng_seed=0, every=8):
    frames = physarum_frames(K=K, rng_seed=rng_seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for k in range(0, K, every):
        frame_png(frames[k], out / f"TARGET_{k:05d}.png")
    frame_png(frames[-1], out / f"TARGET_{K-1:05d}.png")
    print(f"Wrote {len(range(0, K, every)) + 1} proposed frames to {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--text", default="SLIME")
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--frames", type=int, default=120)
    p.add_argument("--rng-seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--snap-dir", default=None)
    p.add_argument("--sensor-d", type=float, default=5.0)
    p.add_argument("--evap", type=float, default=0.12)
    p.add_argument("--agents", type=int, default=4000)
    p.add_argument("--substeps", type=int, default=3)
    p.add_argument("--food-text", default=None)
    p.add_argument("--food-w", type=float, default=0.0)
    p.add_argument("--preview", action="store_true")
    a = p.parse_args()

    sim = {"sensor_d": a.sensor_d, "evap": a.evap,
           "n_agents": a.agents, "substeps": a.substeps}
    if a.preview:
        preview_targets(a.snap_dir or "snaps_slime_preview",
                        K=a.frames, rng_seed=a.rng_seed)
    else:
        train(a.text, steps=a.steps, K=a.frames, rng_seed=a.rng_seed,
              log_every=a.log_every, snap_dir=a.snap_dir, sim_kwargs=sim,
              food_text=a.food_text, food_w=a.food_w)
