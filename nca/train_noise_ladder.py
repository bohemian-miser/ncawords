"""Staged noise-annealing curriculum ("noise ladder").

Each stage trains the NCA to map states at `input_noise` to states at
`target_noise`. Stages run in sequence:

  phase 1 (ladder): 90->80, 80->70, ... 10->0   (decrements of 10)
  phase 2 (jumps):  100->30, 90->60, 80->50, ... 30->0

Stage advance is either a fixed number of steps (--stage-mode fixed) or
adaptive (--stage-mode adaptive): move on once the moving-average loss has
dropped meaningfully below the stage's starting baseline, with a minimum
dwell and a hard cap.
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
from nca.train_diffusion import render_word_9_line, build_state
from nca.checkpoint import save_checkpoint, try_resume
from nca.runmeta import RunMeta, export_run_weights

LADDER = [(round(n / 10, 1), round((n - 1) / 10, 1)) for n in range(9, 0, -1)]
JUMPS = [(1.0, 0.3), (0.9, 0.6), (0.8, 0.5), (0.7, 0.4),
         (0.6, 0.3), (0.5, 0.2), (0.4, 0.1), (0.3, 0.0)]
SCHEDULES = {
    "ladder": LADDER,
    "jumps": JUMPS,
    "ladder+jumps": LADDER + JUMPS,
}


def parse_schedule(spec):
    """Named preset ('ladder+jumps') or custom pairs ('90:80,80:70')."""
    if spec in SCHEDULES:
        return SCHEDULES[spec]
    stages = []
    for pair in spec.split(","):
        a, b = pair.split(":")
        stages.append((float(a) / 100 if float(a) > 1 else float(a),
                       float(b) / 100 if float(b) > 1 else float(b)))
    return stages


def train(text, schedule="ladder+jumps", stage_mode="fixed", stage_steps=100,
          stage_min=30, stage_cap=400, improve_factor=0.7,
          glyph=12, channel_n=16, hidden_n=80, batch=8, lr=2e-3,
          ca_min=48, ca_max=64, log_every=50, snap_dir=None):
    torch.manual_seed(sum(map(ord, text)) + 7)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")

    stages = parse_schedule(schedule)
    max_total = len(stages) * (stage_steps if stage_mode == "fixed" else stage_cap)

    tgt = render_word_9_line(text, glyph)
    tgt_single = torch.from_numpy(tgt).to(device)

    if snap_dir:
        Path(snap_dir).mkdir(parents=True, exist_ok=True)
        Image.fromarray((tgt.transpose(1, 2, 0) * 255).astype(np.uint8)) \
            .save(Path(snap_dir) / "target.png")

    model = NCA(channel_n, hidden_n=hidden_n).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    start_step, ckpt_extra = try_resume(snap_dir, model, opt, device=device)
    stage_idx = ckpt_extra.get("stage_idx", 0)
    steps_in_stage = ckpt_extra.get("steps_in_stage", 0)
    baseline = ckpt_extra.get("baseline", None)

    meta = RunMeta(snap_dir, text, "nca.train_noise_ladder",
                   {"schedule": schedule, "stages": stages,
                    "stage_mode": stage_mode, "stage_steps": stage_steps,
                    "stage_min": stage_min, "stage_cap": stage_cap,
                    "batch": batch, "lr": lr},
                   channel_n, hidden_n, "noise", max_total, device)

    recent = []
    t0 = time.time()
    step = start_step
    loss = None
    while stage_idx < len(stages):
        input_noise, target_noise = stages[stage_idx]

        with torch.no_grad():
            x0 = build_state(tgt_single, input_noise, channel_n, batch, device)
            x_target = build_state(tgt_single, target_noise, channel_n, batch, device)

        n_ca = int(torch.randint(ca_min, ca_max + 1, (1,)))
        x = model(x0, steps=n_ca)
        loss = F.mse_loss(x[:, :4], x_target[:, :4])

        opt.zero_grad()
        loss.backward()
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    p.grad /= (p.grad.norm() + 1e-8)
        opt.step()

        recent.append(loss.item())
        if len(recent) > 10:
            recent.pop(0)
        avg = sum(recent) / len(recent)
        if baseline is None and len(recent) == 10:
            baseline = avg

        steps_in_stage += 1
        advance = False
        if stage_mode == "fixed":
            advance = steps_in_stage >= stage_steps
        else:
            if steps_in_stage >= stage_cap:
                advance = True
            elif steps_in_stage >= stage_min and baseline is not None \
                    and avg < baseline * improve_factor:
                advance = True
        if advance:
            print(f"[stage {stage_idx}] {input_noise:.1f}->{target_noise:.1f} done "
                  f"after {steps_in_stage} steps (avg loss {avg:.5f})", flush=True)
            stage_idx += 1
            steps_in_stage = 0
            baseline = None
            recent.clear()

        if step % log_every == 0 or advance or stage_idx == len(stages):
            print(f"[noise_ladder_{text}] step {step} stage {stage_idx}"
                  f" {input_noise:.1f}->{target_noise:.1f} loss {loss.item():.5f}"
                  f" ({time.time() - t0:.1f}s)", flush=True)
            if snap_dir:
                s = f"{step:05d}"
                Image.fromarray((to_rgb(x)[0].detach().cpu().clamp(0, 1)
                                 .permute(1, 2, 0).numpy() * 255).astype(np.uint8)) \
                    .resize((tgt.shape[2] * 8, tgt.shape[1] * 8), Image.NEAREST) \
                    .save(Path(snap_dir) / f"COMP_{s}.png")
                Image.fromarray((to_rgb(x_target)[0].detach().cpu().clamp(0, 1)
                                 .permute(1, 2, 0).numpy() * 255).astype(np.uint8)) \
                    .resize((tgt.shape[2] * 8, tgt.shape[1] * 8), Image.NEAREST) \
                    .save(Path(snap_dir) / f"TARGET_{s}.png")
                torch.save(model.state_dict(), str(Path(snap_dir) / "latest.pth"))
                save_checkpoint(snap_dir, step, model, opt,
                                extra={"stage_idx": stage_idx,
                                       "steps_in_stage": steps_in_stage,
                                       "baseline": baseline})
                meta.log(step, loss.item(), stage_idx=stage_idx,
                         stage=f"{input_noise:.1f}->{target_noise:.1f}",
                         steps_in_stage=steps_in_stage)
                export_run_weights(model, snap_dir, text, glyph,
                                   grid_w=tgt.shape[2], grid_h=tgt.shape[1],
                                   seed_type="noise")
        step += 1

    final = f"{loss.item():.5f}" if loss is not None else "n/a (already complete)"
    print(f"Noise ladder complete for {text}: {step} total steps, "
          f"{len(stages)} stages, final loss {final}")
    return model


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--text", required=True)
    p.add_argument("--schedule", default="ladder+jumps",
                   help="'ladder', 'jumps', 'ladder+jumps', or custom '90:80,80:70'")
    p.add_argument("--stage-mode", default="fixed", choices=["fixed", "adaptive"])
    p.add_argument("--stage-steps", type=int, default=100)
    p.add_argument("--stage-min", type=int, default=30)
    p.add_argument("--stage-cap", type=int, default=400)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--snap-dir", default=None)
    a = p.parse_args()

    train(a.text, schedule=a.schedule, stage_mode=a.stage_mode,
          stage_steps=a.stage_steps, stage_min=a.stage_min,
          stage_cap=a.stage_cap, log_every=a.log_every, snap_dir=a.snap_dir)
