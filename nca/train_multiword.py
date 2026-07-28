"""One model, several words: the seed's code channels choose what grows.

Standard pool/growth training (ladder_seed recipe) with N words on a
shared canvas. Each word's seed sets a one-hot code in the last channels
at the seed cell; the local rules must propagate and honor that choice.
Damage training included so each word's pattern is also regenerative.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from nca.model import NCA, to_rgba
from nca.train_web_hidden import render_word_9_line, damage_mask_rect
from nca.checkpoint import save_checkpoint, try_resume
from nca.runmeta import RunMeta, export_run_weights
from nca.rollout import adaptive_rollout


def render_targets(words, glyph=12, strand_alpha=64):
    """Render each word centered on a shared canvas sized for the longest.
    strand_alpha > 0 keeps the 9-line scaffold — the multi-letter
    coordination mechanism the basics sweep proved necessary."""
    rendered = [render_word_9_line(w, glyph, char_alpha=255,
                                   strand_alpha=strand_alpha)
                for w in words]
    H = max(r.shape[1] for r in rendered)
    W = max(r.shape[2] for r in rendered)
    out = []
    for r in rendered:
        pad = np.zeros((4, H, W), np.float32)
        y0 = (H - r.shape[1]) // 2
        x0 = (W - r.shape[2]) // 2
        pad[:, y0:y0 + r.shape[1], x0:x0 + r.shape[2]] = r
        out.append(pad)
    return out, H, W


def make_code_seed(tgt, channel_n, word_idx, n_words):
    _, h, w = tgt.shape
    x = torch.zeros(1, channel_n, h, w)
    ys, xs = np.where(tgt[3] > 0.5)
    cy, cx = h // 2, w // 2
    if len(ys):
        i = np.argmin((ys - cy) ** 2 + (xs - cx) ** 2)
        cy, cx = ys[i], xs[i]
    x[:, 3:, cy, cx] = 1.0
    # one-hot word code in the last n_words channels, at the seed cell
    x[:, channel_n - n_words:, cy, cx] = 0.0
    x[:, channel_n - n_words + word_idx, cy, cx] = 1.0
    return x


def train(words="COMP,NCA", steps=16000, glyph=12, channel_n=16, hidden_n=96,
          batch=24, pool_per_word=128, lr=2e-3, ca_min=64, ca_max=96,
          damage_p=0.3, adaptive=False, log_every=200, ckpt_every=500,
          snap_dir=None):
    word_list = [w for w in words.split(",") if w]
    n = len(word_list)
    torch.manual_seed(sum(map(ord, words)) + 77)
    rng = np.random.default_rng(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}, words: {word_list}")

    tgts_np, h, w = render_targets(word_list, glyph)
    targets = [torch.from_numpy(t)[None].repeat(batch, 1, 1, 1).to(device)
               for t in tgts_np]
    seeds = [make_code_seed(tgts_np[i], channel_n, i, n).to(device)
             for i in range(n)]

    if snap_dir:
        Path(snap_dir).mkdir(parents=True, exist_ok=True)
        strip = np.concatenate(tgts_np, axis=2)
        vis = (1 - strip[3] + strip[:3]).clip(0, 1).transpose(1, 2, 0)
        Image.fromarray((vis * 255).astype(np.uint8)) \
            .resize((vis.shape[1] * 6, vis.shape[0] * 6), Image.NEAREST) \
            .save(Path(snap_dir) / "target.png")

    model = NCA(channel_n, hidden_n=hidden_n).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=[int(steps * 0.85)], gamma=0.1)

    pools = [seeds[i].repeat(pool_per_word, 1, 1, 1) for i in range(n)]

    start_step, _ = try_resume(snap_dir, model, opt, sched, device=device)

    meta = RunMeta(snap_dir, words, "nca.train_multiword",
                   {"steps": steps, "words": word_list, "batch": batch,
                    "lr": lr, "damage_p": damage_p, "adaptive": adaptive},
                   channel_n, hidden_n, "coded", steps, device,
                   tags=["multiword"] + (["adaptive"] if adaptive else []))

    t0 = time.time()
    for step in range(start_step, steps):
        wi = step % n
        target, seed, pool = targets[wi], seeds[wi], pools[wi]

        idx = torch.randperm(pool_per_word, device=device)[:batch]
        x = pool[idx]
        with torch.no_grad():
            rank = F.mse_loss(to_rgba(x), target, reduction="none") \
                .mean(dim=(1, 2, 3)).argsort(descending=True)
        x = x[rank]; idx = idx[rank]
        x[:1] = seed
        if torch.rand(1).item() < damage_p:
            m = damage_mask_rect(2, h, w, device)
            x[-2:] = x[-2:] * m

        x_start = x[-1:].detach().clone()
        if adaptive:
            x, _u = adaptive_rollout(model, x, target, chunk=12, max_chunks=8)
        else:
            n_ca = int(torch.randint(ca_min, ca_max + 1, (1,)))
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

        if step % log_every == 0 or step == steps - 1:
            print(f"[multiword] step {step} word {word_list[wi]} "
                  f"loss {loss.item():.5f} ({time.time() - t0:.1f}s)", flush=True)
            if snap_dir:
                s = f"{step:05d}"
                for tag, t in [("COMP", to_rgba(x)[-1]), ("START", to_rgba(x_start)[0])]:
                    img = t.detach().cpu().clamp(0, 1)
                    vis = (1 - img[3:4] + img[:3]).clamp(0, 1).permute(1, 2, 0).numpy()
                    Image.fromarray((vis * 255).astype(np.uint8)) \
                        .resize((w * 6, h * 6), Image.NEAREST) \
                        .save(Path(snap_dir) / f"{tag}_{s}.png")
                torch.save(model.state_dict(), str(Path(snap_dir) / "latest.pth"))
                meta.log(step, loss.item(), word=word_list[wi])
                export_run_weights(model, snap_dir, word_list[0], glyph,
                                   grid_w=w + 20, grid_h=h + 10)
        if snap_dir and (step % ckpt_every == 0 or step == steps - 1):
            save_checkpoint(snap_dir, step, model, opt, sched)

    print(f"Final loss: {loss.item():.5f}")
    return model


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--words", default="COMP,NCA")
    p.add_argument("--steps", type=int, default=16000)
    p.add_argument("--adaptive", action="store_true")
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--snap-dir", default=None)
    a = p.parse_args()
    train(a.words, steps=a.steps, adaptive=a.adaptive,
          log_every=a.log_every, snap_dir=a.snap_dir)
