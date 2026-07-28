"""Adaptive CA rollout: run in chunks until the loss stops dropping.

Replaces the classic fixed random horizon (n_ca ~ U[min,max]) with
compute-on-demand: keep stepping while each chunk still improves the
batch RGBA loss by a relative eps; stop otherwise (or at the cap).
Gradients flow through every executed chunk.
"""
import torch
import torch.nn.functional as F

from nca.model import to_rgba


def adaptive_rollout(model, x, target, chunk=8, max_chunks=10, eps=0.02):
    """Returns (x_after, steps_used)."""
    prev = None
    used = 0
    for _ in range(max_chunks):
        x = model(x, steps=chunk)
        used += chunk
        cur = float(F.mse_loss(to_rgba(x), target).detach())
        if prev is not None and (prev - cur) < eps * prev:
            break
        prev = cur
    return x, used


def fester(model, x, damage_fn=None, min_steps=150, max_steps=500):
    """Long no-grad pre-roll: optionally damage, then let the model run far
    past its training horizon and return whatever state it lands in
    (detached). Training from these states toward the original target
    teaches recovery from long-horizon drift ('eaten by angry flames').
    """
    with torch.no_grad():
        if damage_fn is not None and torch.rand(1).item() < 0.5:
            x = damage_fn(x)
        n = int(torch.randint(min_steps, max_steps + 1, (1,)))
        x = model(x, steps=n)
    return x.detach()
