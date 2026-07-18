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
