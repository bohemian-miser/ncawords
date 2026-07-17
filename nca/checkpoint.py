"""Checkpoint save/restore for spot-preemptible training.

Vertex spot jobs can be preempted at any time and restarted from scratch.
Scripts call save_checkpoint() at each log interval and try_resume() before
the training loop; a restarted job then continues from the last saved step
instead of step 0. The sample pool is included so regeneration training
doesn't lose its mature pool states on preemption.
"""
import os
from pathlib import Path

import torch

CKPT_NAME = "ckpt.pth"


def save_checkpoint(snap_dir, step, model, opt, sched=None, pool=None, extra=None):
    if not snap_dir:
        return
    state = {
        "step": step,
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "sched": sched.state_dict() if sched is not None else None,
        "pool": pool.pool if pool is not None else None,
        "extra": extra or {},
    }
    path = Path(snap_dir) / CKPT_NAME
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def try_resume(snap_dir, model, opt, sched=None, pool=None, device="cpu"):
    """Returns (start_step, extra_dict). start_step is 0 if no checkpoint."""
    if not snap_dir:
        return 0, {}
    path = Path(snap_dir) / CKPT_NAME
    if not path.exists():
        return 0, {}
    try:
        state = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        if sched is not None and state.get("sched") is not None:
            sched.load_state_dict(state["sched"])
        if pool is not None and state.get("pool") is not None:
            # Restore to wherever this script keeps its pool (CPU for the
            # scaffold scripts, GPU for organic) — map_location=device would
            # otherwise silently move a CPU pool to CUDA and crash the next
            # pool.commit after a preemption resume.
            pool.pool = state["pool"].to(pool.pool.device)
        start = state["step"] + 1
        print(f"Resumed from checkpoint at step {state['step']} ({path})", flush=True)
        return start, state.get("extra", {})
    except Exception as e:
        print(f"Could not resume from {path}: {e} — starting fresh", flush=True)
        return 0, {}
