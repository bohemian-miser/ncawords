"""Generate a deterministic golden rollout for cross-checking the JS engine.

Runs the CA from the seed with fire_rate=1.0 (no stochastic mask) for N
steps and dumps the full final state grid. A correct JS implementation
must reproduce it to ~1e-4.

Usage: python -m nca.make_golden weights/0041.json site/test/golden_0041.json
"""

import json
import sys
from pathlib import Path

import torch

from nca.model import make_seed
from nca.ocr_eval import load_model


def main(wpath, opath, steps=30):
    model, d = load_model(wpath)
    with torch.no_grad():
        x = make_seed(d["grid"], d["channel_n"])
        x = model(x, fire_rate=1.0, steps=steps)
    out = {
        "weights": str(wpath), "steps": steps, "fire_rate": 1.0,
        "grid": d["grid"], "channel_n": d["channel_n"],
        # [C, H, W] flattened C-major
        "state": [round(v, 6) for v in x[0].flatten().tolist()],
    }
    Path(opath).parent.mkdir(parents=True, exist_ok=True)
    Path(opath).write_text(json.dumps(out))
    print(f"golden: {opath} ({steps} steps, sum={x.sum().item():.4f})")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2],
         int(sys.argv[3]) if len(sys.argv) > 3 else 30)
