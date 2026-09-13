#!/usr/bin/env python3
"""High-throughput parallel training for Flame Emoji NCA at 80% GPU utilization.

Uses batch size 32 (32 parallel organism boards per step) and 80% GPU compute duty cycle.
Trained from 100% PURE NOISE (cleared pool, pure noise nucleation) up to 100,000 steps.
"""

import os
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
SNAP_DIR = BASE_DIR / "nca_runs" / "snaps_lenia_dynkernel5x5_fire_noise"
SNAP_DIR.mkdir(parents=True, exist_ok=True)
DOCS_WEIGHTS = BASE_DIR / "docs" / "weights"
DOCS_WEIGHTS.mkdir(parents=True, exist_ok=True)

env = os.environ.copy()
env["PYTHONPATH"] = str(BASE_DIR)
env["NCA_GPU_UTIL"] = "0.80"
env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

cmd = [
    sys.executable,
    str(BASE_DIR / "nca" / "train_dynamic_kernel.py"),
    "--target", "emoji:1f525",
    "--init-kernels", "sobel",
    "--num-kernels", "2",
    "--num-basis", "4",
    "--kernel-size", "5",
    "--damage-p", "0.35",
    "--batch", "32",
    "--steps", "10000",
    "--pool-init", "noise",
    "--force-reset",
    "--log-every", "100",
    "--ckpt-every", "1000",
    "--snap-dir", str(SNAP_DIR)
]

def main():
    print(f"=== Launching Flame Emoji 10k Pure Noise Training ===", flush=True)
    print(f"Batch Size: 32 parallel boards | GPU Target Util: 80% | Steps: 10,000 | Pool: 100% Noise", flush=True)
    res = subprocess.run(cmd, env=env)
    if res.returncode == 0:
        w_file = SNAP_DIR / "weights.json"
        if w_file.exists():
            dest = DOCS_WEIGHTS / "snaps_lenia_dynkernel5x5_fire_noise.json"
            dest.write_text(w_file.read_text())
            print(f"Copied final weights to {dest}", flush=True)

if __name__ == '__main__':
    main()
