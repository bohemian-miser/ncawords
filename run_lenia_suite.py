"""Master Runner for Trainable & Dynamic Kernel Lenia Suite.

Executes sequential training runs capped at <= 30% GPU utilization and
under 700 MB VRAM, keeping ample headroom for other concurrent workloads.

Covers:
1. Trainable-Kernel Lenia (3x3 and larger 5x5 kernels)
2. Dynamic Layer-Modulated Kernel NCA (where explicit weights and biases
   connect state and hidden layers to modulate the convolution filters).
"""

import os
import sys
import subprocess
import time
from pathlib import Path
import torch

BASE_DIR = Path('/nca')
RUNS_DIR = BASE_DIR / 'nca_runs'
RUNS_DIR.mkdir(parents=True, exist_ok=True)
DOCS_WEIGHTS = BASE_DIR / 'docs' / 'weights'
DOCS_WEIGHTS.mkdir(parents=True, exist_ok=True)

TRAIN_STATIC = str(BASE_DIR / 'nca' / 'train_lenia_pool.py')
TRAIN_DYNAMIC = str(BASE_DIR / 'nca' / 'train_dynamic_kernel.py')

JOBS = [
    # --- Part 1: Trainable Static Kernels (3x3 & larger 5x5) ---
    {
        'type': 'static',
        'name': 'snaps_lenia_kernel_fire_ring',
        'target': 'emoji:1f525',
        'init': 'ring',
        'num_k': 2,
        'ks': 3,
        'steps': 6000,
    },
    {
        'type': 'static',
        'name': 'snaps_lenia_kernel_rocket',
        'target': 'emoji:1f680',
        'init': 'sobel',
        'num_k': 2,
        'ks': 3,
        'steps': 6000,
    },
    {
        'type': 'static',
        'name': 'snaps_lenia_kernel_comp',
        'target': 'word:COMP',
        'init': 'sobel',
        'num_k': 2,
        'ks': 3,
        'steps': 6000,
    },
    {
        'type': 'static',
        'name': 'snaps_lenia_kernel5x5_fire',
        'target': 'emoji:1f525',
        'init': 'sobel',
        'num_k': 2,
        'ks': 5,
        'steps': 6000,
    },
    {
        'type': 'static',
        'name': 'snaps_lenia_kernel5x5_dots_ring',
        'target': 'dots',
        'init': 'ring',
        'num_k': 2,
        'ks': 5,
        'steps': 6000,
    },

    # --- Part 2: Dynamic Layer-Modulated Kernels (State & Hidden weights/biases) ---
    {
        'type': 'dynamic',
        'name': 'snaps_lenia_dynkernel5x5_dots',
        'target': 'dots',
        'init': 'sobel',
        'num_k': 2,
        'ks': 5,
        'steps': 6000,
    },
    {
        'type': 'dynamic',
        'name': 'snaps_lenia_dynkernel5x5_fire',
        'target': 'emoji:1f525',
        'init': 'sobel',
        'num_k': 2,
        'ks': 5,
        'steps': 6000,
    },
    {
        'type': 'dynamic',
        'name': 'snaps_lenia_dynkernel5x5_hex',
        'target': 'hex',
        'init': 'sobel',
        'num_k': 2,
        'ks': 5,
        'steps': 6000,
    },
]

def main():
    print(f"=== Starting Trainable & Dynamic Kernel Suite ({len(JOBS)} jobs) ===", flush=True)
    print("Target GPU Utilization: 30% | VRAM Cap: ~700 MB", flush=True)

    env = os.environ.copy()
    env['PYTHONPATH'] = str(BASE_DIR)
    env['NCA_GPU_UTIL'] = '0.30'
    env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    for i, job in enumerate(JOBS):
        name = job['name']
        snap_dir = RUNS_DIR / name
        target_steps = job['steps']

        # Check if already completed
        ckpt_file = snap_dir / 'ckpt.pth'
        if ckpt_file.exists():
            try:
                state = torch.load(ckpt_file, map_location='cpu', weights_only=False)
                step = state.get('step', 0)
                if step >= target_steps - 1:
                    print(f"[{i+1}/{len(JOBS)}] {name} already completed ({step+1} >= {target_steps} steps), skipping.", flush=True)
                    continue
                else:
                    print(f"[{i+1}/{len(JOBS)}] {name} found checkpoint at step {step+1}/{target_steps}. Resuming...", flush=True)
            except Exception:
                pass

        script = TRAIN_STATIC if job['type'] == 'static' else TRAIN_DYNAMIC
        cmd = [
            sys.executable, script,
            '--target', job['target'],
            '--init-kernels', job['init'],
            '--num-kernels', str(job['num_k']),
            '--kernel-size', str(job['ks']),
            '--steps', str(target_steps),
            '--batch', '8',
            '--pool-size', '256',
            '--log-every', '100',
            '--snap-dir', str(snap_dir)
        ]

        print(f"\n[{i+1}/{len(JOBS)}] Launching {name} ({job['type']}, target={job['target']}, ks={job['ks']}, init={job['init']}) ...", flush=True)
        t0 = time.time()
        res = subprocess.run(cmd, cwd=str(BASE_DIR), env=env)
        elapsed = time.time() - t0

        if res.returncode == 0:
            print(f"[SUCCESS] {name} completed in {elapsed:.1f}s!", flush=True)
            w_file = snap_dir / 'weights.json'
            if w_file.exists():
                dest = DOCS_WEIGHTS / f"{name}.json"
                dest.write_text(w_file.read_text())
                print(f"  Copied browser weights -> {dest}", flush=True)
        else:
            print(f"[FAILED] {name} exited with code {res.returncode}", flush=True)

    print("\n=== All Trainable & Dynamic Kernel Jobs Finished! ===", flush=True)

if __name__ == '__main__':
    main()
