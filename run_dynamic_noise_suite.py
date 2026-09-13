#!/usr/bin/env python3
"""Runner for Dynamic Kernel Noise-Robust Suite on aisb.

Trains dynamic-kernel NCAs from scratch with noise curriculum so they can
both nucleate/self-organize from pure noise and maintain/regenerate patterns.
Enforces strict <= 30% GPU duty-cycle throttle and exports POOL_#####.png mosaics.
"""

import os
import sys
import subprocess
import time
from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent
RUNS_DIR = BASE_DIR / 'nca_runs'
RUNS_DIR.mkdir(parents=True, exist_ok=True)
DOCS_WEIGHTS = BASE_DIR / 'docs' / 'weights'
DOCS_WEIGHTS.mkdir(parents=True, exist_ok=True)

TRAIN_DYNAMIC = str(BASE_DIR / 'nca' / 'train_dynamic_kernel.py')

JOBS = [
    {
        'name': 'snaps_lenia_dynkernel5x5_fire_noise',
        'target': 'emoji:1f525',
        'init': 'sobel',
        'num_k': 2,
        'num_basis': 4,
        'ks': 5,
        'damage_p': 0.35,
        'steps': 6000,
        'batch': 8,
    },
    {
        'name': 'snaps_lenia_dynkernel5x5_dots_noise',
        'target': 'dots',
        'init': 'sobel',
        'num_k': 2,
        'num_basis': 4,
        'ks': 5,
        'damage_p': 0.25,
        'steps': 6000,
        'batch': 8,
    },
    {
        'name': 'snaps_lenia_dynkernel5x5_hex_noise',
        'target': 'hex',
        'init': 'sobel',
        'num_k': 2,
        'num_basis': 4,
        'ks': 5,
        'damage_p': 0.15,
        'steps': 6000,
        'batch': 8,
    }
]


def main():
    print(f"=== Starting Dynamic Kernel Noise-Robust Suite ({len(JOBS)} jobs) ===", flush=True)
    print("GPU Utilization Cap: 30% (duty-cycle sleep) | Base Dir: " + str(BASE_DIR), flush=True)

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
                ckpt = torch.load(str(ckpt_file), map_location='cpu', weights_only=False)
                if ckpt.get('step', 0) >= target_steps:
                    print(f"[{i+1}/{len(JOBS)}] {name} already completed ({target_steps} steps). Skipping.", flush=True)
                    continue
            except Exception:
                pass

        print(f"\n=======================================================", flush=True)
        print(f"[{i+1}/{len(JOBS)}] Launching {name} ({job['target']}) for {target_steps} steps...", flush=True)
        print(f"=======================================================", flush=True)

        cmd = [
            sys.executable, TRAIN_DYNAMIC,
            '--target', job['target'],
            '--init-kernels', job['init'],
            '--num-kernels', str(job['num_k']),
            '--num-basis', str(job['num_basis']),
            '--kernel-size', str(job['ks']),
            '--damage-p', str(job['damage_p']),
            '--batch', str(job['batch']),
            '--steps', str(target_steps),
            '--log-every', '100',
            '--ckpt-every', '500',
            '--snap-dir', str(snap_dir)
        ]

        t0 = time.time()
        res = subprocess.run(cmd, env=env)
        elapsed = time.time() - t0

        if res.returncode != 0:
            print(f"ERROR: {name} failed with return code {res.returncode}", flush=True)
        else:
            print(f"SUCCESS: {name} completed in {elapsed:.1f}s", flush=True)

            # Copy exported weights to docs/weights for instant browser availability
            w_file = snap_dir / 'weights.json'
            if w_file.exists():
                dest_w = DOCS_WEIGHTS / f"{name}.json"
                dest_w.write_text(w_file.read_text())
                print(f"Copied weights to {dest_w}", flush=True)


if __name__ == '__main__':
    main()
