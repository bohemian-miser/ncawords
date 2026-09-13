import os
import sys
import subprocess
import time
from pathlib import Path

BASE_DIR = Path('/nca')
RUNS_DIR = BASE_DIR / 'nca_runs'
RUNS_DIR.mkdir(parents=True, exist_ok=True)
DOCS_WEIGHTS = BASE_DIR / 'docs' / 'weights'
DOCS_WEIGHTS.mkdir(parents=True, exist_ok=True)
TRAIN_SCRIPT = str(BASE_DIR / 'nca' / 'train_lenia_pool.py')

# Trainable-Kernel Lenia Suite:
# Uses standard Distill continuous CA dynamics with sample pool,
# but the spatial convolution kernels are trained end-to-end with gradient descent.
JOBS = [
    {
        'name': 'snaps_lenia_kernel_dots',
        'target': 'dots',
        'init': 'sobel',
        'num_k': 2,
        'ks': 3,
        'steps': 6000,
    },
    {
        'name': 'snaps_lenia_kernel_dots_ring',
        'target': 'dots',
        'init': 'ring',
        'num_k': 2,
        'ks': 3,
        'steps': 6000,
    },
    {
        'name': 'snaps_lenia_kernel_hex',
        'target': 'hex',
        'init': 'sobel',
        'num_k': 2,
        'ks': 3,
        'steps': 6000,
    },
    {
        'name': 'snaps_lenia_kernel_fire',
        'target': 'emoji:1f525',
        'init': 'sobel',
        'num_k': 2,
        'ks': 3,
        'steps': 6000,
    },
    {
        'name': 'snaps_lenia_kernel_fire_ring',
        'target': 'emoji:1f525',
        'init': 'ring',
        'num_k': 2,
        'ks': 3,
        'steps': 6000,
    },
    {
        'name': 'snaps_lenia_kernel_rocket',
        'target': 'emoji:1f680',
        'init': 'sobel',
        'num_k': 2,
        'ks': 3,
        'steps': 6000,
    },
    {
        'name': 'snaps_lenia_kernel_comp',
        'target': 'word:COMP',
        'init': 'sobel',
        'num_k': 2,
        'ks': 3,
        'steps': 6000,
    },
    {
        'name': 'snaps_lenia_kernel_heart',
        'target': 'emoji:2764',
        'init': 'sobel',
        'num_k': 2,
        'ks': 3,
        'steps': 6000,
    },
    {
        'name': 'snaps_lenia_kernel_tri',
        'target': 'tri',
        'init': 'sobel',
        'num_k': 2,
        'ks': 3,
        'steps': 6000,
    },
    {
        'name': 'snaps_lenia_kernel_alien',
        'target': 'emoji:1f47e',
        'init': 'sobel',
        'num_k': 2,
        'ks': 3,
        'steps': 6000,
    },
    {
        'name': 'snaps_lenia_kernel_butterfly',
        'target': 'emoji:1f98b',
        'init': 'sobel',
        'num_k': 2,
        'ks': 3,
        'steps': 6000,
    },
    {
        'name': 'snaps_lenia_kernel_square',
        'target': 'square',
        'init': 'ring',
        'num_k': 2,
        'ks': 3,
        'steps': 6000,
    },
]

def main():
    print(f"=== Starting Trainable-Kernel Lenia Suite ({len(JOBS)} jobs) ===", flush=True)
    env = os.environ.copy()
    env['PYTHONPATH'] = str(BASE_DIR)
    env['NCA_GPU_UTIL'] = '0.50'
    env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    for i, job in enumerate(JOBS):
        name = job['name']
        snap_dir = RUNS_DIR / name
        cmd = [
            sys.executable, TRAIN_SCRIPT,
            '--target', job['target'],
            '--init-kernels', job['init'],
            '--num-kernels', str(job['num_k']),
            '--kernel-size', str(job['ks']),
            '--steps', str(job['steps']),
            '--batch', '8',
            '--pool-size', '256',
            '--log-every', '100',
            '--snap-dir', str(snap_dir)
        ]

        print(f"\n[{i+1}/{len(JOBS)}] Launching {name} (target={job['target']}, init={job['init']}) ...", flush=True)
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

    print("\n=== Trainable-Kernel Lenia Suite Finished! ===", flush=True)

if __name__ == '__main__':
    main()
