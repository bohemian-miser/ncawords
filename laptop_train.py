import os, subprocess, time, sys
from pathlib import Path
import torch

RUNS_DIR = Path("nca_runs")
TRAIN_SCRIPT = "nca/train_emoji_grid.py"

def main():
    print(f"Detected GPU: {torch.cuda.is_available() or torch.backends.mps.is_available()}")
    print("Executing sequential fast-GPU progression loop.")
    runs = [d for d in RUNS_DIR.iterdir() if d.is_dir() and d.name.startswith("snaps_grid_") and "1f600" not in d.name]
    
    for target_steps in [5000, 20000]:
        print(f"=== Starting Target Phase: {target_steps} steps ===")
        for count, run_dir in enumerate(runs):
            ckpt_file = run_dir / "ckpt.pth"
            current_step = 0
            if ckpt_file.exists():
                try:
                    state = torch.load(ckpt_file, map_location="cpu", weights_only=False)
                    current_step = state["step"] + 1
                except Exception: pass
                
            if current_step >= target_steps:
                continue
                
            name = run_dir.name
            parts = name.split("_")
            try:
                emoji, hidden, noise, fester = parts[2], int(parts[3][1:]), float(parts[4][1:]), int(parts[5][1:])
            except Exception: continue
            
            cmd = [
                sys.executable, TRAIN_SCRIPT,
                "--emoji", emoji,
                "--steps", str(target_steps),
                "--hidden-n", str(hidden),
                "--ceil-noise", str(noise),
                "--max-fester", str(fester),
                "--snap-dir", str(run_dir)
            ]
            
            print(f"[{count+1}/{len(runs)}] Accelerating {name} to step {target_steps} (current: {current_step}) ...")
            subprocess.run(cmd)

if __name__ == "__main__":
    main()
