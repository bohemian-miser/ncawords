import torch
from pathlib import Path

runs_dir = Path("nca_runs")
runs = [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("snaps_grid_")]

steps = []
for run in runs:
    ckpt = run / "ckpt.pth"
    if ckpt.exists():
        try:
            state = torch.load(ckpt, map_location="cpu", weights_only=False)
            steps.append(state["step"])
        except Exception:
            steps.append(-1)
    else:
        steps.append(-1)

print(f"Total runs: {len(steps)}")
print(f"Min steps: {min(steps)}")
print(f"Max steps: {max(steps)}")
print(f"Avg steps: {sum(steps)/len(steps):.1f}")
print(f"Completed (>5000): {len([s for s in steps if s >= 5000])}")
