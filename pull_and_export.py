import os
import subprocess

jobs = [
    "nca-train-train_web_hidden",
    "snaps_dyn_clear_n500_vol0_1",
    "snaps_dyn_clear_n1000_vol0_7",
    "snaps_dyn_clear_n1000_vol0_4",
    "snaps_dyn_clear_n1000_vol0_1"
]

for job in jobs:
    print(f"Pulling {job} ...")
    os.makedirs(job, exist_ok=True)
    subprocess.run(["gcloud", "storage", "cp", f"gs://recipe-lanes-nca-jobs/{job}/latest.pth", f"{job}/latest.pth"])
    
    print(f"Exporting {job} for WebGL ...")
    subprocess.run(["python", "export_webgl.py", f"{job}/latest.pth"])

print("All models exported to docs/weights/")
