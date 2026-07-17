import sys
import time
from submit_vertex_job import submit_job

core_scripts = [
    "nca/train_linear_cloud.py",
    "nca/train_stage_cloud.py",
    "nca/train_guided.py",
    "nca/train_adaptive_cloud.py",
    "nca/train_diffusion.py",
    "nca/train_web_9_line.py",
    "nca/train_web_evaporate.py",
    "nca/train_web_hidden.py",
]

dyn_organic_sweeps = [
    {"n": 100, "vol": 0.4},
    {"n": 100, "vol": 0.1},
    {"n": 500, "vol": 0.4},
    {"n": 500, "vol": 0.1},
    {"n": 1000, "vol": 0.7},
    {"n": 1000, "vol": 0.4},
    {"n": 1000, "vol": 0.1},
]

print("--- Submitting Core Curriculum Jobs ---")
for script in core_scripts:
    print(f"\n[Core Job] Submitting {script}...")
    try:
        submit_job(script)
        time.sleep(2)
    except Exception as e:
        print(f"Failed to submit {script}: {e}")

print("\n--- Submitting Dynamic Organic Sweep Jobs ---")
for sweep in dyn_organic_sweeps:
    n = sweep["n"]
    vol = sweep["vol"]
    vol_str = str(vol).replace(".", "_")
    job_name = f"snaps_dyn_clear_n{n}_vol{vol_str}"
    
    extra_args = [
        "--text", "COMP",
        "--steps", "16000",
        "--log-every", "100",
        "--no-noise",
        "--update-every", str(n),
        "--support-vol", str(vol),
    ]
    print(f"\n[Sweep Job] Submitting Dynamic Organic (N={n}, Vol={vol}) as {job_name}...")
    try:
        submit_job("nca/train_dynamic_organic.py", extra_args=extra_args, job_name=job_name)
        time.sleep(2)
    except Exception as e:
        print(f"Failed to submit sweep {job_name}: {e}")

print("\nAll remaining jobs submitted to Vertex AI!")
