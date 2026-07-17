import time

from submit_vertex_job import build_and_upload_package, submit_job

# Each entry: (script, extra_args). Scripts differ in which flags they accept —
# train_guided.py takes no --text, the newer web scripts require it.
core_jobs = [
    ("nca/train_linear_cloud.py", ["--text", "COMP"]),
    ("nca/train_stage_cloud.py", ["--text", "COMP"]),
    ("nca/train_guided.py", []),
    ("nca/train_adaptive_cloud.py", ["--text", "COMP"]),
    ("nca/train_diffusion.py", ["--text", "COMP"]),
    ("nca/train_web_9_line.py", ["--text", "COMP"]),
    ("nca/train_web_evaporate.py", ["--text", "COMP"]),
    ("nca/train_web_hidden.py", ["--text", "COMP"]),
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

# Build/upload the source package once and reuse it for every job.
package_uri = build_and_upload_package()

print("--- Submitting Core Curriculum Jobs ---")
for script, extra_args in core_jobs:
    print(f"\n[Core Job] Submitting {script}...")
    try:
        submit_job(script, extra_args=extra_args, package_uri=package_uri)
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
        submit_job("nca/train_dynamic_organic.py", extra_args=extra_args,
                   job_name=job_name, package_uri=package_uri)
        time.sleep(2)
    except Exception as e:
        print(f"Failed to submit sweep {job_name}: {e}")

print("\nAll jobs submitted to Vertex AI!")
