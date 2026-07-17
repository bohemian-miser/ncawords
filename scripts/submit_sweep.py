"""Submit the 30-experiment sweep, round-robin across the 5 regions that
have spot-T4 training quota (1 GPU each, so 5 parallel lanes)."""
import sys
sys.path.insert(0, ".")
from submit_vertex_job import build_and_upload_package, submit_job

REGIONS = ["us-east1", "us-west1", "europe-west2", "europe-west4", "us-central1"]

ORG = "nca/train_organic_reveal.py"
LSD = "nca/train_ladder_seed.py"
NLD = "nca/train_noise_ladder.py"
SLM = "nca/train_slime.py"

base_org = ["--text", "COMP", "--steps", "8000", "--log-every", "100"]
base_lsd = ["--text", "COMP", "--steps", "8000", "--log-every", "100"]
base_nld = ["--text", "COMP", "--log-every", "50"]
base_slm = ["--steps", "8000", "--log-every", "100", "--frames", "120"]

SWEEP = [
    # organic-reveal: letter-weight x growth
    (ORG, "org-lw4-bfs",      base_org + ["--letter-w", "4"]),
    (ORG, "org-lw16-bfs",     base_org + ["--letter-w", "16"]),
    (ORG, "org-lw8-dfs",      base_org + ["--growth", "dfs", "--frames", "80"]),
    (ORG, "org-lw16-dfs",     base_org + ["--growth", "dfs", "--frames", "80", "--letter-w", "16"]),
    # rotation modes, with the restart bug fixed
    (ORG, "org-norot-v2",     base_org + ["--rot-mode", "none"]),
    (ORG, "org-late20-v2",    base_org + ["--rot-mode", "late", "--rot-at", "1000", "--rot-deg", "20"]),
    # lifespan family rerun with weighted loss
    (ORG, "org-life6-v2",     base_org + ["--growth", "dfs", "--frames", "100", "--lifespan", "6"]),
    (ORG, "org-life12-v2",    base_org + ["--growth", "dfs", "--frames", "100", "--lifespan", "12"]),
    (ORG, "org-life24-v2",    base_org + ["--growth", "dfs", "--frames", "100", "--lifespan", "24"]),
    (ORG, "org-life12-bfs",   base_org + ["--frames", "100", "--lifespan", "12"]),
    # depth/coarseness
    (ORG, "org-16k",          ["--text", "COMP", "--steps", "16000", "--log-every", "200"]),
    (ORG, "org-frames40",     base_org + ["--frames", "40"]),
    (ORG, "org-frames120",    base_org + ["--growth", "dfs", "--frames", "120"]),
    (ORG, "org-text-nca",     ["--text", "NCA", "--steps", "8000", "--log-every", "100"]),
    # ladder-seed: normal-batch fraction x damage
    (LSD, "lseed-np10",       base_lsd + ["--normal-p", "0.1"]),
    (LSD, "lseed-np50",       base_lsd + ["--normal-p", "0.5"]),
    (LSD, "lseed-np10-dmg",   base_lsd + ["--normal-p", "0.1", "--damage-occasional"]),
    (LSD, "lseed-np50-dmg",   base_lsd + ["--normal-p", "0.5", "--damage-occasional"]),
    (LSD, "lseed-16k",        ["--text", "COMP", "--steps", "16000", "--log-every", "200"]),
    # noise ladder with anti-forgetting replay
    (NLD, "nladder-replay",       base_nld + ["--schedule", "ladder", "--replay-p", "0.3"]),
    (NLD, "nladder-jumps-replay", base_nld + ["--schedule", "ladder+jumps", "--replay-p", "0.3"]),
    (NLD, "nladder-jumps-replay50", base_nld + ["--schedule", "ladder+jumps", "--replay-p", "0.5"]),
    # slime: sensing and decay dynamics
    (SLM, "slime-sd3",        base_slm + ["--sensor-d", "3"]),
    (SLM, "slime-sd9",        base_slm + ["--sensor-d", "9"]),
    (SLM, "slime-evap06",     base_slm + ["--evap", "0.06"]),
    (SLM, "slime-evap24",     base_slm + ["--evap", "0.24"]),
    (SLM, "slime-agents2k",   base_slm + ["--agents", "2000"]),
    (SLM, "slime-agents8k",   base_slm + ["--agents", "8000"]),
    (SLM, "slime-sub6",       base_slm + ["--substeps", "6"]),
    (SLM, "slime-16k",        ["--steps", "16000", "--log-every", "200", "--frames", "120"]),
]

pkg = build_and_upload_package()
ok, fail = 0, 0
for i, (script, name, args) in enumerate(SWEEP):
    region = REGIONS[i % len(REGIONS)]
    try:
        submit_job(script, extra_args=args, job_name=name,
                   package_uri=pkg, location=region)
        ok += 1
        print(f"--- {name} -> {region}")
    except Exception as e:
        fail += 1
        print(f"!!! {name} -> {region} FAILED: {e}")
print(f"\nSubmitted {ok}/{len(SWEEP)} jobs ({fail} failures)")
