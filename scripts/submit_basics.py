"""Back-to-basics sweep: ~100 runs over the fundamentals — seeds, simple
supports, noise regimes, damage, and training length — using the proven
training scripts and the verified synchronous regional submitter.

Regions are weighted by quota (us-central1 has 8 L4 + 8 T4 spot; others
2 L4 + 1 T4). Machines alternate l4/t4 to draw on both quota pools.
"""
import sys
sys.path.insert(0, ".")
from submit_vertex_job import build_and_upload_package, submit_job

WH = "nca/train_web_hidden.py"      # hidden 9-line scaffold (proven recipe)
W9 = "nca/train_web_9_line.py"      # visible scaffold
WE = "nca/train_web_evaporate.py"   # scaffold that fades out
LS = "nca/train_ladder_seed.py"     # modern pool trainer (noise schedule opt-out via normal-p)
NL = "nca/train_noise_ladder.py"    # staged denoiser
DF = "nca/train_diffusion.py"       # noise-pair curriculum
OR_ = "nca/train_organic_reveal.py"
SL = "nca/train_slime.py"

RUNS = []

def add(script, name, args):
    RUNS.append((script, name, args))

# A/B: scaffold recipes x text x seed x length
for script, tag in [(WH, "hid"), (W9, "9line")]:
    for text in ["COMP", "NCA", "6841"]:
        for seed in ["single", "noise"]:
            for steps in [8000, 16000]:
                add(script, f"base-{tag}-{text.lower()}-{seed}-{steps//1000}k",
                    ["--text", text, "--seed-type", seed, "--steps", str(steps),
                     "--log-every", "200"])

# C: evaporating scaffold
for text in ["COMP", "NCA", "6841"]:
    for steps in [8000, 16000]:
        add(WE, f"base-evap-{text.lower()}-{steps//1000}k",
            ["--text", text, "--steps", str(steps), "--log-every", "200"])

# D: plain pool trainer (no noise), damage on/off
for text in ["COMP", "NCA", "6841"]:
    for steps in [8000, 16000]:
        for dmg in [False, True]:
            add(LS, f"base-plain-{text.lower()}-{steps//1000}k{'-dmg' if dmg else ''}",
                ["--text", text, "--steps", str(steps), "--log-every", "200",
                 "--normal-p", "1.0"] + (["--damage-occasional"] if dmg else []))

# E: noise regime — fraction of clean batches
for np_ in ["0.0", "0.25", "0.5", "0.75"]:
    for steps in [8000, 16000]:
        add(LS, f"base-noise-np{np_.replace('.','')}-{steps//1000}k",
            ["--text", "COMP", "--steps", str(steps), "--log-every", "200",
             "--normal-p", np_])

# F: denoiser replay refinement
for rp in ["0.2", "0.4"]:
    for sched in ["ladder", "ladder+jumps"]:
        tag = "l" if sched == "ladder" else "lj"
        add(NL, f"base-nlad-{tag}-rp{rp.replace('.','')}",
            ["--text", "COMP", "--log-every", "50", "--schedule", sched,
             "--replay-p", rp])
add(NL, "base-nlad-lj-rp30-x2",
    ["--text", "COMP", "--log-every", "50", "--schedule", "ladder+jumps",
     "--replay-p", "0.3", "--stage-steps", "200"])
add(NL, "base-nlad-l-rp30-x2",
    ["--text", "COMP", "--log-every", "50", "--schedule", "ladder",
     "--replay-p", "0.3", "--stage-steps", "200"])

# G: organic with the fixed loss
for lw in ["2", "8"]:
    for growth in ["bfs", "dfs"]:
        add(OR_, f"base-org-lw{lw}-{growth}",
            ["--text", "COMP", "--steps", "8000", "--log-every", "100",
             "--letter-w", lw, "--growth", growth]
            + (["--frames", "80"] if growth == "dfs" else []))
for n in ["6", "12", "24"]:
    add(OR_, f"base-org-life{n}",
        ["--text", "COMP", "--steps", "8000", "--log-every", "100",
         "--growth", "dfs", "--frames", "100", "--lifespan", n])
add(OR_, "base-org-norot", ["--text", "COMP", "--steps", "8000",
                            "--log-every", "100", "--rot-mode", "none"])
add(OR_, "base-org-late20", ["--text", "COMP", "--steps", "8000",
                             "--log-every", "100", "--rot-mode", "late",
                             "--rot-at", "1000", "--rot-deg", "20"])
add(OR_, "base-org-16k", ["--text", "COMP", "--steps", "16000",
                          "--log-every", "200"])
add(OR_, "base-org-nca", ["--text", "NCA", "--steps", "8000",
                          "--log-every", "100"])
for fr in ["40", "80"]:
    add(OR_, f"base-org-frames{fr}",
        ["--text", "COMP", "--steps", "8000", "--log-every", "100",
         "--frames", fr])

# H/N/S: slime — replication and parameter crosses
for rs in ["1", "2"]:
    add(SL, f"base-slime-rng{rs}",
        ["--steps", "8000", "--log-every", "100", "--frames", "120",
         "--rng-seed", rs])
add(SL, "base-slime-sd9-rng1", ["--steps", "8000", "--log-every", "100",
                                "--frames", "120", "--sensor-d", "9",
                                "--rng-seed", "1"])
add(SL, "base-slime-sd3-evap06", ["--steps", "8000", "--log-every", "100",
                                  "--frames", "120", "--sensor-d", "3",
                                  "--evap", "0.06"])
add(SL, "base-slime-sd3-evap24", ["--steps", "8000", "--log-every", "100",
                                  "--frames", "120", "--sensor-d", "3",
                                  "--evap", "0.24"])
add(SL, "base-slime-sd9-evap24", ["--steps", "8000", "--log-every", "100",
                                  "--frames", "120", "--sensor-d", "9",
                                  "--evap", "0.24"])
add(SL, "base-slime-frames240", ["--steps", "8000", "--log-every", "100",
                                 "--frames", "240"])
add(SL, "base-slime-16k-sd9", ["--steps", "16000", "--log-every", "200",
                               "--frames", "120", "--sensor-d", "9"])

# I/M: diffusion baseline
for text in ["COMP", "NCA", "6841"]:
    add(DF, f"base-diff-{text.lower()}",
        ["--text", text, "--steps", "8000", "--log-every", "200"])
add(DF, "base-diff-comp-16k", ["--text", "COMP", "--steps", "16000",
                               "--log-every", "200"])

# Q/T: long-haul
add(WH, "base-hid-comp-single-32k", ["--text", "COMP", "--seed-type", "single",
                                     "--steps", "32000", "--log-every", "400"])
add(LS, "base-plain-comp-32k", ["--text", "COMP", "--steps", "32000",
                                "--log-every", "400", "--normal-p", "1.0"])

# Region cycle weighted by quota; machines alternate to use both pools.
REGION_CYCLE = ["us-central1", "us-central1", "us-east1", "us-west1",
                "us-central1", "europe-west2", "europe-west4"]

if __name__ == "__main__":
    print(f"{len(RUNS)} runs planned")
    if "--dry-run" in sys.argv:
        for s, n, a in RUNS:
            print(f"  {n}: {s} {' '.join(a)}")
        sys.exit(0)
    pkg = build_and_upload_package()
    ok, fail = 0, 0
    for i, (script, name, args) in enumerate(RUNS):
        region = REGION_CYCLE[i % len(REGION_CYCLE)]
        machine = "t4" if i % 3 == 2 else "l4"
        try:
            submit_job(script, extra_args=args, job_name=name,
                       package_uri=pkg, location=region, machine=machine)
            ok += 1
        except Exception as e:
            fail += 1
            print(f"!!! {name} -> {region}: {str(e)[:140]}")
    print(f"\nCreated {ok}/{len(RUNS)} ({fail} failures)")
