#!/usr/bin/env python3
"""Multi-Emoji Concurrent Suite Runner on aisb.

Trains 6 standard emojis from 100% PURE NOISE up to 100,000 steps:
  Cohort 1:
    🔥 Fire:      emoji:1f525  -> snaps_lenia_dynkernel5x5_fire_noise
    🚀 Rocket:    emoji:1f680  -> snaps_lenia_dynkernel5x5_rocket_noise
    🦎 Gecko:     emoji:1f98e  -> snaps_lenia_dynkernel5x5_gecko_noise

  Cohort 2:
    ❤️ Heart:     emoji:2764   -> snaps_lenia_dynkernel5x5_heart_noise
    👾 Alien:     emoji:1f47e  -> snaps_lenia_dynkernel5x5_alien_noise
    🦋 Butterfly: emoji:1f98b  -> snaps_lenia_dynkernel5x5_butterfly_noise

Design:
  - 3 Concurrent workers active simultaneously on GPU (80% aggregate GPU utilization: 26% each)
  - VRAM safety: ~11.5 GB allocated across 3 jobs, leaving 11.5 GB headroom on A10
  - Rotating cohorts every 500 steps so all 6 emojis advance together in the UI
  - 100% Pure Noise Pool initialization & nucleation (--pool-init noise)
  - Automatic browser weights export and POOL mosaic sync
"""

import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent
RUNS_DIR = BASE_DIR / "nca_runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)
DOCS_WEIGHTS = BASE_DIR / "docs" / "weights"
DOCS_WEIGHTS.mkdir(parents=True, exist_ok=True)
TRAIN_DYNAMIC = BASE_DIR / "nca" / "train_dynamic_kernel.py"

ALL_EMOJIS = [
    {"id": "fire", "emoji": "🔥", "target": "emoji:1f525", "name": "snaps_lenia_dynkernel5x5_fire_noise", "damage_p": 0.35},
    {"id": "rocket", "emoji": "🚀", "target": "emoji:1f680", "name": "snaps_lenia_dynkernel5x5_rocket_noise", "damage_p": 0.35},
    {"id": "gecko", "emoji": "🦎", "target": "emoji:1f98e", "name": "snaps_lenia_dynkernel5x5_gecko_noise", "damage_p": 0.35},
    {"id": "heart", "emoji": "❤️", "target": "emoji:2764", "name": "snaps_lenia_dynkernel5x5_heart_noise", "damage_p": 0.30},
    {"id": "alien", "emoji": "👾", "target": "emoji:1f47e", "name": "snaps_lenia_dynkernel5x5_alien_noise", "damage_p": 0.30},
    {"id": "butterfly", "emoji": "🦋", "target": "emoji:1f98b", "name": "snaps_lenia_dynkernel5x5_butterfly_noise", "damage_p": 0.30},
]

MAX_CONCURRENT = 2
CHUNK_STEPS = int(os.environ.get("NCA_CHUNK_STEPS", "500"))
BATCH_SIZE = int(os.environ.get("NCA_BATCH", "16"))
PER_JOB_UTIL = "1.0"  # 100% compute duty cycle (aggressive GPU utilization)


def get_total_steps():
    cfg_file = BASE_DIR / "max_steps.txt"
    if cfg_file.exists():
        try:
            return int(cfg_file.read_text().strip())
        except Exception:
            pass
    return int(os.environ.get("NCA_STEPS", "10000"))


def get_current_step(snap_dir):
    """Reads the current step from checkpoint if present."""
    ckpt = snap_dir / "ckpt.pth"
    if not ckpt.exists():
        return 0
    try:
        data = torch.load(ckpt, map_location="cpu", weights_only=False)
        return data.get("step", 0) + 1
    except Exception:
        return 0


def parse_latest_log(log_path):
    if not log_path.exists():
        return {"step": 0, "mse": "-", "spread": "-", "alive_std": "-"}
    try:
        lines = log_path.read_text().splitlines()
        for l in reversed(lines):
            if "step" in l and "mse" in l:
                parts = l.split("|")
                step_str = parts[0].split("step")[-1].strip()
                mse_str = parts[1].split("mse")[-1].strip() if len(parts) > 1 else "-"
                std_str = parts[2].split("alive_std")[-1].strip() if len(parts) > 2 else "-"
                spread_str = parts[3].split("spread")[-1].strip() if len(parts) > 3 else "-"
                return {
                    "step": int(step_str),
                    "mse": mse_str,
                    "alive_std": std_str,
                    "spread": spread_str
                }
    except Exception:
        pass
    return {"step": 0, "mse": "-", "spread": "-", "alive_std": "-"}


def print_status_table():
    total_steps = get_total_steps()
    now_str = datetime.now().strftime("%H:%M:%S")
    print("\n" + "=" * 68, flush=True)
    print(f"📊 Multi-Emoji Noise Suite Dashboard [{now_str}] (Cap: {total_steps:,})", flush=True)
    print(f"{'Emoji':<10} {'Target':<14} {'Step':<12} {'MSE':<12} {'Alive Std':<12} {'Spread':<8}", flush=True)
    print("-" * 68, flush=True)
    for e in ALL_EMOJIS:
        snap_dir = RUNS_DIR / e["name"]
        log_file = BASE_DIR / f"emoji_{e['id']}.log"
        info = parse_latest_log(log_file)
        cur_step = get_current_step(snap_dir)
        step_disp = f"{max(cur_step, info['step']):,}/{total_steps:,}"
        print(f"{e['emoji']} {e['id']:<7} {e['target']:<14} {step_disp:<12} {info['mse']:<12} {info['alive_std']:<12} {info['spread']:<8}", flush=True)
    print("=" * 68 + "\n", flush=True)


def main():
    total_steps = get_total_steps()
    print("=" * 70, flush=True)
    print(f"🚀 MULTI-EMOJI NOISE SUITE: 100% AGGRESSIVE GPU UTILIZATION PIPELINE", flush=True)
    print(f"   Target GPU Util: 100% ({MAX_CONCURRENT} continuous workers @ 100% duty cycle, 0% idle)", flush=True)
    print(f"   Batch Size:      {BATCH_SIZE} boards per step", flush=True)
    print(f"   Cap per Emoji:   {total_steps:,} steps", flush=True)
    print(f"   Dispatch Chunk:  {CHUNK_STEPS} steps per epoch", flush=True)
    print(f"   Initial Pool:    100% PURE NOISE (no center seeds)", flush=True)
    print("=" * 70, flush=True)

    # Clean non-fire run directories if first launch
    for e in ALL_EMOJIS:
        snap_dir = RUNS_DIR / e["name"]
        if e["id"] != "fire" and not (snap_dir / "ckpt.pth").exists():
            snap_dir.mkdir(parents=True, exist_ok=True)

    # active_procs: dict of e_id -> (Popen, log_fp, emoji_dict, target_step)
    active_procs = {}

    while True:
        total_steps = get_total_steps()

        # 1. Reap any completed workers
        finished_ids = []
        for e_id, (p, fp, e, target_step) in list(active_procs.items()):
            ret = p.poll()
            if ret is not None:
                fp.close()
                finished_ids.append(e_id)
                if ret == 0:
                    print(f"  [SUCCESS] {e['emoji']} {e_id} finished chunk at {target_step:,} steps!", flush=True)
                    w_file = RUNS_DIR / e["name"] / "weights.json"
                    if w_file.exists():
                        dest = DOCS_WEIGHTS / f"{e['name']}.json"
                        dest.write_text(w_file.read_text())
                else:
                    print(f"  [ERROR] {e['emoji']} {e_id} exited with code {ret}", flush=True)
        for d in finished_ids:
            del active_procs[d]

        if finished_ids:
            print_status_table()

        # 2. Check if all 6 emojis have completed total_steps
        all_steps = {e["id"]: get_current_step(RUNS_DIR / e["name"]) for e in ALL_EMOJIS}
        if all(s >= total_steps for s in all_steps.values()):
            if not active_procs:
                print(f"\n🎉 ALL 6 EMOJIS HAVE COMPLETED {total_steps:,} STEPS!", flush=True)
                break

        # 3. Aggressively fill open worker slots up to MAX_CONCURRENT
        while len(active_procs) < MAX_CONCURRENT:
            candidates = [
                e for e in ALL_EMOJIS
                if e["id"] not in active_procs
                and get_current_step(RUNS_DIR / e["name"]) < total_steps
            ]
            if not candidates:
                break  # All eligible emojis are currently in-flight or done

            # Prioritize organism with lowest current step to keep them progressing evenly
            best_candidate = min(candidates, key=lambda e: get_current_step(RUNS_DIR / e["name"]))
            e_id = best_candidate["id"]
            snap_dir = RUNS_DIR / best_candidate["name"]
            cur_step = get_current_step(snap_dir)
            target_step = min(cur_step + CHUNK_STEPS, total_steps)
            log_file = BASE_DIR / f"emoji_{e_id}.log"

            env = os.environ.copy()
            env["PYTHONPATH"] = str(BASE_DIR)
            env["NCA_GPU_UTIL"] = "1.0"
            env["NCA_GPU_MEM_FRAC"] = "0.45"
            env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

            cmd = [
                sys.executable, str(TRAIN_DYNAMIC),
                "--target", best_candidate["target"],
                "--init-kernels", "sobel",
                "--num-kernels", "2",
                "--num-basis", "4",
                "--kernel-size", "5",
                "--damage-p", str(best_candidate["damage_p"]),
                "--batch", str(BATCH_SIZE),
                "--steps", str(target_step),
                "--pool-init", "noise",
                "--log-every", "100",
                "--ckpt-every", str(CHUNK_STEPS),
                "--snap-dir", str(snap_dir),
                "--target-util", "1.0"
            ]
            if cur_step == 0:
                cmd.append("--force-reset")

            log_fp = open(log_file, "a")
            print(f"  [DISPATCH] {best_candidate['emoji']} {e_id.upper()} stepping {cur_step:,} -> {target_step:,} (100% duty)...", flush=True)
            p = subprocess.Popen(cmd, env=env, stdout=log_fp, stderr=subprocess.STDOUT)
            active_procs[e_id] = (p, log_fp, best_candidate, target_step)
            time.sleep(1.0)  # brief 1s pause between process starts

        time.sleep(0.5)  # rapid 0.5s poll for instant handoff


if __name__ == "__main__":
    main()
