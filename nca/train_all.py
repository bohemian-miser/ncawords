"""Train NCAs for a set of characters using a small process pool.

Each worker trains one character at a time with a limited thread count so
several run in parallel on the Pi. After each finishes, the glyph is grown
and OCR-checked; failures are queued for one retry with more steps.

Usage:
  python -m nca.train_all --chars ABCDEFGHIJKLMNOPQRSTUVWXYZ --workers 2
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv" / "bin" / "python")


def spawn(ch, steps, threads, log_dir):
    out = ROOT / "weights" / f"{ord(ch):04x}.json"
    log = open(log_dir / f"{ord(ch):04x}.log", "a")
    p = subprocess.Popen(
        [PY, "-m", "nca.train", "--char", ch, "--steps", str(steps),
         "--out", str(out)],
        cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
        env={"OMP_NUM_THREADS": str(threads), "PATH": "/usr/bin:/bin",
             "HOME": str(Path.home())})
    return p, out


def ocr_check(ch):
    r = subprocess.run(
        [PY, "-m", "nca.ocr_eval", str(ROOT / "weights" / f"{ord(ch):04x}.json"),
         "--report", str(ROOT / "ocr" / f"{ord(ch):04x}.json"),
         "--img-dir", str(ROOT / "grown")],
        cwd=ROOT, capture_output=True, text=True,
        env={"OMP_NUM_THREADS": "1", "PATH": "/usr/bin:/bin",
             "HOME": str(Path.home())})
    return r.returncode == 0, r.stdout.strip().splitlines()[:1]


def write_index():
    chars = []
    for f in sorted((ROOT / "weights").glob("*.json")):
        try:
            chars.append(json.loads(f.read_text())["char"])
        except Exception:
            pass
    idx = ROOT / "weights" / "index.json"
    idx.write_text(json.dumps({"chars": sorted(set(chars))}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chars", default="ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--retry-steps", type=int, default=2500)
    ap.add_argument("--skip-existing", action="store_true")
    a = ap.parse_args()

    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    (ROOT / "ocr").mkdir(exist_ok=True)

    queue = [(ch, a.steps, 0) for ch in a.chars
             if not (a.skip_existing
                     and (ROOT / "weights" / f"{ord(ch):04x}.json").exists())]
    running = {}   # ch -> (proc, steps, attempt)
    failed, passed = [], []
    t0 = time.time()

    while queue or running:
        while queue and len(running) < a.workers:
            ch, steps, attempt = queue.pop(0)
            p, _ = spawn(ch, steps, a.threads, log_dir)
            running[ch] = (p, steps, attempt)
            print(f"[orch] start {ch} steps={steps} attempt={attempt} "
                  f"({len(queue)} queued)", flush=True)
        time.sleep(10)
        for ch in list(running):
            p, steps, attempt = running[ch]
            if p.poll() is None:
                continue
            del running[ch]
            if p.returncode != 0:
                print(f"[orch] {ch} CRASHED (rc={p.returncode})", flush=True)
                failed.append(ch)
                continue
            ok, line = ocr_check(ch)
            print(f"[orch] {ch} done in attempt {attempt}: "
                  f"{'OCR-PASS' if ok else 'OCR-FAIL'} {line} "
                  f"[{(time.time()-t0)/60:.0f}m elapsed]", flush=True)
            if ok:
                passed.append(ch)
            elif attempt == 0:
                queue.append((ch, a.retry_steps, 1))
            else:
                failed.append(ch)
            write_index()

    write_index()
    print(f"[orch] ALL DONE in {(time.time()-t0)/60:.0f}m. "
          f"passed={''.join(sorted(passed))} failed={''.join(sorted(failed))}",
          flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
