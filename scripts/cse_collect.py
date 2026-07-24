"""Collect CSE run results and upload to the public bucket.

rsyncs /tmp/z3425319-nca/runs/* from the CSE host to ~/cse_runs/ locally,
then uploads any file newer than its bucket copy under the same run-dir
layout the dashboard/gallery already read. Also prints each run's last
log line and whether its process is still alive.

Usage: python scripts/cse_collect.py [--status-only]
"""
import argparse
import os
import subprocess
from pathlib import Path

os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS",
                      os.path.expanduser("~/.config/nca/submitter-key.json"))
from google.cloud import storage  # noqa: E402

LOCAL = Path.home() / "cse_runs"
BASE = "nca-runs"


def main(status_only=False):
    r = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "cse",
         f"for d in {BASE}/*/; do echo \"$(basename $d): "
         f"$(ls $d | wc -l) files\"; done 2>/dev/null"],
        capture_output=True, text=True, timeout=60)
    print(r.stdout.strip() or "(no runs on CSE)")
    if status_only:
        return
    LOCAL.mkdir(exist_ok=True)
    subprocess.run(["rsync", "-az", "--exclude=pid",
                    f"cse:{BASE}/", str(LOCAL) + "/"], check=True, timeout=600)
    client = storage.Client(project="recipe-lanes-staging")
    bucket = client.bucket("recipe-lanes-nca-jobs")
    n = 0
    for run_dir in LOCAL.iterdir():
        if not run_dir.is_dir():
            continue
        for f in run_dir.iterdir():
            if f.name in ("job.log", "pid") or not f.is_file():
                continue
            blob = bucket.blob(f"{run_dir.name}/{f.name}")
            mtime = f.stat().st_mtime
            blob_fresh = False
            if blob.exists():
                blob.reload()
                blob_fresh = blob.updated.timestamp() >= mtime
            if not blob_fresh:
                blob.upload_from_filename(str(f))
                n += 1
    print(f"uploaded {n} files")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--status-only", action="store_true")
    a = p.parse_args()
    main(a.status_only)
