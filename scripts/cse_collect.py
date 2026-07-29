"""Collect remote run results and upload to the public bucket.

rsyncs ~/nca-runs/* from each configured remote host (fleet.config.json
remote_hosts) into the local runs dir, then uploads any file newer than
its bucket copy under the same run-dir layout the dashboard/gallery
already read. Also prints each host's run dirs and file counts.

Hosts that share an NFS home (like a university login-VM fleet) will show
the same runs; the rsync is idempotent so that is only mildly wasteful.

Usage: python scripts/cse_collect.py [--status-only]
"""
import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nca import fleetconfig  # noqa: E402

CFG = fleetconfig.setup_credentials()
from google.cloud import storage  # noqa: E402

LOCAL = Path(CFG["local_runs_dir"])
BASE = "nca-runs"


def main(status_only=False):
    hosts = CFG["remote_hosts"]
    # The remote home is NFS-shared, so one host's listing covers the fleet;
    # printing every host repeated the same runs and hid which files the
    # rsync below actually pulled.
    for host in hosts:
        r = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", host,
             f"for d in {BASE}/*/; do echo \"$(basename $d): "
             f"$(ls $d | wc -l) files\"; done 2>/dev/null"],
            capture_output=True, text=True, timeout=60)
        if r.stdout.strip():
            print(r.stdout.strip())
            break
    else:
        print("(no runs reachable)")
    if status_only:
        return
    LOCAL.mkdir(parents=True, exist_ok=True)
    # The remote home is NFS-shared across hosts: one rsync sees every
    # run. Try hosts in order until one succeeds (fallback if a VM is down).
    for host in hosts:
        try:
            subprocess.run(["rsync", "-az", "--exclude=pid",
                            f"{host}:{BASE}/", str(LOCAL) + "/"],
                           check=True, timeout=600)
        except subprocess.SubprocessError as e:
            print(f"[{host}] rsync failed: {e}")
            continue
        break
    client = storage.Client(project=CFG["project"])
    bucket = client.bucket(CFG["bucket"])
    n = 0
    # Only consider files touched since the last successful pass (with an
    # hour of slack) — blob.exists() round-trips across every historical
    # file made the pass take ~9 minutes for 0 uploads.
    import time as _time
    stamp = LOCAL / ".last_collect"
    cutoff = (stamp.stat().st_mtime - 3600) if stamp.exists() else 0
    for run_dir in LOCAL.iterdir():
        if not run_dir.is_dir():
            continue
        for f in run_dir.iterdir():
            if f.name in ("job.log", "pid") or not f.is_file():
                continue
            mtime = f.stat().st_mtime
            if mtime < cutoff:
                continue
            blob = bucket.blob(f"{run_dir.name}/{f.name}")
            blob_fresh = False
            if blob.exists():
                blob.reload()
                blob_fresh = blob.updated.timestamp() >= mtime
            if not blob_fresh:
                blob.upload_from_filename(str(f))
                n += 1
    stamp.touch()
    print(f"uploaded {n} files")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--status-only", action="store_true")
    a = p.parse_args()
    main(a.status_only)
