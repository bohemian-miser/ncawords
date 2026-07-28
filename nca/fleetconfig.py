"""Fleet configuration for a cloned NCA fleet.

Reads fleet.config.json at the repo root (written by ./init.sh, gitignored)
and falls back to built-in defaults for every missing key, so an
unconfigured checkout keeps working exactly as before.

Keys:
  bucket          GCS bucket name that runs are published to (public-read)
  project         GCP project that owns the bucket
  sa_key_path     service-account key used for uploads (optional at runtime)
  remote_hosts    ssh aliases of remote CPU workers (first one is the
                  default lane host)
  local_runs_dir  where collected / locally-trained run dirs live
"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "fleet.config.json"

DEFAULTS = {
    "bucket": "recipe-lanes-nca-jobs",
    "project": "recipe-lanes-staging",
    "sa_key_path": "~/.config/nca/submitter-key.json",
    "remote_hosts": ["cse10", "cse11", "cse12"],
    "local_runs_dir": "~/cse_runs",
}


def load():
    """Return the fleet config as a dict (defaults merged with the file).

    Paths (sa_key_path, local_runs_dir) are returned tilde-expanded.
    """
    cfg = dict(DEFAULTS)
    if CONFIG_PATH.exists():
        try:
            cfg.update(json.loads(CONFIG_PATH.read_text()))
        except (OSError, ValueError):
            pass  # a broken config file should not take the fleet down
    cfg["sa_key_path"] = os.path.expanduser(cfg["sa_key_path"])
    cfg["local_runs_dir"] = os.path.expanduser(cfg["local_runs_dir"])
    return cfg


def bucket_url(cfg=None):
    """Public https base URL of the runs bucket (no trailing slash)."""
    return "https://storage.googleapis.com/" + (cfg or load())["bucket"]


def bucket_api(cfg=None):
    """GCS JSON-API objects endpoint for the runs bucket."""
    return ("https://storage.googleapis.com/storage/v1/b/"
            + (cfg or load())["bucket"] + "/o")


def setup_credentials(cfg=None):
    """Point GOOGLE_APPLICATION_CREDENTIALS at the SA key if it exists."""
    cfg = cfg or load()
    if os.path.exists(cfg["sa_key_path"]):
        os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS",
                              cfg["sa_key_path"])
    return cfg
