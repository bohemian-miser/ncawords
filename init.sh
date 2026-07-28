#!/bin/bash
# One-time setup for a cloned NCA fleet.
#
#   ./init.sh --bucket my-bucket --project my-gcp-project \
#             [--sa-key ~/.config/nca/key.json] \
#             [--remote-host alias]... [--local-runs-dir ~/nca_runs]
#
# Any flag you omit is prompted for interactively. Writes:
#   fleet.config.json  (gitignored) — read by nca/fleetconfig.py and the
#                      lane/collect scripts
#   docs/config.js     (gitignored) — window.NCA_CONFIG for the static pages
set -euo pipefail
cd "$(dirname "$0")"

BUCKET=""
PROJECT=""
SA_KEY="$HOME/.config/nca/submitter-key.json"
LOCAL_RUNS_DIR="$HOME/nca_runs"
REMOTE_HOSTS=()

while [ $# -gt 0 ]; do
  case "$1" in
    --bucket)         BUCKET=$2; shift 2;;
    --project)        PROJECT=$2; shift 2;;
    --sa-key)         SA_KEY=$2; shift 2;;
    --remote-host)    REMOTE_HOSTS+=("$2"); shift 2;;
    --local-runs-dir) LOCAL_RUNS_DIR=$2; shift 2;;
    -h|--help)        grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0;;
    *) echo "unknown flag: $1 (see --help)"; exit 1;;
  esac
done

if [ -z "$BUCKET" ]; then
  read -r -p "GCS bucket for published runs (public-read): " BUCKET
fi
if [ -z "$PROJECT" ]; then
  read -r -p "GCP project that owns the bucket: " PROJECT
fi
if [ ${#REMOTE_HOSTS[@]} -eq 0 ]; then
  read -r -p "Remote worker ssh aliases, space-separated (blank = local only): " -a REMOTE_HOSTS || true
fi
[ -n "$BUCKET" ] || { echo "a bucket is required"; exit 1; }
[ -n "$PROJECT" ] || { echo "a project is required"; exit 1; }

HOSTS_JSON=$(printf '%s\n' "${REMOTE_HOSTS[@]:-}" | python3 -c '
import json, sys
print(json.dumps([l.strip() for l in sys.stdin if l.strip()]))')

python3 - "$BUCKET" "$PROJECT" "$SA_KEY" "$LOCAL_RUNS_DIR" "$HOSTS_JSON" <<'PY'
import json, sys
bucket, project, sa_key, runs_dir, hosts = sys.argv[1:6]
cfg = {
    "bucket": bucket,
    "project": project,
    "sa_key_path": sa_key,
    "remote_hosts": json.loads(hosts),
    "local_runs_dir": runs_dir,
}
with open("fleet.config.json", "w") as f:
    json.dump(cfg, f, indent=2)
    f.write("\n")
with open("docs/config.js", "w") as f:
    f.write("window.NCA_CONFIG = " + json.dumps({"bucket": bucket}) + ";\n")
PY

echo "wrote fleet.config.json:"
python3 -m json.tool fleet.config.json
echo "wrote docs/config.js"
echo
echo "Next steps:"
echo "  1. ./serve.sh                # local gallery + dashboard at http://localhost:8000"
echo "  2. open Claude Code in this directory and ask it to train things —"
echo "     it will queue jobs (see CLAUDE.md), run lanes, and publish results"
echo "     to gs://$BUCKET"
