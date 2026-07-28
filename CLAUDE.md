# NCA fleet — instructions for Claude

You are the fleet manager for this NCA research repo. The user asks for
things to train; you queue jobs, run lanes, collect results, and point
them at the gallery. Read README.md for the run-dir contract and queue
format details.

## Configuration

`nca/fleetconfig.py` loads `fleet.config.json` (repo root, written by
`./init.sh`, gitignored): `bucket`, `project`, `sa_key_path`,
`remote_hosts` (ssh aliases), `local_runs_dir`. Missing keys fall back to
built-in defaults. Never hardcode bucket/project/host names in new code —
use `fleetconfig.load()` / `bucket_url()`.

## Adding jobs

Append lines to a queue file in `queues/` (create one per lane, e.g.
`queues/lane1.txt`; they are gitignored):

```
<run-name> <python-module> <args...>
```

- Run names must be unique (they become bucket prefixes). Trainers are
  `nca.train_*` modules; each accepts `--snap-dir` (the runner adds it —
  do NOT put it in the queue line) plus its own argparse flags. Check a
  module's flags before queueing: `grep add_argument nca/train_X.py`.
- Continue-training: name the derived run `<base>__<tag>` and pass
  `--source=<base>` so lineage is recorded in run.json.

## Launching lanes (detached)

One lane runs its queue sequentially. Launch detached so it survives the
session (use a background Bash task, or nohup):

```bash
nohup scripts/remote_queue.sh queues/lane1.txt [host] > logs/lane1.log 2>&1 &
nohup scripts/local_queue.sh  queues/lanepi.txt      > logs/lanepi.log 2>&1 &
```

- `[host]` is an ssh alias from `remote_hosts`; default is the first one.
  Run at most ~2 lanes per remote host (shared cores, jobs are niced).
- The remote runner holds the ssh session open for the job's lifetime and
  auto-resumes dropped sessions from checkpoints; don't kill it mid-job
  unless you mean to abandon the run.
- Editing a queue file after a lane started has no effect on the remote
  runner (it snapshots the file at launch); the local runner reads
  line-by-line, so appended lines DO get picked up.

## Collecting results

```bash
.venv/bin/python scripts/cse_collect.py --status-only   # what's running/done
.venv/bin/python scripts/cse_collect.py                 # rsync + upload to bucket
```

The remote runner collects automatically after each job; run collect
manually to publish mid-job progress. Lenia runs additionally need
`scripts/export_lenia_weights.py --run <name> --upload` to (re)generate
their browser weights.

## Where results live

- Bucket: `gs://<bucket>/<run-name>/` — public URLs
  `https://storage.googleapis.com/<bucket>/<run-name>/...`
- Local: `<local_runs_dir>/<run-name>/`
- Gallery/dashboard: `./serve.sh` then http://localhost:8000/lenia.html
  and /dashboard.html (static pages reading the bucket anonymously;
  bucket name comes from `docs/config.js`).

## Gotchas

- GCS public objects cache ~1h; the frontends cache-bust with `?t=`.
- Uploads need the service-account key at `sa_key_path` (or ADC).
- `submit_vertex_job.py` (GPU spot jobs on Vertex AI) still works but is
  retired in favour of the CPU lanes.
- Remote homes may have small quotas — the remote runner deletes each run
  dir after collection; don't stockpile runs remotely.
