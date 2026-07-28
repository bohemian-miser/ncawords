# NCA fleet — Growing Neural Cellular Automata

A cloneable research fleet for training Neural Cellular Automata (Distill's
[Growing NCA](https://distill.pub/2020/growing-ca/) lineage plus Lenia-style
continuous CAs) on cheap CPUs — a local machine and any ssh-reachable
workers — publishing every run to a public GCS bucket that a static
gallery/dashboard reads directly.

## Quick start

```bash
git clone <this-repo> && cd <repo>
./init.sh --bucket my-bucket --project my-gcp-project \
          [--sa-key ~/.config/nca/key.json] [--remote-host alias ...]
./serve.sh          # gallery at http://localhost:8000/lenia.html
```

Then open Claude Code in this directory and say what you want to train —
CLAUDE.md teaches it how to queue jobs, run lanes, and collect results.

`init.sh` writes `fleet.config.json` (gitignored; read by
`nca/fleetconfig.py`) and `docs/config.js` (`window.NCA_CONFIG` for the
static pages). Everything falls back to built-in defaults when
unconfigured. The bucket must be public-read (with CORS) for the
gallery/dashboard; the service-account key only needs write access for
uploads.

## Queues and lanes

A queue file is one job per line (blank lines and `#` comments skipped):

```
<run-name> <python-module> <args...>
```

See `queues/example.txt`. Lane runners execute a queue sequentially:

```bash
scripts/remote_queue.sh queues/mylane.txt [host]  # on a remote worker
scripts/local_queue.sh  queues/mylane.txt         # on this machine
```

(`remote_queue.sh`/`local_queue.sh` are aliases for `cse_queue.sh` /
`pi_queue.sh`.) `[host]` defaults to the first `remote_hosts` entry. The
remote runner ships `nca/` to the worker, holds the ssh session for the
job's lifetime, retries dropped sessions (jobs resume from checkpoints),
then collects, uploads and deletes the remote run dir. The local runner
trains niced on this machine into `local_runs_dir`.
`scripts/cse_collect.py [--status-only]` rsyncs remote runs down and
uploads anything newer than its bucket copy.

## Run-dir contract

Each run is a directory `<name>/` (locally under `local_runs_dir`, and
mirrored to `gs://<bucket>/<name>/`) containing:

- `run.json` — manifest, updated every log interval: `text`, `module`,
  `args`, `code_sha` (git SHA of the shipped code), `source_run` (parent
  run when continuing), `channel_n`/`hidden_n`, `seed_type`,
  `steps_total`, `step`, `losses` `[[step, loss], ...]`, `history`
  (per-interval dicts with `step`, `loss` and extras such as `ca_steps`,
  phase, aux metrics), `tags`, timestamps.
- `weights.json` — playground-ready weights the web viewer loads.
- `latest.pth` / `ckpt.pth` — model state / resumable checkpoint.
- snapshot PNGs, prefixed by kind: `COMP_*` (composite board at a step),
  `START_*` (initial state), `TARGET*` (training target), `KERNEL_*`
  and `COUPLING_*` (Lenia kernels / coupling matrix).
- `code.tgz` — the exact `nca/` source tree the job ran.

**Continue-training convention:** a derived run is named
`<base>__<tag>` (double underscore) and passes `--source=<base>` so its
`run.json` records `source_run` and the lineage stays traceable.

**Reproduce any run:** download `gs://<bucket>/<run>/code.tgz`, unpack it
onto `PYTHONPATH`, and re-run the module with the `args` recorded in that
run's `run.json`.

## Layout

```
nca/
  model.py       # the CA update rule (PyTorch): perception -> 1x1 MLP ->
                 # stochastic residual update -> alive masking
  fleetconfig.py # fleet.config.json loader (bucket/project/hosts/dirs)
  train.py       # train one letter model (sample pool + damage; exports JSON)
  train_word.py  # ONE model grows a whole string on one wide grid: one seed
                 # per letter, a 5-bit letter code in hidden channels 4-8
  train_lenia.py # continuous (Lenia-style) CA variants
  ocr_eval.py    # grow each letter from seed, OCR with tesseract (psm 10)
  ocr_word.py    # grow a word model, OCR the whole picture as a word (psm 8)
  make_golden.py # deterministic rollout dump for verifying the JS engine
  train_all.py   # multi-process orchestrator with per-letter OCR gates
scripts/
  remote_queue.sh / local_queue.sh   # lane runners (see above)
  cse_collect.py   # pull remote runs + upload to the bucket
  ladder.sh        # escalation: singles -> double "GO" -> word "GROW",
                   # each rung OCR-gated
  build_report.py  # aggregate OCR reports + weights index for the site
docs/
  index.html / style.css / main.js   # the article
  lenia.html                         # run gallery (reads the bucket)
  dashboard.html                     # training dashboard
  nca.js                             # browser engine (WebGL2 + CPU fallback)
  API.md                             # engine <-> page contract
  test/test_engine.mjs               # node test vs Python golden rollout
queues/          # lane queue files (gitignored except example.txt)
weights/  grown/  ocr/  logs/        # training artifacts (synced into docs/)
```

## Model

Distill's architecture, shrunk for CPU training: letters use 12 channels
(RGB, alpha, 8 hidden), 36 perception features (identity + Sobel x/y),
64 hidden units — ~2.4k parameters per letter on a 32×32 grid. Word models
use 16 channels / 80 hidden; seeds are distinguished only by 5 code numbers
in their initial hidden state, so one rule grows different glyphs.

Reference implementation: [google-research/self-organising-systems](https://github.com/google-research/self-organising-systems)
(Apache 2.0). This repo is an independent PyTorch/JS port trained from
scratch on CPU.

## Direct usage

```bash
.venv/bin/python -m nca.train --char A            # train one letter
.venv/bin/python -m nca.ocr_eval weights/0041.json  # grow + OCR it
.venv/bin/python -m nca.train_word --text GO      # whole string, one grid
.venv/bin/python -m nca.ocr_word weights/word_GO.json
node docs/test/test_engine.mjs                    # JS engine vs golden
./serve.sh                                        # view the site locally
```
