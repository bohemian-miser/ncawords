# NCA Fleet

Train **neural cellular automata** — tiny models where every pixel runs the same
little rule, and structure grows from a single seed — then watch them in a
browser gallery. Words, emoji, textures, Game-of-Life physics, Lenia kernels.

### 🌐 Live Interactive Demos (GitHub Pages)

- **[Live Runs Gallery](https://bohemian-miser.github.io/ncawords/lenia.html)** — Unified gallery listing all runs with training snapshots, loss curves, and live in-browser simulations.
- **[Dynamic Kernel Interpolation Demo](https://bohemian-miser.github.io/ncawords/demo_dynkernel.html)** — Compare and interpolate between dynamic-kernel & fixed-kernel models (Fire, Rocket, Heart, Gecko, Alien, Butterfly, Hex, Dots) with real-time kernel inspector HUD and pure noise nucleation.
- **[Interactive Article & Research Demos](https://bohemian-miser.github.io/ncawords/)** — Interactive writeup on text NCAs, feature modulation, multi-species coexistence, and gradient steering.

You say what you want in chat; a coding agent writes and launches the
experiments; results stream to your bucket and appear in the gallery, live.

---

## Quickstart

```bash
git clone <this-repo> && cd ncawords
./init.sh --bucket YOUR_GCS_BUCKET --project YOUR_GCP_PROJECT
./serve.sh                      # gallery at http://localhost:8791/lenia.html
```

Then open an agent (Claude Code, etc.) in this directory and just say what you
want:

> *"Train a growing NCA that writes HELLO, then harden it with damage and noise."*

`CLAUDE.md` tells the agent how the fleet works, so it can queue jobs, run them,
and collect results without further instruction.

Requirements: Python 3.10+, a GCS bucket (public-read if you want the gallery to
load without auth), and a service-account key or `gcloud` login.

---

## Running jobs yourself

A queue file is one job per line — `<run-name> <python-module> <args...>`:

```
mytext-r0  nca.train_ladder_seed --text=HELLO --steps=20000 --scaffold=3line
smiley     nca.train_emoji_vanilla --emoji=1f642 --label=smiley --damage-p=0.5
```

```bash
scripts/local_queue.sh  queues/mine.txt          # run here
scripts/remote_queue.sh queues/mine.txt myhost   # or on an ssh host
python3 scripts/cse_collect.py                   # pull results -> bucket
```

Lanes run one job at a time and resume from checkpoints if interrupted, so
stopping and restarting is safe.

## What a run looks like

Every run gets a folder in your bucket:

| File | What it is |
|---|---|
| `run.json` | config, loss history, timings, `code_sha`, `source_run`, `ca_steps` |
| `weights.json` | browser-runnable model — powers the gallery's live widget |
| `COMP_*.png` | snapshots over training (plus `START_`, `TARGET`, `KERNEL_`, …) |
| `code.tgz` | the exact source that produced this run |
| `ckpt.pth` | checkpoint for resuming or continuing |

Because the code ships *with* the run, anything you like is reproducible:
download `code.tgz`, read the args from `run.json`, run it again.

## Continuing a trained model

Point a new run at an existing one and keep training — with damage, noise,
longer horizons, whatever:

```
mytext-r0__tough  nca.train_noisefester --source=mytext-r0 --mix
```

The `<base>__<tag>` name keeps the lineage obvious, and the gallery links the
child back to its parent.

## The gallery & demos

Visit the **[Live Web App](https://bohemian-miser.github.io/ncawords/lenia.html)** or run `./serve.sh`:
- **Runs (`lenia.html`)**: The unified gallery listing every run with snapshots, loss curves, learned kernels, and live in-browser simulations you can seed, damage, and interact with (supersedes the legacy Playground).
- **Dynamic Kernel Interpolation (`demo_dynkernel.html`)**: Real-time spatial interpolation, continuous morphing, and pure noise nucleation across dynamic-kernel and fixed-kernel models with live basis HUD.
- **Interactive Demos**: Multi-species coexistence (`demo_coexist.html`), target steering (`demo_steer.html`), and feature modulation (`demo_modulation.html`).

## Layout

```
nca/       training modules (the experiments)
scripts/   queue runners, collection, exports, scoring
docs/      the static site (gallery, demos, engines)
queues/    your job lists (gitignored)
```

Config lives in `fleet.config.json` (gitignored, written by `init.sh`) and
`docs/config.js` for the web pages; everything falls back to defaults when
unset.
