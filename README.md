# ncawords — Growing Neural Cellular Automata for Text

A reimplementation of [Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca/)
(Mordvintsev et al., Distill 2020) where the organisms are **letter glyphs and
whole words**: tiny neural CAs trained so a single seed pixel grows into a
character — verified by Tesseract OCR — plus an interactive distill-style
article that runs the trained models live in the browser.

Reference implementation: [google-research/self-organising-systems](https://github.com/google-research/self-organising-systems)
(Apache 2.0). This repo is an independent PyTorch/JS port trained from scratch
on a Raspberry Pi 5 (CPU only).

## Layout

```
nca/
  model.py       # the CA update rule (PyTorch): perception -> 1x1 MLP ->
                 # stochastic residual update -> alive masking
  train.py       # train one letter model (sample pool + damage; exports JSON)
  train_word.py  # ONE model grows a whole string on one wide grid: one seed
                 # per letter, a 5-bit letter code in hidden channels 4-8
  ocr_eval.py    # grow each letter from seed, OCR with tesseract (psm 10)
  ocr_word.py    # grow a word model, OCR the whole picture as a word (psm 8)
  make_golden.py # deterministic rollout dump for verifying the JS engine
  train_all.py   # multi-process orchestrator with per-letter OCR gates
scripts/
  ladder.sh        # escalation: singles -> double "GO" -> word "GROW",
                   # each rung OCR-gated
  build_report.py  # aggregate OCR reports + weights index for the site
docs/
  index.html / style.css / main.js   # the article
  nca.js                             # browser engine (WebGL2 + CPU fallback)
  API.md                             # engine <-> page contract
  test/test_engine.mjs               # node test vs Python golden rollout
weights/  grown/  ocr/  logs/        # training artifacts (synced into docs/)
```

## Model

Distill's architecture, shrunk for CPU training: letters use 12 channels
(RGB, alpha, 8 hidden), 36 perception features (identity + Sobel x/y),
64 hidden units — ~2.4k parameters per letter on a 32×32 grid. Word models
use 16 channels / 80 hidden; seeds are distinguished only by 5 code numbers
in their initial hidden state, so one rule grows different glyphs.

## Usage

```bash
.venv/bin/python -m nca.train --char A            # train one letter
.venv/bin/python -m nca.ocr_eval weights/0041.json  # grow + OCR it
.venv/bin/python -m nca.train_word --text GO      # whole string, one grid
.venv/bin/python -m nca.ocr_word weights/word_GO.json
node docs/test/test_engine.mjs                    # JS engine vs golden
python3 -m http.server -d docs 8000              # view the article
```
