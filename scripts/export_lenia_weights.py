"""Export a trained Lenia physics to weights.json for the web engine.

Evaluates every parameterised quantity to plain arrays so the JS engine
needs no knowledge of the sigmoid bounds — kernels [n,KS,KS], growth
centres/widths, gains, coupling matrix, and (for dyn variants) the basis
kernels + mixing-MLP matrices. Run either on a local snap dir or post-hoc
against bucket runs (downloads latest.pth, uploads weights.json).

Usage:
  python scripts/export_lenia_weights.py --run lenia-multik-dots [--upload]
  python scripts/export_lenia_weights.py --all [--upload]
"""
import argparse
import io
import json
import subprocess
import sys
import urllib.request

import torch

sys.path.insert(0, ".")
from nca.train_lenia import Lenia, sig, KS  # noqa: E402

BUCKET = "https://storage.googleapis.com/recipe-lanes-nca-jobs"


def export(variant, C, K, state_dict):
    if variant == "sphere":
        raise ValueError("sphere export needs a dedicated engine branch (todo)")
    model = Lenia(variant, C=C, K=K)
    model.load_state_dict(state_dict)
    model.eval()
    out = {"kind": "lenia", "variant": variant, "C": C, "K": K,
           "dt": model.dt, "ks": KS, "leak": 0.05}
    with torch.no_grad():
        bank = model.bank
        if variant in ("dyn1", "dynwave"):
            out["basis"] = model.basis[:, 0].numpy().round(6).tolist()
            w0, b0 = model.mix[0].weight, model.mix[0].bias
            w2, b2 = model.mix[2].weight, model.mix[2].bias
            out["mix"] = {
                "w0": w0[:, :, 0, 0].numpy().round(6).tolist(),
                "b0": b0.numpy().round(6).tolist(),
                "w2": w2[:, :, 0, 0].numpy().round(6).tolist(),
                "b2": b2.numpy().round(6).tolist(),
            }
        else:
            out["kernels"] = bank.kernels(torch.device("cpu")) \
                .numpy().round(6).tolist()
        if variant == "wave" or variant == "dynwave":
            mu = sig(bank.mu, -0.6, 1.0)
        else:
            mu = sig(bank.mu, 0.0, 1.0)
        out["mu"] = mu.numpy().round(6).tolist()
        out["sg"] = sig(bank.sg, 0.02, 0.35).numpy().round(6).tolist()
        out["h"] = torch.tanh(bank.h).numpy().round(6).tolist()
        if variant == "sharedk":
            out["H"] = torch.tanh(model.H).numpy().round(6).tolist()
    return out


def run_meta(run):
    with urllib.request.urlopen(f"{BUCKET}/{run}/run.json") as r:
        d = json.load(r)
    a = d.get("args", {})
    return a.get("variant"), a.get("C", 1), a.get("K", 3)


def process(run, upload):
    variant, C, K = run_meta(run)
    with urllib.request.urlopen(f"{BUCKET}/{run}/latest.pth") as r:
        sd = torch.load(io.BytesIO(r.read()), map_location="cpu",
                        weights_only=True)
    out = export(variant, C, K, sd)
    path = f"/tmp/{run}-weights.json"
    with open(path, "w") as f:
        json.dump(out, f)
    print(f"{run}: exported {variant} C={C} K={K} -> {path}")
    if upload:
        subprocess.run(["gcloud", "storage", "cp", path,
                        f"gs://recipe-lanes-nca-jobs/{run}/weights.json",
                        "-q"], check=True)
        print(f"{run}: uploaded")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run")
    p.add_argument("--all", action="store_true")
    p.add_argument("--upload", action="store_true")
    a = p.parse_args()
    if a.all:
        with urllib.request.urlopen(
                "https://storage.googleapis.com/storage/v1/b/"
                "recipe-lanes-nca-jobs/o?prefix=lenia-&delimiter=/"
                "&fields=prefixes&maxResults=1000") as r:
            runs = [x.rstrip("/") for x in json.load(r).get("prefixes", [])]
        for run in runs:
            try:
                process(run, a.upload)
            except Exception as e:
                print(f"{run}: SKIP ({str(e)[:60]})")
    else:
        process(a.run, a.upload)
