"""Per-run metadata and browser-ready weight export.

Training jobs write two files next to their snapshots so a static frontend
can consume runs straight from the public GCS bucket with no backend:

- run.json     manifest: what was trained, with what config, loss history,
               and progress — updated at every log interval.
- weights.json playground-ready weights in the docs/weights/*.json format
               the WebGL viewer already loads.
"""
import json
import time
from pathlib import Path

from nca.model import NCA
from nca.train import export_weights


class RunMeta:
    def __init__(self, snap_dir, text, module, args, channel_n, hidden_n,
                 seed_type, steps_total, device):
        self.path = Path(snap_dir) / "run.json" if snap_dir else None
        self.d = {
            "text": text,
            "module": module,
            "args": args,
            "channel_n": channel_n,
            "hidden_n": hidden_n,
            "seed_type": seed_type,
            "steps_total": steps_total,
            "device": str(device),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "updated_at": None,
            "step": -1,
            "losses": [],
        }
        # A preempted-and-resumed job should extend the history, not clobber it.
        if self.path and self.path.exists():
            try:
                prev = json.loads(self.path.read_text())
                self.d["started_at"] = prev.get("started_at", self.d["started_at"])
                self.d["losses"] = prev.get("losses", [])
            except Exception:
                pass
        self._write()

    def log(self, step, loss, **extra):
        self.d["step"] = step
        self.d["losses"].append([step, round(float(loss), 6)])
        self.d["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.d.update(extra)
        self._write()

    def _write(self):
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.d))


def export_run_weights(model, snap_dir, text, glyph=12, grid_w=100, grid_h=40):
    """Write weights.json (viewer format) next to the run's snapshots."""
    if not snap_dir:
        return
    cpu_model = NCA(model.channel_n, hidden_n=model.fc0.out_channels)
    cpu_model.load_state_dict({k: v.cpu() for k, v in model.state_dict().items()})
    out = Path(snap_dir) / "weights.json"
    export_weights(cpu_model, text, None, glyph, out)
    d = json.loads(out.read_text())
    d.update({"kind": "word", "text": text, "grid_w": grid_w, "grid_h": grid_h,
              "grid": None})
    out.write_text(json.dumps(d))
