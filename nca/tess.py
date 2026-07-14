"""Minimal tesseract wrapper.

Uses pytesseract when available, else shells out to the tesseract binary
(some hosts block pip installs; the binary is all we actually need).
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

try:
    import pytesseract as _pt
except ImportError:
    _pt = None


def image_to_string(img, config=""):
    if _pt is not None:
        return _pt.image_to_string(img, config=config)
    if not shutil.which("tesseract"):
        raise RuntimeError("neither pytesseract nor the tesseract binary found")
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "in.png"
        img.save(src)
        cmd = ["tesseract", str(src), "stdout"] + config.split()
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"tesseract failed: {r.stderr.strip()[:200]}")
        return r.stdout
