import os

files = [
    "nca/train_web_method1.py",
    "nca/train_method2.py",
    "nca/train_web_method4.py",
    "nca/train_web_method5.py",
    "nca/train_web_9_line.py"
]

inject_imports = """import os
import numpy as np
from PIL import Image
from pathlib import Path
"""

for fname in files:
    if os.path.exists(fname):
        with open(fname, "r") as f:
            content = f.read()
        
        content = inject_imports + content
        with open(fname, "w") as f:
            f.write(content)
        print(f"Added imports to {fname}")
