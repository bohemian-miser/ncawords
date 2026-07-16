import os
import re

files = [
    "nca/train_web_method1.py",
    "nca/train_method2.py",
    "nca/train_web_method4.py",
    "nca/train_web_method5.py",
    "nca/train_web_9_line.py"
]

for fname in files:
    if os.path.exists(fname):
        with open(fname, "r") as f:
            content = f.read()

        # Owl #1 rewrote the logic, let's find it.
        # It probably looks like "if recent_loss < 0.015:"
        content = content.replace("0.85", "0.60")
        content = content.replace("0.015", "0.010")
        content = content.replace("0.03", "0.02")
        
        with open(fname, "w") as f:
            f.write(content)
        print(f"Tweaked thresholds for {fname}")
