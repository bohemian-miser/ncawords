import os
import re

files = [
    "nca/train_web_method1.py",
    "nca/train_method2.py",
    "nca/train_web_method4.py",
    "nca/train_web_method5.py",
    "nca/train_web_9_line.py",
]

for fname in files:
    if os.path.exists(fname):
        with open(fname, "r") as f:
            content = f.read()
        
        # Replace the seed assignment with a check for NOISE_START
        original = r"import os\s+x\[:, 3:, cy, cx\] = 1\.0\s+return x"
        replacement = """import os
    if os.getenv("NOISE_START") == "1":
        x = torch.rand_like(x)
    else:
        x[:, 3:, cy, cx] = 1.0
    return x"""
        content = re.sub(original, replacement, content)
        
        with open(fname, "w") as f:
            f.write(content)
        print(f"Patched NOISE_START in {fname}")
