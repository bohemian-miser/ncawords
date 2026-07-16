import os

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
        
        # Change noise schedule to start at 0.85 instead of 1.0, so the underlying target is 15% visible
        content = content.replace(
            "noise_idx = max(0.0, 1.0 - (step / (steps * 0.5)))", 
            "noise_idx = max(0.0, 0.85 - (step / (steps * 0.5)))"
        )
        
        with open(fname, "w") as f:
            f.write(content)
        print(f"Diminished noise starting point in {fname}")
