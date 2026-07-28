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
        
        # Replace the problematic local import inside train loop
        content = content.replace("            from pathlib import Path\n", "            import pathlib\n")
        content = content.replace("Path(snap_dir) / f'TARGET_", "pathlib.Path(snap_dir) / f'TARGET_")
        
        with open(fname, "w") as f:
            f.write(content)
        print(f"Fixed {fname}")
