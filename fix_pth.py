import os

files = [
    "nca/train_web_method1.py",
    "nca/train_method2.py",
    "nca/train_web_method4.py",
    "nca/train_web_method5.py",
    "nca/train_web_9_line.py",
    "nca/train_cloud.py"
]

corrupted = "def torch.save(model.state_dict(), str(Path(snap_dir) / 'latest.pth'))\\n                    save_word_png("
original = "def save_word_png("

for fname in files:
    if os.path.exists(fname):
        with open(fname, "r") as f:
            content = f.read()
        
        # It was replaced globally, so we have both:
        # 1. The definition got corrupted
        # 2. The call inside the loop got replaced with torch.save(...)\nsave_word_png(
        
        # We need to target the definition specifically and fix it.
        # Wait, the best way is to view the exact corrupted text...
        
        # Actually I can just write a regex:
        import re
        content = re.sub(
            r"def torch\.save\(model\.state_dict\(\), str\(Path\(snap_dir\) / 'latest\.pth'\)\)\s+save_word_png\(",
            "def save_word_png(",
            content
        )
        
        with open(fname, "w") as f:
            f.write(content)
        print(f"Fixed definition in {fname}")
