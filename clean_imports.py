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

        # REVERT fix_local_import.py changes
        content = content.replace("import pathlib", "")
        content = content.replace("pathlib.Path(snap_dir)", "Path(snap_dir)")
        
        # REVERT fix_imports.py changes
        content = content.replace("import numpy as fallback_np\n", "\n")
        content = content.replace("from PIL import Image as fallback_Image\n", "\n")
        content = content.replace("fallback_Image.fromarray", "Image.fromarray")
        content = content.replace("getattr(fallback_Image, ", "getattr(Image, ")
        
        # Delete original problem imports too if any are left
        content = content.replace("            import numpy as np\n", "")
        content = content.replace("            from PIL import Image\n", "")
        content = content.replace("            from pathlib import Path\n", "")
        
        with open(fname, "w") as f:
            f.write(content)
        print(f"Cleaned {fname}")
