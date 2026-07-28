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
            
        content = content.replace("            import numpy as np\n", "            import numpy as fallback_np\n")
        content = content.replace("            from PIL import Image\n", "            from PIL import Image as fallback_Image\n")
        content = content.replace("Image.fromarray", "fallback_Image.fromarray")
        content = content.replace("getattr(Image, ", "getattr(fallback_Image, ")
        
        with open(fname, "w") as f:
            f.write(content)
        print(f"Fixed {fname}")
