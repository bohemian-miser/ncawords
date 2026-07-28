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
            
        if "TARGET_" not in content:
            content = re.sub(
                r'([ \t]+)(if step % log_every == 0 or step == steps - 1:)',
                lambda m: m.group(1) + m.group(2) + "\n" + \
m.group(1) + "    if snap_dir:\n" + \
m.group(1) + "        try:\n" + \
m.group(1) + "            import numpy as np\n" + \
m.group(1) + "            from PIL import Image\n" + \
m.group(1) + "            from pathlib import Path\n" + \
m.group(1) + "            tgt_img_arr = target_noisy[0, :3].cpu().clamp(0,1).permute(1,2,0).numpy() if 'target_noisy' in locals() else target[0, :3].cpu().clamp(0,1).permute(1,2,0).numpy()\n" + \
m.group(1) + "            Image.fromarray((tgt_img_arr * 255).astype(np.uint8)).resize((target.shape[3] * 8, target.shape[2] * 8), getattr(Image, 'Resampling', Image).NEAREST).save(Path(snap_dir) / f'TARGET_{step:05d}.png')\n" + \
m.group(1) + "        except Exception as e:\n" + \
m.group(1) + "            print(f'Fail target: {e}')",
                content,
                count=1
            )
            with open(fname, "w") as f:
                f.write(content)
                print(f"Patched {fname}")
