import os
import glob

files = [
    "nca/train_web_method1.py",
    "nca/train_method2.py",
    "nca/train_web_method4.py",
    "nca/train_web_method5.py",
    "nca/train_web_9_line.py"
]

target_loss_line = "loss = F.mse_loss(to_rgba(x), target)"

noisy_target_logic = """
        if os.getenv("NOISE_SEED") == "1":
            # Noise schedule: 1.0 at step 0 -> 0.0 at half of training (e.g. 4000)
            noise_idx = max(0.0, 1.0 - (step / (steps * 0.5)))
            if noise_idx > 0:
                target_noisy = target * (1.0 - noise_idx) + torch.rand_like(target) * noise_idx
                loss = F.mse_loss(to_rgba(x), target_noisy)
            else:
                loss = F.mse_loss(to_rgba(x), target)
        else:
            loss = F.mse_loss(to_rgba(x), target)
"""

for fname in files:
    if os.path.exists(fname):
        with open(fname, "r") as f:
            content = f.read()
            
        # Avoid double-patching
        if "noise_idx =" not in content:
            # Note: Need to match exact indentation. Searching for the line exactly
            # But the indentation might not be exactly 8 spaces. Let's find it via regex.
            import re
            content = re.sub(
                r'([ \t]+)loss = F\.mse_loss\(to_rgba\(x\), target\)',
                lambda m: noisy_target_logic.replace('\n', '\n' + m.group(1)).strip('\n'),
                content,
                count=1
            )
            with open(fname, "w") as f:
                f.write(content)
                print(f"Patched {fname}")
