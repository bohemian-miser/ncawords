import glob
import re

files = ["nca/train_web_method1.py", "nca/train_method2.py", "nca/train_web_method4.py", "nca/train_web_method5.py", "nca/train_web_9_line.py"]

for f in files:
    with open(f, "r") as file:
        content = file.read()
    
    # Standardize argparse portion
    main_pattern = re.compile(r'if __name__ == "__main__":.*', re.DOTALL)
    
    new_main = """if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--text", required=True)
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--out", default=None)
    p.add_argument("--snap-dir", default=None)
    p.add_argument("--seed-type", default="single")
    a = p.parse_args()
    
    train(a.text, steps=a.steps, log_every=a.log_every, out=a.out, snap_dir=a.snap_dir, seed_type=a.seed_type)
"""
    if main_pattern.search(content):
        content = main_pattern.sub(new_main, content)
        with open(f, "w") as file:
            file.write(content)
        print(f"Standardized {f}")
    else:
        print(f"Could not find __main__ in {f}")
