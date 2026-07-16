import os
import re

files = [
    "nca/train_method2.py",
    "nca/train_web_method4.py",
    "nca/train_web_method5.py",
    "nca/train_web_9_line.py",
]

for fname in files:
    if os.path.exists(fname):
        with open(fname, "r") as f:
            lines = f.readlines()
        
        for i, line in enumerate(lines):
            line_str = line
            if "torch.save(model.state_dict()" in line_str:
                # Get whitespace from this line
                leading_spaces = len(line_str) - len(line_str.lstrip())
                # The next line is `                    save_word_png(` Which is causing error
                if i+1 < len(lines) and "save_word_png(" in lines[i+1]:
                    # Fix the indentation of the next line!
                    lines[i+1] = (" " * leading_spaces) + lines[i+1].lstrip()
        
        with open(fname, "w") as f:
            f.writelines(lines)
        print(f"Fixed indentation in {fname}")
