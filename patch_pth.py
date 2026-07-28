import os

files = [
    "nca/train_web_method1.py",
    "nca/train_method2.py",
    "nca/train_web_method4.py",
    "nca/train_web_method5.py",
    "nca/train_web_9_line.py",
    "nca/train_cloud.py"
]

for fname in files:
    if os.path.exists(fname):
        with open(fname, "r") as f:
            content = f.read()
        
        # Add a line to save latest.pth right after it prints the loss (inside the log_every block)
        # Search for: print(f"[{text}] step {step} loss ...
        # Since the print statement spans multiple lines, we can look for `if snap_dir:`
        
        # In cloud.py it's:
        #        if step % log_every == 0 or step == steps - 1:
        #            print(f"[{text}] step {step} loss {loss.item():.5f} "
        #                  f"({(time.time() - t0):.1f}s)", flush=True)
        #            if snap_dir:
        #                try:
        
        # Let's cleanly inject it where `save_word_png` is called
        if "save_word_png(" in content:
            content = content.replace(
                "save_word_png(", 
                "torch.save(model.state_dict(), str(Path(snap_dir) / 'latest.pth'))\n                    save_word_png("
            )
        else:
            print(f"Could not find save_word_png in {fname}")

        with open(fname, "w") as f:
            f.write(content)
        print(f"Added latest.pth saving to {fname}")
