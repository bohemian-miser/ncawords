import re

with open("nca/train_web_method1.py", "r") as f:
    text = f.read()

# Add saving logic block
save_code = """
def save_word_png(model, text, channel_n, path, seed_type, device, n_steps=120):
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        if os.getenv("NOISE_SEED") == "1":
            tgt = render_word_method1(text, 12)
            x = make_single_seed(text, channel_n, tgt=tgt).to(device)
            x = torch.rand_like(x)
        else:
            tgt = render_word_method1(text, 12)
            x = make_single_seed(text, channel_n, tgt=tgt).to(device)
        x = model(x, steps=n_steps)
    img = to_rgba(x)[0, :3].cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    im = Image.fromarray((img * 255).astype(np.uint8))
    rez_method = getattr(Image, "Resampling", Image).NEAREST
    im = im.resize((im.width * 8, im.height * 8), rez_method)
    im.save(path)

if __name__ == "__main__":
"""

text = text.replace('if __name__ == "__main__":', save_code)

# Add save call to train loop
train_loop = """        if step % log_every == 0 or step == steps - 1:
            print(f"[{text}] step {step} loss {loss.item():.5f} "
                  f"({(time.time() - t0):.1f}s)", flush=True)"""

train_loop_new = """        if step % log_every == 0 or step == steps - 1:
            print(f"[{text}] step {step} loss {loss.item():.5f} "
                  f"({(time.time() - t0):.1f}s)", flush=True)
            if snap_dir:
                try:
                    save_word_png(model, text, channel_n,
                                  str(Path(snap_dir) / f"COMP_{step:05d}.png"),
                                  seed_type=seed_type, device=device)
                except Exception as e:
                    print(f"Save failed: {e}")"""
                    
text = text.replace(train_loop, train_loop_new)

with open("nca/train_web_method1.py", "w") as f:
    f.write(text)

