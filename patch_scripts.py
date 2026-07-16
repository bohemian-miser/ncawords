import os
import glob

files = [
    "nca/train_web_method1.py",
    "nca/train_method2.py",
    "nca/train_web_method4.py",
    "nca/train_web_method5.py",
    "nca/train_web_9_line.py"
]

def patch_file(path):
    with open(path, "r") as f:
        content = f.read()

    # 1. target image color inversion
    old_target_img = "tgt_img_arr = target_noisy[0, :3].cpu().clamp(0,1).permute(1,2,0).numpy() if 'target_noisy' in locals() else target[0, :3].cpu().clamp(0,1).permute(1,2,0).numpy()"
    new_target_img = """tgt_t = target_noisy if 'target_noisy' in locals() else target
                    a = tgt_t[0, 3:4].cpu()
                    rgb = tgt_t[0, :3].cpu()
                    tgt_img_arr = (1.0 - a + rgb).clamp(0,1).permute(1,2,0).numpy()"""
    content = content.replace(old_target_img, new_target_img)

    # 2. Add noise initialization before `for step in range(steps):`
    old_loop_start = "for step in range(steps):"
    new_loop_start = """noise_idx = 0.85
    recent_losses = []
    for step in range(steps):"""
    if "noise_idx = 0.85" not in content:
        content = content.replace(old_loop_start, new_loop_start)

    # 3. Replace noise curriculum inside loop
    old_noise_logic = """        if os.getenv("NOISE_SEED") == "1":
            # Noise schedule: 1.0 at step 0 -> 0.0 at half of training (e.g. 4000)
            noise_idx = max(0.0, 0.85 - (step / (steps * 0.5)))
            if noise_idx > 0:
                target_noisy = target * (1.0 - noise_idx) + torch.rand_like(target) * noise_idx
                loss = F.mse_loss(to_rgba(x), target_noisy)
            else:
                loss = F.mse_loss(to_rgba(x), target)
        else:
            loss = F.mse_loss(to_rgba(x), target)"""
            
    old_noise_logic2 = """        if os.getenv("NOISE_SEED") == "1":
            # Noise schedule: 1.0 at step 0 -> 0.0 at half of training (e.g. 4000)
            noise_idx = max(0.0, 0.85 - (step / (steps * 0.5)))
            if noise_idx > 0:
                target_noisy = target * (1.0 - noise_idx) + torch.rand_like(target) * noise_idx
                loss = F.mse_loss(to_rgba(x), target_noisy)
            else:
                loss = F.mse_loss(to_rgba(x), target)
                if 'target_noisy' in locals(): del target_noisy
        else:
            loss = F.mse_loss(to_rgba(x), target)"""

    new_noise_logic = """        if noise_idx > 0:
            target_noisy = target * (1.0 - noise_idx) + torch.rand_like(target) * noise_idx
            loss = F.mse_loss(to_rgba(x), target_noisy)
        else:
            loss = F.mse_loss(to_rgba(x), target)
            if 'target_noisy' in locals():
                del target_noisy"""
    content = content.replace(old_noise_logic, new_noise_logic)
    content = content.replace(old_noise_logic2, new_noise_logic)
    
    # 4. Adaptive curriculum append after pool.commit
    old_commit = "pool.commit(idx, x.cpu())"
    new_commit = """pool.commit(idx, x.cpu())

        recent_losses.append(loss.item())
        if len(recent_losses) > 100:
            recent_losses.pop(0)
            
        if len(recent_losses) == 100:
            avg_loss = sum(recent_losses) / 100.0
            if avg_loss < 0.015:
                noise_idx = max(0.0, noise_idx - 0.05)
                recent_losses.clear()
            elif avg_loss > 0.03:
                noise_idx = min(0.85, noise_idx + 0.01)
                recent_losses.clear()"""
    if "recent_losses.append" not in content:
        content = content.replace(old_commit, new_commit)
        
    # 5. Remove any os.getenv("NOISE_SEED") blocks in seed generation
    old_seed_cond_1 = """    if os.getenv("NOISE_SEED") == "1":
        x = torch.rand_like(x)
    else:
        x[:, 3:, cy, cx] = 1.0"""
    new_seed_cond_1 = "    x[:, 3:, cy, cx] = 1.0"
    content = content.replace(old_seed_cond_1, new_seed_cond_1)
    
    # also some files might have it in save_word_png
    old_save_cond_1 = """        if os.getenv("NOISE_SEED") == "1":
            tgt = render_word_method1(text, 12)
            x = make_single_seed(text, channel_n, tgt=tgt).to(device)
            x = torch.rand_like(x)
        else:
            tgt = render_word_method1(text, 12)
            x = make_single_seed(text, channel_n, tgt=tgt).to(device)"""
    new_save_cond_1 = """        tgt = render_word_method1(text, 12)
        x = make_single_seed(text, channel_n, tgt=tgt).to(device)"""
    content = content.replace(old_save_cond_1, new_save_cond_1)
    
    # for method5 it calls render_word
    old_save_cond_5 = """        if os.getenv("NOISE_SEED") == "1":
            tgt = render_word_method5(text, 12)
            x = make_single_seed(text, channel_n, tgt=tgt).to(device)
            x = torch.rand_like(x)
        else:
            tgt = render_word_method5(text, 12)
            x = make_single_seed(text, channel_n, tgt=tgt).to(device)"""
    new_save_cond_5 = """        tgt = render_word_method5(text, 12)
        x = make_single_seed(text, channel_n, tgt=tgt).to(device)"""
    content = content.replace(old_save_cond_5, new_save_cond_5)

    old_save_cond_4 = """        if os.getenv("NOISE_SEED") == "1":
            tgt = render_word_method4(text, 12)
            x = make_single_seed(text, channel_n, tgt=tgt).to(device)
            x = torch.rand_like(x)
        else:
            tgt = render_word_method4(text, 12)
            x = make_single_seed(text, channel_n, tgt=tgt).to(device)"""
    new_save_cond_4 = """        tgt = render_word_method4(text, 12)
        x = make_single_seed(text, channel_n, tgt=tgt).to(device)"""
    content = content.replace(old_save_cond_4, new_save_cond_4)
    
    old_save_cond_9 = """        if os.getenv("NOISE_SEED") == "1":
            tgt = render_word_method2(text, 12)
            x = make_single_seed(text, channel_n, tgt=tgt).to(device)
            x = torch.rand_like(x)
        else:
            tgt = render_word_method2(text, 12)
            x = make_single_seed(text, channel_n, tgt=tgt).to(device)"""
    new_save_cond_9 = """        tgt = render_word_method2(text, 12)
        x = make_single_seed(text, channel_n, tgt=tgt).to(device)"""
    content = content.replace(old_save_cond_9, new_save_cond_9)

    with open(path, "w") as f:
        f.write(content)

for f in files:
    patch_file(f)
    print(f"Patched {f}")
