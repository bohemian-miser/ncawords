import torch
import time

def damage_mask_rect(n, h, w, device):
    y = torch.linspace(-1, 1, h, device=device)
    x = torch.linspace(-1, 1, w, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    
    xx = xx.unsqueeze(0).repeat(n, 1, 1)
    yy = yy.unsqueeze(0).repeat(n, 1, 1)
    
    cx = (torch.rand(n, 1, 1, device=device) - 0.5)
    cy = (torch.rand(n, 1, 1, device=device) - 0.5)
    
    r = torch.rand(n, 1, 1, device=device) * 0.3 + 0.1
    
    mask = ((xx - cx)**2 + (yy - cy)**2) > r**2
    return mask.unsqueeze(1)

t0 = time.time()
m = damage_mask_rect(1, 20, 68, "cpu")
print(f"Time taken: {time.time() - t0:.5f}s")
print(f"Mask shape: {m.shape}")
