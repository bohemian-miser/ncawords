import torch
torch.set_num_threads(1)
import numpy as np
import torch.nn.functional as F
from PIL import Image
from nca.train_web_method4 import render_word, word_geometry
from nca.model import NCA, to_rgba
from nca.train import SamplePool

def make_single_seed(text, channel_n=16, n=1, tgt=None):
    print("In make_single_seed")
    w, h = word_geometry(text)
    x = torch.zeros(n, channel_n, h, w)
    
    cy, cx = h // 2, w // 2
    if tgt is not None:
        print("tgt is not None")
        y_ids, x_ids = np.where(tgt[3] > 0.5)
        print(f"y_ids len: {len(y_ids)}")
        if len(y_ids) > 0:
            distances = (y_ids - cy)**2 + (x_ids - cx)**2
            best_idx = np.argmin(distances)
            cy, cx = y_ids[best_idx], x_ids[best_idx]
            
    x[:, 3:, cy, cx] = 1.0
    print("Out make_single_seed")
    return x


print("Rendering word...")
tgt = render_word("COMP", 12)
print("Done rendering.")
print(f"Target shape: {tgt.shape}")

# Test one step of training
device = "cpu"
batch = 8
channel_n = 32
hidden_n = 128
ca_min = 5
ca_max = 10

target = torch.from_numpy(tgt)[None].repeat(batch, 1, 1, 1).to(device)
model = NCA(channel_n, hidden_n=hidden_n).to(device)
opt = torch.optim.Adam(model.parameters(), lr=2e-3)

seed = make_single_seed("COMP", channel_n, tgt=tgt)
pool = SamplePool(seed, 256)

print("Starting one step...")
idx, x = pool.sample(batch)
n_ca = int(torch.randint(ca_min, ca_max + 1, (1,)))
print(f"Running {n_ca} steps of NCA...")
x = model(x, steps=n_ca)
loss = F.mse_loss(to_rgba(x), target)

opt.zero_grad()
loss.backward()
opt.step()
print("Done one step.")

tgt_img = (tgt.transpose(1, 2, 0) * 255).astype(np.uint8)
Image.fromarray(tgt_img).save("test_render_target.png")
print("Saved test_render_target.png")

