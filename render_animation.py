import sys
import numpy as np
import torch
from PIL import Image
from moviepy import ImageSequenceClip
import argparse

from nca.ocr_eval import load_model
from nca.model import make_seed, to_rgb

def main():
    path = "weights/0047.json"  # Letter 'G' from ladder.sh rung 1
    model, d = load_model(path)
    
    with torch.no_grad():
        x = make_seed(d["grid"], d["channel_n"])
        frames = []
        for i in range(150):
            img = to_rgb(x)[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
            im = Image.fromarray((img * 255).astype(np.uint8))
            
            # upscale by 8x for crisp pixel art visibility
            im = im.resize((im.width * 8, im.height * 8), Image.NEAREST)
            frames.append(np.array(im))
            
            x = model(x, steps=1)
            
    clip = ImageSequenceClip(frames, fps=30)
    clip.write_videofile("/usr/local/google/home/rop/.gemini/jetski/brain/6c347665-318b-457d-a25d-e40b0fa9c864/morph_single.mp4", codec="libx264")
    print("Saved morph_single.mp4")
    
if __name__ == "__main__":
    main()
