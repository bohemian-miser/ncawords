import glob
from moviepy import ImageSequenceClip
import os

def main():
    images = sorted(glob.glob("snaps_comp/*.png"))
    if not images:
        raise ValueError("No images found in snaps_comp/")

    clip = ImageSequenceClip(images, fps=6)
    clip_path = "/usr/local/google/home/rop/.gemini/jetski/brain/6c347665-318b-457d-a25d-e40b0fa9c864/training_comp.mp4"
    clip.write_videofile(clip_path, codec="libx264")
    print("Saved training progression video for COMP.")

if __name__ == "__main__":
    main()
