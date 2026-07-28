import glob
from moviepy import ImageSequenceClip
import os

def main():
    # Only grab every second frame (i.e. every 200 iterations since it saves every 100)
    images = sorted(glob.glob("snaps_q/*.png"))[::2]
    if not images:
        raise ValueError("No images found in snaps_q/")

    # 4 frames per second to clearly see the progression
    clip = ImageSequenceClip(images, fps=4)
    clip.write_videofile("/usr/local/google/home/rop/.gemini/jetski/brain/6c347665-318b-457d-a25d-e40b0fa9c864/training_q.mp4", codec="libx264")
    print("Saved training progression video.")

if __name__ == "__main__":
    main()
