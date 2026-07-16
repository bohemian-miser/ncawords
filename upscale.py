import sys
from PIL import Image

def upscale_image(path, upscale=8):
    try:
        im = Image.open(path)
        # Use NEAREST for pixel art style
        im_resized = im.resize((im.width * upscale, im.height * upscale), Image.Resampling.NEAREST)
        new_path = path.replace(".png", "_upscaled.png")
        im_resized.save(new_path)
        print(f"Upscaled image saved to {new_path}")
        return new_path
    except Exception as e:
        print(f"Error upscaling image: {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) > 1:
        upscale_image(sys.argv[1])
    else:
        print("Usage: python upscale.py <path_to_image>")
