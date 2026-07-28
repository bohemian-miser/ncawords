import numpy as np
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import tensorflow as tf

def create_target_image(text="C", size=(40, 40)):
    img = PIL.Image.new('RGBA', size, (0, 0, 0, 0))
    d = PIL.ImageDraw.Draw(img)
    try:
        font = PIL.ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    except Exception:
        font = PIL.ImageFont.load_default()
    
    try:
        left, top, right, bottom = d.textbbox((0, 0), text, font=font)
        text_width = right - left
        text_height = bottom - top
    except AttributeError:
        text_width, text_height = d.textsize(text, font=font)

    x = (size[0] - text_width) / 2
    y = (size[1] - text_height) / 2
    
    d.text((x, y), text, font=font, fill=(255, 255, 255, 255))
    arr = np.float32(img)/255.0
    arr[..., :3] *= arr[..., 3:]
    return arr

arr = create_target_image()
rgb, a = arr[..., :3], arr[..., 3:4]
vis = 1.0 - a + rgb
PIL.Image.fromarray((vis * 255).astype(np.uint8)).save("true_target.png")
print("Target saved, max alpha:", np.max(a))
