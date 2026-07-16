import os
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 40, 40
OUT_DIR = "snaps_proposed_targets"
os.makedirs(OUT_DIR, exist_ok=True)

def draw_char(char, x, y):
    img = Image.new("RGBA", (W, H), (0,0,0,0))
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf", 10)
    except:
        font = ImageFont.load_default()
    d.text((x, y), char, fill="white", font=font)
    return np.array(img)[..., 3] > 0

arr_C = draw_char("C", 6, 14)
arr_O = draw_char("O", 14, 14)
arr_M = draw_char("M", 22, 14)
arr_P = draw_char("P", 30, 14)

def jagged_line(p1, p2, variance=2.0, segments=8):
    # Create a jagged path from p1 to p2
    pts = [np.array(p1, dtype=float)]
    for i in range(1, segments):
        t = i / float(segments)
        mid = np.array(p1) * (1 - t) + np.array(p2) * t
        # perpendicular displacement
        dv = np.array([p2[1] - p1[1], p1[0] - p2[0]], dtype=float)
        norm = np.linalg.norm(dv)
        if norm > 0:
            dv /= norm
        displacement = (np.random.rand() - 0.5) * 2 * variance
        pt = mid + dv * displacement
        pts.append(pt)
    pts.append(np.array(p2, dtype=float))
    return pts

def create_web_frame_sequence(start_mask, end_mask, frames):
    # Find rightmost points of start, leftmost of end
    start_pts = np.argwhere(start_mask)
    end_pts = np.argwhere(end_mask)
    
    start_pts_sorted = sorted(start_pts, key=lambda p: -p[1]) # sort rightmost
    end_pts_sorted = sorted(end_pts, key=lambda p: p[1])      # sort leftmost
    
    if len(start_pts_sorted) == 0 or len(end_pts_sorted) == 0:
        return [np.zeros((H,W), dtype=bool) for _ in range(frames)]
        
    # Pick a few connection points
    num_lines = 3
    all_lines_pts = [] # lists of pixels
    
    img = Image.new("1", (W, H), 0)
    d = ImageDraw.Draw(img)
    
    # Pre-draw all jagged lines
    for _ in range(num_lines):
        p1 = start_pts_sorted[np.random.randint(min(5, len(start_pts_sorted)))]
        p2 = end_pts_sorted[np.random.randint(min(5, len(end_pts_sorted)))]
        
        # p is (y, x), ImageDraw uses (x, y)
        path = jagged_line((p1[1], p1[0]), (p2[1], p2[0]), variance=2.0)
        path_tuples = [(pt[0], pt[1]) for pt in path]
        d.line(path_tuples, fill=1, width=1)
        
    web_mask = np.array(img).astype(bool)
    web_pts = np.argwhere(web_mask)
    
    # Now we need to form 'frames' by slowly revealing this web from left to right
    # Let's say web goes from min_x to max_x
    if len(web_pts) > 0:
        min_x = np.min(web_pts[:, 1])
        max_x = np.max(web_pts[:, 1])
    else:
        min_x, max_x = 0, 0
        
    seq = []
    for f in range(frames):
        alpha = f / float(frames - 1) if frames > 1 else 1.0
        current_max_x = min_x + (max_x - min_x) * alpha
        # reveal pixels whose x <= current_max_x
        curr_mask = np.zeros_like(web_mask)
        for (y, x) in web_pts:
            # We add a little randomness so it's not a perfect vertical scanline reveal
            if x <= current_max_x + (np.random.rand()-0.5)*2:
                curr_mask[y, x] = True
        seq.append(curr_mask)
        
    return seq

def save_mask(rgba_arr, idx):
    Image.fromarray(rgba_arr).save(os.path.join(OUT_DIR, f"TARGET_{idx:05d}.png"))

def apply_mask(target_arr, mask, color):
    target_arr[mask] = color

def generate_all():
    step = 0
    current_persistent = np.zeros((H, W, 4), dtype=np.uint8)
    apply_mask(current_persistent, arr_C, [255, 255, 255, 255])
    
    # 0 - 500: C
    for _ in range(500):
        save_mask(current_persistent, step)
        step += 1
        
    # 500 - 1000: Web from C to O
    web_CO = create_web_frame_sequence(arr_C, arr_O, 500)
    for w_mask in web_CO:
        frame = current_persistent.copy()
        apply_mask(frame, w_mask, [128, 128, 128, 255])
        save_mask(frame, step)
        step += 1
    apply_mask(current_persistent, web_CO[-1], [128, 128, 128, 255])
    
    # 1000 - 1500: Reveal O
    for i in range(500):
        if i > 250:
            apply_mask(current_persistent, arr_O, [255, 255, 255, 255])
        save_mask(current_persistent, step)
        step += 1
        
    # 1500 - 2000: Web from O to M
    web_OM = create_web_frame_sequence(arr_O, arr_M, 500)
    for w_mask in web_OM:
        frame = current_persistent.copy()
        apply_mask(frame, w_mask, [128, 128, 128, 255])
        save_mask(frame, step)
        step += 1
    apply_mask(current_persistent, web_OM[-1], [128, 128, 128, 255])
    
    # 2000 - 2500: Reveal M
    for i in range(500):
        if i > 250:
            apply_mask(current_persistent, arr_M, [255, 255, 255, 255])
        save_mask(current_persistent, step)
        step += 1
        
    # 2500 - 3000: Web from M to P
    web_MP = create_web_frame_sequence(arr_M, arr_P, 500)
    for w_mask in web_MP:
        frame = current_persistent.copy()
        apply_mask(frame, w_mask, [128, 128, 128, 255])
        save_mask(frame, step)
        step += 1
    apply_mask(current_persistent, web_MP[-1], [128, 128, 128, 255])
    
    # 3000 - 4000: Reveal P and hold
    apply_mask(current_persistent, arr_P, [255, 255, 255, 255])
    for _ in range(1000):
        save_mask(current_persistent, step)
        step += 1

if __name__ == "__main__":
    np.random.seed(42)
    generate_all()
    print("Generated targets with jagged lines in", OUT_DIR)
