import numpy as np
import PIL.Image
import PIL.ImageDraw

def make_voronoi(w, h, n_sites):
    sites = np.random.rand(n_sites, 2) * [w, h]
    
    # Grid of coordinates
    Y, X = np.mgrid[0:h, 0:w]
    
    # Distances
    dists = np.zeros((n_sites, h, w))
    for i in range(n_sites):
        dists[i] = (X - sites[i, 0])**2 + (Y - sites[i, 1])**2
        
    # Closest site
    closest = np.argmin(dists, axis=0)
    
    # Boundaries (where neighbor differs)
    boundaries = np.zeros((h, w), dtype=bool)
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0:
                continue
            shifted = np.roll(closest, shift=(dy, dx), axis=(0, 1))
            boundaries |= (shifted != closest)
            
    # Render
    img = PIL.Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = PIL.ImageDraw.Draw(img)
    
    for y in range(h):
        for x in range(w):
            if boundaries[y, x]:
                img.putpixel((x, y), (128, 128, 128, 128))
                
    img.save("scratch/dev_voronoi.png")
    print("Saved scratch/dev_voronoi.png")

if __name__ == "__main__":
    import os
    os.makedirs("scratch", exist_ok=True)
    make_voronoi(100, 20, 10)
