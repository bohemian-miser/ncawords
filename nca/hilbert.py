"""Hilbert curve ground truth + validity instruments.

Curves are rasterized as 2px-wide paths with a color gradient along
arclength. Validity of an arbitrary alpha field is scored by exact graph
invariants of its path cells: connected components (must be 1), cycles
via Euler's formula E - V + C (must be 0), and degree violations (every
path cell has exactly 2 path-neighbors except the two endpoints).
"""
import numpy as np


def hilbert_points(order):
    """Unit-square Hilbert curve vertices, order n -> 4**n points."""
    # d2xy conversion, standard bit-twiddling construction
    n = 2 ** order
    pts = np.zeros((n * n, 2), np.float64)
    for d in range(n * n):
        rx = ry = 0
        x = y = 0
        t = d
        s = 1
        while s < n:
            rx = 1 & (t // 2)
            ry = 1 & (t ^ rx)
            if ry == 0:
                if rx == 1:
                    x, y = s - 1 - x, s - 1 - y
                x, y = y, x
            x += s * rx
            y += s * ry
            t //= 4
            s *= 2
        pts[d] = (x, y)
    return pts / (n - 1)   # normalized 0..1


def native_curve(order):
    """Boolean path grid at native lattice resolution plus arclength map.

    Vertices sit at even coordinates of a (2*2**order - 1)^2 grid with edge
    cells between them — the ON set is exactly a 4-connected simple path.
    """
    n = 2 ** order
    N = 2 * n - 1
    pts = (hilbert_points(order) * (n - 1)).round().astype(int)
    on = np.zeros((N, N), bool)
    arc = np.full((N, N), -1.0, np.float32)
    m = len(pts)
    for i, (x, y) in enumerate(pts):
        on[2 * y, 2 * x] = True
        arc[2 * y, 2 * x] = i / (m - 1)
        if i + 1 < m:
            nx, ny = pts[i + 1]
            ey, ex = y + ny, x + nx      # midpoint in doubled coords
            on[ey, ex] = True
            arc[ey, ex] = (i + 0.5) / (m - 1)
    return on, arc


def rasterize_curve(order, canvas=64, margin=None):
    """RGBA [4,canvas,canvas]: exact lattice path, arclength gradient,
    NEAREST-upscaled so invariants survive rasterization."""
    on, arc = native_curve(order)
    N = on.shape[0]
    cell = max(1, (canvas - 4) // N)
    if margin is None:
        margin = (canvas - N * cell) // 2
    arr = np.zeros((4, canvas, canvas), np.float32)
    ys, xs = np.where(on)
    for y, x in zip(ys, xs):
        f = float(arc[y, x])
        r, g, b = 1 - f, 0.35 + 0.5 * abs(0.5 - f), f
        y0, x0 = margin + y * cell, margin + x * cell
        arr[0, y0:y0 + cell, x0:x0 + cell] = r
        arr[1, y0:y0 + cell, x0:x0 + cell] = g
        arr[2, y0:y0 + cell, x0:x0 + cell] = b
        arr[3, y0:y0 + cell, x0:x0 + cell] = 1.0
    return arr


def field_invariants(alpha, order, canvas=None, margin=None, thresh=0.35):
    """Pool a model's alpha field back to the curve's native lattice and
    run the exact graph invariants there."""
    on, _ = native_curve(order)
    N = on.shape[0]
    canvas = canvas or alpha.shape[0]
    cell = max(1, (canvas - 4) // N)
    if margin is None:
        margin = (canvas - N * cell) // 2
    pooled = np.zeros((N, N), np.float32)
    for y in range(N):
        for x in range(N):
            y0, x0 = margin + y * cell, margin + x * cell
            pooled[y, x] = alpha[y0:y0 + cell, x0:x0 + cell].mean()
    return path_invariants(pooled, thresh)


def path_invariants(alpha, thresh=0.35):
    """(components, cycles, degree_violation_frac, coverage) of a field."""
    on = alpha > thresh
    H, W = on.shape
    idx = -np.ones((H, W), np.int64)
    ys, xs = np.where(on)
    V = len(ys)
    if V == 0:
        return 0, 0, 1.0, 0.0
    idx[ys, xs] = np.arange(V)
    parent = np.arange(V)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    E = 0
    deg = np.zeros(V, np.int32)
    for dy, dx in [(0, 1), (1, 0)]:
        a = on[:H - dy or H, :W - dx or W] & on[dy:, dx:]
        ay, ax = np.where(a)
        for y, x in zip(ay, ax):
            u, v = idx[y, x], idx[y + dy, x + dx]
            E += 1
            deg[u] += 1
            deg[v] += 1
            ru, rv = find(u), find(v)
            if ru != rv:
                parent[ru] = rv
    C = len({find(i) for i in range(V)})
    cycles = E - V + C
    # a clean 4-connected path: all degree 2 except two endpoints (degree 1)
    viol = float(np.sum((deg != 2)) - 2) / max(V, 1)
    return C, cycles, max(0.0, viol), V / (H * W)
