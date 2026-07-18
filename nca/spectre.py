"""Spectre (Tile(1,1)) aperiodic monotile tilings.

Implements the substitution algorithm of Smith, Myers, Kaplan &
Goodman-Strauss (arXiv:2305.17743, Appendix A): nine metatile classes,
the Gamma 'Mystic' two-Spectre compound, and one round of supertile
placement rules, iterated to produce arbitrarily large valid patches.
Leaves are collected as (label, affine) pairs and rasterized with a
per-class color so tiling validity is visually inspectable.
"""
import numpy as np
from PIL import Image, ImageDraw

SQ3 = np.sqrt(3.0)

SPECTRE = np.array([
    [0.0, 0.0], [1.0, 0.0], [1.5, -SQ3 / 2],
    [1.5 + SQ3 / 2, 0.5 - SQ3 / 2], [1.5 + SQ3 / 2, 1.5 - SQ3 / 2],
    [2.5 + SQ3 / 2, 1.5 - SQ3 / 2], [3.0 + SQ3 / 2, 1.5], [3.0, 2.0],
    [3.0 - SQ3 / 2, 1.5], [2.5 - SQ3 / 2, 1.5 + SQ3 / 2],
    [1.5 - SQ3 / 2, 1.5 + SQ3 / 2], [0.5 - SQ3 / 2, 1.5 + SQ3 / 2],
    [-SQ3 / 2, 1.5], [0.0, 1.0],
])
QUAD_IDX = [3, 5, 7, 11]
NAMES = ["Gamma", "Delta", "Theta", "Lambda", "Xi", "Pi", "Sigma", "Phi", "Psi"]

PALETTE = {
    "Gamma1": (196, 201, 169), "Gamma2": (156, 160, 116),
    "Delta": (220, 220, 220), "Theta": (255, 191, 191),
    "Lambda": (255, 160, 122), "Xi": (255, 242, 0),
    "Pi": (135, 206, 250), "Sigma": (245, 245, 220),
    "Phi": (0, 255, 0), "Psi": (0, 255, 255),
}


def A(mat=None):
    return np.eye(3) if mat is None else np.asarray(mat, float)


def rot(deg):
    r = np.deg2rad(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def trans(x, y):
    return np.array([[1, 0, x], [0, 1, y], [0, 0, 1]])


MIRROR = np.diag([-1.0, 1.0, 1.0])


def apply(T, pts):
    P = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
    return (P @ T.T)[:, :2]


class Leaf:
    def __init__(self, label):
        self.label = label
        self.quad = SPECTRE[QUAD_IDX]

    def collect(self, T, out):
        out.append((self.label, T))


class Meta:
    def __init__(self, children, quad):
        self.children = children   # [(node, T)]
        self.quad = quad

    def collect(self, T, out):
        for node, ct in self.children:
            node.collect(T @ ct, out)


def build_base():
    sys = {n: Leaf(n) for n in NAMES if n != "Gamma"}
    mystic = Meta(
        [(Leaf("Gamma1"), A()),
         (Leaf("Gamma2"), trans(*SPECTRE[8]) @ rot(30))],
        SPECTRE[QUAD_IDX])
    sys["Gamma"] = mystic
    return sys


RULES = [(60, 3, 1), (0, 2, 0), (60, 3, 1), (60, 3, 1),
         (0, 2, 0), (60, 3, 1), (-120, 3, 3)]
SUPER = {
    "Gamma":  ["Pi", "Delta", None, "Theta", "Sigma", "Xi", "Phi", "Gamma"],
    "Delta":  ["Xi", "Delta", "Xi", "Phi", "Sigma", "Pi", "Phi", "Gamma"],
    "Theta":  ["Psi", "Delta", "Pi", "Phi", "Sigma", "Pi", "Phi", "Gamma"],
    "Lambda": ["Psi", "Delta", "Xi", "Phi", "Sigma", "Pi", "Phi", "Gamma"],
    "Xi":     ["Psi", "Delta", "Pi", "Phi", "Sigma", "Psi", "Phi", "Gamma"],
    "Pi":     ["Psi", "Delta", "Xi", "Phi", "Sigma", "Psi", "Phi", "Gamma"],
    "Sigma":  ["Xi", "Delta", "Xi", "Phi", "Sigma", "Pi", "Lambda", "Gamma"],
    "Phi":    ["Psi", "Delta", "Psi", "Phi", "Sigma", "Pi", "Phi", "Gamma"],
    "Psi":    ["Psi", "Delta", "Psi", "Phi", "Sigma", "Psi", "Phi", "Gamma"],
}


def build_super(sys):
    quad = sys["Delta"].quad
    transformations = [A()]
    total, rotation = 0, A()
    tq = quad.copy()
    for ang, i_from, i_to in RULES:
        if ang != 0:
            total += ang
            rotation = rot(total)
            tq = apply(rotation, quad)
        prev_pt = apply(transformations[-1], quad[i_from:i_from + 1])[0]
        ttt = trans(*(prev_pt - tq[i_to]))
        transformations.append(ttt @ rotation)
    transformations = [MIRROR @ t for t in transformations]
    super_quad = np.stack([
        apply(transformations[6], quad[2:3])[0],
        apply(transformations[5], quad[1:2])[0],
        apply(transformations[3], quad[2:3])[0],
        apply(transformations[0], quad[1:2])[0],
    ])
    return {
        label: Meta(
            [(sys[s], t) for s, t in zip(subs, transformations) if s],
            super_quad)
        for label, subs in SUPER.items()}


def spectre_leaves(iterations=3, root="Delta"):
    sys = build_base()
    for _ in range(iterations):
        sys = build_super(sys)
    out = []
    sys[root].collect(A(), out)
    return out


def rasterize(leaves, canvas=72, scale=5.0, center=None, upscale=1):
    """RGBA [4, canvas, canvas] float; colors by tile class, dark edges."""
    if center is None:
        pts = np.concatenate([apply(T, SPECTRE) for _, T in leaves[:400]])
        center = pts.mean(axis=0)
    C = canvas * upscale
    img = Image.new("RGBA", (C, C), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    half = canvas / 2.0
    for label, T in leaves:
        pts = apply(T, SPECTRE)
        pix = (pts - center) * scale + half
        if pix[:, 0].max() < -2 or pix[:, 0].min() > canvas + 2 \
                or pix[:, 1].max() < -2 or pix[:, 1].min() > canvas + 2:
            continue
        draw.polygon([tuple(p * upscale) for p in pix],
                     fill=PALETTE[label] + (255,), outline=(25, 25, 25, 255))
    if upscale > 1:
        img = img.resize((canvas, canvas), Image.LANCZOS)
    arr = np.asarray(img, np.float32) / 255.0
    arr[..., :3] *= arr[..., 3:]
    return arr.transpose(2, 0, 1)


def rasterize_crisp(leaves, canvas=72, scale=5.0, center=None, edge_px=0.6):
    """Exact rasterization, no image library: crossing-number interior test
    on pixel centers plus true distance-to-boundary for 1px crisp edges.

    Returns (edge_mask [H,W] bool, interior_label [H,W] int32 (-1 = none),
    labels list) — callers build whatever target/loss they want from it.
    """
    if center is None:
        pts = np.concatenate([apply(T, SPECTRE) for _, T in leaves[:400]])
        center = pts.mean(axis=0)
    H = W = canvas
    half = canvas / 2.0
    edge = np.zeros((H, W), bool)
    interior = -np.ones((H, W), np.int32)
    labels = []
    for li, (label, T) in enumerate(leaves):
        pix = (apply(T, SPECTRE) - center) * scale + half   # (x, y)
        x0 = max(0, int(np.floor(pix[:, 0].min() - 1)))
        x1 = min(W - 1, int(np.ceil(pix[:, 0].max() + 1)))
        y0 = max(0, int(np.floor(pix[:, 1].min() - 1)))
        y1 = min(H - 1, int(np.ceil(pix[:, 1].max() + 1)))
        if x1 < x0 or y1 < y0:
            labels.append(label)
            continue
        gy, gx = np.mgrid[y0:y1 + 1, x0:x1 + 1]
        px = gx.astype(np.float64)
        py = gy.astype(np.float64)
        inside = np.zeros(px.shape, bool)
        dmin = np.full(px.shape, 1e9)
        n = len(pix)
        for i in range(n):
            xa1, ya1 = pix[i]
            xa2, ya2 = pix[(i + 1) % n]
            cond = ((ya1 > py) != (ya2 > py))
            with np.errstate(divide="ignore", invalid="ignore"):
                xin = (xa2 - xa1) * (py - ya1) / (ya2 - ya1) + xa1
            inside ^= cond & (px < xin)
            # point-segment distance
            vx, vy = xa2 - xa1, ya2 - ya1
            L2 = vx * vx + vy * vy
            t = np.clip(((px - xa1) * vx + (py - ya1) * vy) / max(L2, 1e-12), 0, 1)
            d = np.hypot(px - (xa1 + t * vx), py - (ya1 + t * vy))
            dmin = np.minimum(dmin, d)
        e = dmin <= edge_px
        edge[y0:y1 + 1, x0:x1 + 1] |= e
        put = inside & ~e
        interior[y0:y1 + 1, x0:x1 + 1] = np.where(
            put, li, interior[y0:y1 + 1, x0:x1 + 1])
        labels.append(label)
    return edge, interior, labels


def crisp_target(edge, interior, labels, mode="fill"):
    """RGBA [4,H,W] from crisp masks. Modes:
    outline: dark edges only, empty interiors.
    fill:    dark edges + exact per-class interior colors.
    free:    dark edges + interiors marked present at alpha=1 with NEUTRAL
             rgb 0.5 (trainers should not supervise interior rgb — only
             presence and being distinct from the near-black outline)."""
    H, W = edge.shape
    arr = np.zeros((4, H, W), np.float32)
    arr[3][edge] = 1.0            # edges: near-black
    arr[:3, edge] = 0.08
    inter = interior >= 0
    if mode == "outline":
        return arr
    arr[3][inter] = 1.0
    if mode == "fill":
        for li, label in enumerate(labels):
            m = interior == li
            if not m.any():
                continue
            c = np.array(PALETTE[label], np.float32) / 255.0
            for ch in range(3):
                arr[ch][m] = c[ch]
    else:                          # free
        for ch in range(3):
            arr[ch][inter] = 0.5
    return arr
