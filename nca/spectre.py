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
