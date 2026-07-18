"""Topological candidate-counting for NCA fields (0-dim persistence).

Sweep a threshold down through the alpha field; components are BORN at
their peak height and DIE when they merge into a taller neighbor (the
elder rule). Persistence = birth - death. Long-lived components are real
candidates; short-lived ones are texture. This is the computable version
of index-theory attractor counting for our canvases: the negotiation
dynamic succeeds iff the ledger converges to exactly one long-lived
component.
"""
import numpy as np


def persistence_0d(field, min_persistence=0.1):
    """[(birth, death, (y, x))] for local-max-born components of a 2D field."""
    H, W = field.shape
    order = np.argsort(field.ravel())[::-1]   # descending by height
    parent = -np.ones(H * W, dtype=np.int64)  # -1 = not yet in any component
    root_birth = {}
    root_peak = {}
    events = []

    def find(i):
        r = i
        while parent[r] != r:
            r = parent[r]
        while parent[i] != r:
            parent[i], i = r, parent[i]
        return r

    for flat in order:
        y, x = divmod(flat, W)
        h = field[y, x]
        nbr_roots = set()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W and parent[ny * W + nx] != -1:
                    nbr_roots.add(find(ny * W + nx))
        if not nbr_roots:
            parent[flat] = flat
            root_birth[flat] = h
            root_peak[flat] = (y, x)
        else:
            roots = sorted(nbr_roots, key=lambda r: root_birth[r], reverse=True)
            elder = roots[0]
            parent[flat] = elder
            for r in roots[1:]:          # younger components die here
                events.append((root_birth[r], h, root_peak[r]))
                parent[r] = elder
    # survivors never die
    seen = set()
    for i in range(H * W):
        if parent[i] == -1:
            continue
        r = find(i)
        if r not in seen:
            seen.add(r)
            events.append((root_birth[r], float(field.min()), root_peak[r]))
    return [(b, d, p) for (b, d, p) in events if b - d >= min_persistence]


def count_candidates(alpha, min_persistence=0.25):
    """Number of persistent blobs in an alpha field (candidates)."""
    return len(persistence_0d(np.asarray(alpha, dtype=np.float32),
                              min_persistence))
