"""Canonical Markov state and object-centric perception.

s_t = (w_t, tau_t, o_t):
  w_t  - the frame decomposed into objects (connected components);
  tau_t - trace summary (visit counts, last lambda, novelty stats);
  o_t  - obligations are implicit in ARC-AGI-3 (goal must be inferred), so we
         carry the exploration frontier summary instead.

NodeCache (runtime/cache.py) is deliberately NOT part of this state.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

N_COLORS = 16
MAX_TOKENS = 64
TOKEN_DIM = 33     # 16 color one-hot + cy,cx + h,w + area + fill + aspect + perim
                   # + 3x3 orientation-sensitive shape mask


@dataclass
class Obj:
    color: int
    y0: int; x0: int; y1: int; x1: int
    cy: float; cx: float
    area: int
    fill: float
    perim: float
    shape3: np.ndarray = None      # (3,3) coarse occupancy mask


def extract_objects(grid: np.ndarray, max_tokens: int = MAX_TOKENS) -> list[Obj]:
    """4-connected same-color components (background color 0 skipped)."""
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    objs: list[Obj] = []
    for y in range(h):
        for x in range(w):
            if seen[y, x] or grid[y, x] == 0:
                continue
            color = grid[y, x]
            stack = [(y, x)]
            seen[y, x] = True
            cells = []
            while stack:
                cy, cx = stack.pop()
                cells.append((cy, cx))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] \
                            and grid[ny, nx] == color:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            ys = np.array([c[0] for c in cells]); xs = np.array([c[1] for c in cells])
            y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
            bh, bw = y1 - y0 + 1, x1 - x0 + 1
            area = len(cells)
            # perimeter: cells with at least one non-member neighbour
            cellset = set(cells)
            perim = sum(
                1 for (cy, cx) in cells
                if any((cy + dy, cx + dx) not in cellset
                       for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)))
            )
            objs.append(Obj(int(color), int(y0), int(x0), int(y1), int(x1),
                            float(ys.mean()), float(xs.mean()), area,
                            area / (bh * bw), perim / max(1, area),
                            _shape3(cells, y0, x0, bh, bw)))
    # deterministic order: big background plates last relevance-wise; sort by
    # (area asc is small-first) -- small objects are usually the interactive ones
    objs.sort(key=lambda o: (o.area, o.color, o.y0, o.x0))
    return objs[:max_tokens]


def _shape3(cells, y0, x0, bh, bw) -> np.ndarray:
    """3x3 normalized occupancy: orientation-sensitive shape signature."""
    m = np.zeros((3, 3))
    cnt = np.zeros((3, 3))
    for (cy, cx) in cells:
        gy = min(2, (cy - y0) * 3 // bh)
        gx = min(2, (cx - x0) * 3 // bw)
        m[gy, gx] += 1
    # normalize by bin capacity
    for gy in range(3):
        for gx in range(3):
            hy = (bh * (gy + 1) // 3) - (bh * gy // 3)
            hx = (bw * (gx + 1) // 3) - (bw * gx // 3)
            cnt[gy, gx] = max(1, hy * hx)
    return m / cnt


def tokenize(objs: list[Obj]) -> np.ndarray:
    """(n_tokens, TOKEN_DIM) float features."""
    n = len(objs)
    F = np.zeros((max(n, 1), TOKEN_DIM))
    for i, o in enumerate(objs):
        F[i, o.color] = 1.0
        F[i, 16] = o.cy / 64.0
        F[i, 17] = o.cx / 64.0
        F[i, 18] = (o.y1 - o.y0 + 1) / 64.0
        F[i, 19] = (o.x1 - o.x0 + 1) / 64.0
        F[i, 20] = min(1.0, o.area / 512.0)
        F[i, 21] = o.fill
        F[i, 22] = min(4.0, (o.x1 - o.x0 + 1) / (o.y1 - o.y0 + 1)) / 4.0
        F[i, 23] = o.perim
        F[i, 24:33] = o.shape3.reshape(-1)
    return F


def frame_key(grid: np.ndarray, level: int) -> str:
    return hashlib.sha1(grid.tobytes() + bytes([level])).hexdigest()[:16]


def semantic_delta(prev: np.ndarray, cur: np.ndarray) -> np.ndarray:
    """Changed-cell mask between frames: the sparse semantic delta of one step."""
    return prev != cur


@dataclass
class TraceSummary:
    """tau_t: enough history to keep the decision Markov for our purposes."""
    visit_count: int = 0
    last_lambda: int = -1
    steps_in_level: int = 0
    novelty_rate: float = 1.0

    def features(self) -> np.ndarray:
        return np.array([
            min(1.0, self.visit_count / 8.0),
            (self.last_lambda + 1) / 128.0,
            min(1.0, self.steps_in_level / 256.0),
            self.novelty_rate,
        ])
