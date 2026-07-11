"""State-conditioned spatial action proposals.

The original DRLQN action library attached ACTION6 only to connected-object
centroids and four object-sized offsets.  That is a useful prior, but it cannot
represent many ARC-AGI-3 mechanics where the correct click is an empty cell, a
cell inside a merged region, or a location on an inferred board lattice.

This module keeps the fixed 24 x 5 click geometry while making the 24 anchors
dynamic.  Anchors combine object locations, inferred tile-cell centres,
changed regions, and boundary-adjacent empty locations.  The Q-router still
selects one of 128 lambda actions, so no large output-head or replay-format
change is required.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

from ..state import TOKEN_DIM

GRID = 64


@dataclass(frozen=True)
class ClickCandidate:
    """One spatial anchor used by the five click-offset operators."""

    x: int
    y: int
    kind: str
    color: int
    scale: int = 1
    score: float = 0.0

    @property
    def group(self) -> tuple:
        step = max(1, int(self.scale))
        return (self.kind, self.color, self.x // step, self.y // step)


def _run_lengths(line: np.ndarray) -> list[int]:
    if line.size == 0:
        return []
    out: list[int] = []
    start = 0
    for i in range(1, line.size):
        if line[i] != line[i - 1]:
            out.append(i - start)
            start = i
    out.append(line.size - start)
    return out


def infer_tile_scale(grid: np.ndarray) -> int:
    """Infer a likely rendered cell size from repeated color-run lengths.

    ARC environments frequently render an 8x8 or 16x16 logical board onto the
    64x64 observation.  Adjacent equal-color cells merge into one connected
    component, so connected-component centroids alone lose the individual
    clickable cells.  A robust divisor vote over horizontal and vertical runs
    recovers common scales such as 2, 4, 8, and 16.
    """

    g = np.asarray(grid)
    if g.shape != (GRID, GRID):
        return 1
    lengths: list[int] = []
    # Sampling every row/column is cheap at 64x64 and is more robust to HUDs
    # than relying on one scanline.
    for row in g:
        lengths.extend(_run_lengths(row))
    for col in g.T:
        lengths.extend(_run_lengths(col))
    useful = [n for n in lengths if 1 < n <= 32]
    if len(useful) < 8:
        return 1

    for scale in (16, 8, 4, 2):
        divisible = sum(1 for n in useful if n % scale == 0)
        exactish = sum(1 for n in useful if n in (scale, 2 * scale, 3 * scale, 4 * scale))
        support = divisible / len(useful)
        if divisible >= 8 and exactish >= 4 and support >= 0.62:
            return scale
    return 1


def _rarity(counts: np.ndarray, color: int) -> float:
    p = max(1.0, float(counts[int(color)])) / float(GRID * GRID)
    return min(6.0, -math.log(p + 1e-12))


def _boundary_density(grid: np.ndarray, y: int, x: int, step: int = 1) -> float:
    c = int(grid[y, x])
    diff = 0
    total = 0
    for dy, dx in ((-step, 0), (step, 0), (0, -step), (0, step)):
        yy, xx = y + dy, x + dx
        if 0 <= yy < GRID and 0 <= xx < GRID:
            total += 1
            diff += int(int(grid[yy, xx]) != c)
    return diff / max(1, total)


def _changed_centres(previous_grid: np.ndarray | None, grid: np.ndarray) -> list[tuple[int, int]]:
    if previous_grid is None or np.asarray(previous_grid).shape != grid.shape:
        return []
    changed = np.asarray(previous_grid) != grid
    seen = np.zeros_like(changed, dtype=bool)
    centres: list[tuple[int, int]] = []
    for y in range(GRID):
        for x in range(GRID):
            if not changed[y, x] or seen[y, x]:
                continue
            stack = [(y, x)]
            seen[y, x] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cy, cx = stack.pop()
                cells.append((cy, cx))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    yy, xx = cy + dy, cx + dx
                    if 0 <= yy < GRID and 0 <= xx < GRID and changed[yy, xx] and not seen[yy, xx]:
                        seen[yy, xx] = True
                        stack.append((yy, xx))
            ys = [p[0] for p in cells]
            xs = [p[1] for p in cells]
            centres.append((int(round(sum(ys) / len(ys))), int(round(sum(xs) / len(xs)))))
    return centres


def _object_candidates(objs: Iterable, counts: np.ndarray) -> list[ClickCandidate]:
    out: list[ClickCandidate] = []
    for obj in objs:
        h = int(obj.y1 - obj.y0 + 1)
        w = int(obj.x1 - obj.x0 + 1)
        scale = max(1, min(h, w))
        base = 4.0 + _rarity(counts, int(obj.color)) + 1.0 / max(1.0, math.sqrt(float(obj.area)))
        out.append(ClickCandidate(int(round(obj.cx)), int(round(obj.cy)), "object", int(obj.color), scale, base))
        # Empty-cell and boundary interactions are common (bridges, mirrors,
        # path extension).  Seed anchors immediately outside each object.
        pad = max(1, min(8, scale))
        neighbours = (
            (int(round(obj.cx)), int(obj.y0) - pad),
            (int(round(obj.cx)), int(obj.y1) + pad),
            (int(obj.x0) - pad, int(round(obj.cy))),
            (int(obj.x1) + pad, int(round(obj.cy))),
        )
        for x, y in neighbours:
            if 0 <= x < GRID and 0 <= y < GRID:
                out.append(ClickCandidate(x, y, "adjacent", -1, pad, base - 0.8))
    return out


def generate_click_candidates(
    grid: np.ndarray,
    objs: Iterable,
    max_candidates: int = 24,
    previous_grid: np.ndarray | None = None,
) -> list[ClickCandidate]:
    """Generate a compact, diverse set of spatial click anchors.

    The ranking is intentionally heuristic and teacher-friendly: demonstrations
    can train the Q-router to choose among the proposals, while the proposal
    generator guarantees that common non-object coordinates are representable.
    """

    g = np.asarray(grid, dtype=np.uint8)
    if g.shape != (GRID, GRID):
        raise ValueError(f"expected a 64x64 grid, got {g.shape}")
    counts = np.bincount(g.reshape(-1), minlength=16)
    pool: dict[tuple[int, int], ClickCandidate] = {}

    def add(c: ClickCandidate) -> None:
        x = int(np.clip(c.x, 0, GRID - 1))
        y = int(np.clip(c.y, 0, GRID - 1))
        color = int(g[y, x]) if c.color < 0 else int(c.color)
        fixed = ClickCandidate(x, y, c.kind, color, max(1, int(c.scale)), float(c.score))
        old = pool.get((x, y))
        if old is None or fixed.score > old.score:
            pool[(x, y)] = fixed

    for candidate in _object_candidates(objs, counts):
        add(candidate)

    scale = infer_tile_scale(g)
    if scale > 1:
        half = scale // 2
        for y in range(half, GRID, scale):
            for x in range(half, GRID, scale):
                color = int(g[y, x])
                boundary = _boundary_density(g, y, x, scale)
                rarity = _rarity(counts, color)
                # Border strips are often timers/HUDs; retain them, but rank
                # interior puzzle cells first unless they are very distinctive.
                border_penalty = 1.2 if min(x, y, GRID - 1 - x, GRID - 1 - y) < scale else 0.0
                score = 1.5 + 1.8 * rarity + 1.4 * boundary - border_penalty
                add(ClickCandidate(x, y, "tile", color, scale, score))

    for y, x in _changed_centres(previous_grid, g):
        add(ClickCandidate(x, y, "changed", int(g[y, x]), max(1, scale), 12.0))

    # A coarse fallback lattice guarantees arbitrary-coordinate reachability
    # even when no clean rendered cell scale can be inferred.
    fallback_step = 8 if scale == 1 else max(4, 2 * scale)
    for y in range(fallback_step // 2, GRID, fallback_step):
        for x in range(fallback_step // 2, GRID, fallback_step):
            score = 0.5 + _boundary_density(g, y, x, max(1, fallback_step // 2))
            add(ClickCandidate(x, y, "lattice", int(g[y, x]), fallback_step, score))

    ranked = sorted(pool.values(), key=lambda c: (-c.score, c.y, c.x, c.kind))
    selected: list[ClickCandidate] = []
    for candidate in ranked:
        # Keep distinct neighbouring logical cells; only suppress near-duplicate
        # proposals that land in essentially the same pixel neighbourhood.
        duplicate = False
        for prior in selected:
            radius = max(1, min(candidate.scale, prior.scale) // 3)
            if abs(candidate.x - prior.x) <= radius and abs(candidate.y - prior.y) <= radius:
                duplicate = True
                break
        if not duplicate:
            selected.append(candidate)
        if len(selected) >= max_candidates:
            break
    return selected


def candidate_token(candidate: ClickCandidate, grid: np.ndarray) -> np.ndarray:
    """Encode a spatial anchor in the same 33-D object-token geometry."""

    g = np.asarray(grid)
    y, x = int(candidate.y), int(candidate.x)
    color = int(g[y, x])
    f = np.zeros(TOKEN_DIM, dtype=np.float64)
    if 0 <= color < 16:
        f[color] = 1.0
    scale = max(1, int(candidate.scale))
    f[16] = y / 64.0
    f[17] = x / 64.0
    f[18] = min(1.0, scale / 64.0)
    f[19] = min(1.0, scale / 64.0)
    f[20] = min(1.0, (scale * scale) / 512.0)
    f[21] = 1.0
    f[22] = 0.25
    f[23] = _boundary_density(g, y, x, max(1, scale // 2))

    # Orientation-sensitive 3x3 local signature, analogous to Obj.shape3.
    for gy, dy in enumerate((-scale, 0, scale)):
        for gx, dx in enumerate((-scale, 0, scale)):
            yy = int(np.clip(y + dy, 0, GRID - 1))
            xx = int(np.clip(x + dx, 0, GRID - 1))
            f[24 + gy * 3 + gx] = float(int(g[yy, xx]) == color)
    return f
