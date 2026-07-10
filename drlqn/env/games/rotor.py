"""rt02 "Rotor": click environment.

A row of tiles each shows a 5x5 glyph in some rotation; a reference tile
(marked border) shows the target orientation. Clicking a tile rotates it 90
degrees clockwise. Later levels: more tiles, linked tiles (clicking one also
rotates its right neighbor), and a mirrored glyph that must be un-mirrored by
double rotation. Win when every tile matches the reference glyph exactly.
Colors: 1 tile bg, 14 glyph, 4 reference border, 6 linked marker.
"""
from __future__ import annotations

import numpy as np

from ..core import ArcEnv, CLICK, canvas, rect, border

TILE = 12          # pixels per tile (5x5 glyph at 2px + margin)
GAP = 3

GLYPH = np.array([
    [0, 1, 1, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 1, 1, 0],
], dtype=np.uint8)


class Rotor(ArcEnv):
    env_id = "rt02"
    action_space = ()
    uses_click = True

    _SPECS = {
        # level: (n_tiles, rotations, linked pairs {i: j})
        1: (1, [1], {}),
        2: (2, [2, 3], {}),
        3: (3, [1, 2, 3], {}),
        4: (3, [3, 1, 2], {0: 1}),
        5: (4, [2, 1, 3, 2], {1: 2}),
        6: (4, [3, 2, 1, 3], {0: 1, 2: 3}),
    }

    def _load_level(self, level: int) -> None:
        n, rots, linked = self._SPECS[level]
        self.n = n
        self.rot = list(rots)
        self.linked = dict(linked)

    def _tile_at(self, px, py):
        y0 = 26
        for i in range(self.n):
            x0 = 4 + i * (TILE + GAP)
            if x0 <= px < x0 + TILE and y0 <= py < y0 + TILE:
                return i
        return None

    def _apply(self, action, x, y):
        i = self._tile_at(x, y)
        if i is None:
            return False
        self.rot[i] = (self.rot[i] + 1) % 4
        if i in self.linked:
            j = self.linked[i]
            self.rot[j] = (self.rot[j] + 1) % 4
        return all(r == 0 for r in self.rot)

    def _draw_tile(self, g, y0, x0, rot, ref=False, link=False):
        rect(g, y0, x0, TILE, TILE, 1)
        if ref:
            border(g, y0, x0, TILE, TILE, 4)
        glyph = np.rot90(GLYPH, k=-rot)
        for yy in range(5):
            for xx in range(5):
                if glyph[yy, xx]:
                    rect(g, y0 + 1 + yy * 2, x0 + 1 + xx * 2, 2, 2, 14)
        if link:
            g[y0 + TILE - 1, x0 + TILE // 2] = 6

    def _render(self):
        g = canvas(0)
        self._draw_tile(g, 6, 4, 0, ref=True)             # reference tile
        for i in range(self.n):
            self._draw_tile(g, 26, 4 + i * (TILE + GAP), self.rot[i],
                            link=(i in self.linked))
        return g
