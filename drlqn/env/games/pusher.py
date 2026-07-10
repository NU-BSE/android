"""pb04 "Pusher": movement environment with block pushing (basic physics).

Push blocks onto sockets. Later levels: two blocks, walls, pits that swallow
a block (soft-locking the level so it must be retried; the environment
auto-restarts the layout to keep it solvable, which costs actions).
Colors: 8 floor, 5 wall, 12 agent, 13 block, 3 socket, 15 block-on-socket,
2 pit.
"""
from __future__ import annotations

import numpy as np

from ..core import ArcEnv, ACTION1, ACTION2, ACTION3, ACTION4, canvas, rect

CELL = 5
W = 12

_MOVES = {ACTION1: (-1, 0), ACTION2: (1, 0), ACTION3: (0, -1), ACTION4: (0, 1)}


class Pusher(ArcEnv):
    env_id = "pb04"
    action_space = (ACTION1, ACTION2, ACTION3, ACTION4)

    _SPECS = {
        # level: (agent, blocks, sockets, walls, pits)
        1: ((5, 2), [(5, 5)], [(5, 8)], set(), set()),
        2: ((2, 2), [(5, 5)], [(8, 8)], set(), set()),
        3: ((2, 2), [(4, 4)], [(8, 8)], {(6, x) for x in range(3, 8)}, set()),
        4: ((6, 1), [(6, 4)], [(6, 9)], set(), {(4, 6), (8, 6)}),
        5: ((2, 2), [(4, 4), (7, 4)], [(4, 9), (7, 9)], set(), set()),
        6: ((2, 6), [(4, 5), (6, 6)], [(9, 3), (9, 9)],
            {(7, x) for x in range(5, 8)}, {(5, 9)}),
    }

    def _load_level(self, level: int) -> None:
        a, b, s, w, p = self._SPECS[level]
        self.pos = a
        self.blocks = list(b)
        self.sockets = list(s)
        self.walls = set(w)
        self.pits = set(p)

    def _free(self, c):
        return (0 <= c[0] < W and 0 <= c[1] < W and c not in self.walls
                and c not in self.blocks)

    def _apply(self, action, x, y):
        dy, dx = _MOVES[action]
        n = (self.pos[0] + dy, self.pos[1] + dx)
        if not (0 <= n[0] < W and 0 <= n[1] < W) or n in self.walls or n in self.pits:
            return False
        if n in self.blocks:
            dest = (n[0] + dy, n[1] + dx)
            if dest in self.pits:                       # block lost -> restart layout
                self._load_level(self.level)
                return False
            if not self._free(dest):
                return False
            self.blocks[self.blocks.index(n)] = dest
        self.pos = n
        return all(b in self.sockets for b in self.blocks)

    def _render(self):
        g = canvas(0)
        rect(g, 2, 2, W * CELL, W * CELL, 8)
        for (yy, xx) in self.walls:
            rect(g, 2 + yy * CELL, 2 + xx * CELL, CELL, CELL, 5)
        for (yy, xx) in self.pits:
            rect(g, 2 + yy * CELL, 2 + xx * CELL, CELL, CELL, 2)
        for (yy, xx) in self.sockets:
            rect(g, 2 + yy * CELL + 1, 2 + xx * CELL + 1, CELL - 2, CELL - 2, 3)
        for b in self.blocks:
            yy, xx = b
            c = 15 if b in self.sockets else 13
            rect(g, 2 + yy * CELL, 2 + xx * CELL, CELL, CELL, c)
        ay, ax = self.pos
        rect(g, 2 + ay * CELL, 2 + ax * CELL, CELL, CELL, 12)
        g[2 + ay * CELL + 2, 2 + ax * CELL + 2] = 0
        return g
