"""fl03 "Conduit": click environment about connectedness (topology prior).

A source and a sink sit on a logical grid; clicking an empty cell places a
conduit segment, clicking a conduit removes it. The level is won when the
conduit 4-connects source to sink. Later levels add rock cells that reject
placement, and a "ripple" mechanic where a click also deposits conduit on the
four orthogonal neighbours (the clicked cell itself still toggles).
A per-level click budget guards against blind flooding: exceeding it wipes
the board clean (the level itself never ends, only progress resets).
Colors: 0 bg, 8 board, 4 source, 6 sink, 11 conduit, 5 rock.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from ..core import ArcEnv, canvas, rect, border

CELL = 5
W = 12
OFF = 2            # pixel offset


class Conduit(ArcEnv):
    env_id = "fl03"
    action_space = ()
    uses_click = True

    _SPECS = {
        # level: (source, sink, rocks, ripple, click budget)
        1: ((5, 2), (5, 5), set(), False, 6),
        2: ((2, 2), (9, 9), set(), False, 17),
        3: ((2, 2), (9, 9), {(5, x) for x in range(1, 8)}, False, 17),
        4: ((5, 1), (5, 10), {(y, 5) for y in range(2, 10)}, False, 20),
        5: ((2, 2), (9, 9), set(), True, 11),
        6: ((9, 1), (2, 10), {(4, 4), (5, 5), (6, 6), (7, 7), (3, 8)}, True, 12),
    }

    def _load_level(self, level: int) -> None:
        src, snk, rocks, ripple, budget = self._SPECS[level]
        self.src, self.snk = src, snk
        self.rocks = set(rocks) - {src, snk}
        self.ripple = ripple
        self.budget = budget
        self.clicks_used = 0
        self.cond = np.zeros((W, W), dtype=bool)

    def _cell_at(self, px, py):
        cx, cy = (px - OFF) // CELL, (py - OFF) // CELL
        if 0 <= cx < W and 0 <= cy < W:
            return int(cy), int(cx)
        return None

    def _blocked(self, y, x):
        return (y, x) in self.rocks or (y, x) in (self.src, self.snk)

    def _apply(self, action, x, y):
        c = self._cell_at(x, y)
        if c is None:
            return False
        cy, cx = c
        self.clicks_used += 1
        if self.clicks_used > self.budget:          # over budget: board wipes
            self.cond[:] = False
            self.clicks_used = 0
            return False
        if not self._blocked(cy, cx):
            self.cond[cy, cx] = not self.cond[cy, cx]
        if self.ripple:
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < W and 0 <= nx < W and not self._blocked(ny, nx):
                    self.cond[ny, nx] = True
        return self._connected()

    def _connected(self):
        passable = lambda p: p == self.snk or (0 <= p[0] < W and 0 <= p[1] < W
                                               and self.cond[p])
        q, seen = deque([self.src]), {self.src}
        while q:
            y, x = q.popleft()
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                n = (y + dy, x + dx)
                if n in seen or not passable(n):
                    continue
                if n == self.snk:
                    return True
                seen.add(n)
                q.append(n)
        return False

    def _render(self):
        g = canvas(0)
        rect(g, OFF, OFF, W * CELL, W * CELL, 8)
        for (yy, xx) in self.rocks:
            rect(g, OFF + yy * CELL, OFF + xx * CELL, CELL, CELL, 5)
        ys, xs = np.where(self.cond)
        for yy, xx in zip(ys, xs):
            rect(g, OFF + yy * CELL + 1, OFF + xx * CELL + 1, CELL - 2, CELL - 2, 11)
        sy, sx = self.src
        rect(g, OFF + sy * CELL, OFF + sx * CELL, CELL, CELL, 4)
        ky, kx = self.snk
        rect(g, OFF + ky * CELL, OFF + kx * CELL, CELL, CELL, 6)
        border(g, 0, 0, 64, 64, 8)
        return g
