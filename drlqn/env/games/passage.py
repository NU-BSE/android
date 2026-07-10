"""xg01 "Passage": movement environment (keys ACTION1..ACTION4).

Mechanics composed across levels: navigation, walls, pressure pads that arm
the exit, teleporter pairs, hazard cells that reset the level.
Colors: 0 void, 8 floor, 5 wall, 12 agent, 3 armed exit, 7 disarmed exit,
9 pad, 10 teleporter, 2 hazard.
"""
from __future__ import annotations

import numpy as np

from ..core import ArcEnv, ACTION1, ACTION2, ACTION3, ACTION4, canvas, rect

CELL = 4          # logical 16x16 grid rendered at 4px per cell
W = 16

FLOOR, WALL, AGENT, EXIT_ON, EXIT_OFF, PAD, TELE, HAZ = 8, 5, 12, 3, 7, 9, 10, 2

_MOVES = {ACTION1: (-1, 0), ACTION2: (1, 0), ACTION3: (0, -1), ACTION4: (0, 1)}


def _maze_walls(rng, density):
    walls = set()
    for _ in range(density):
        y, x = rng.integers(1, W - 1, size=2)
        walls.add((int(y), int(x)))
    return walls


class Passage(ArcEnv):
    env_id = "xg01"
    action_space = (ACTION1, ACTION2, ACTION3, ACTION4)

    def _load_level(self, level: int) -> None:
        rng = np.random.default_rng(1000 + level)
        self.walls = set()
        self.pads, self.pads_hit = [], set()
        self.tele = {}
        self.hazards = set()
        self.pos = (2, 2)
        self.goal = (W - 3, W - 3)
        if level == 1:                                    # tutorial: open room
            self.goal = (2, 8)
        elif level == 2:                                  # walls
            self.walls = {(y, 7) for y in range(1, 11)} | {(10, x) for x in range(7, 13)}
        elif level == 3:                                  # one pad arms exit
            self.walls = _maze_walls(rng, 26) - {(2, 2), (W - 3, W - 3)}
            self.pads = [(12, 3)]
        elif level == 4:                                  # two pads
            self.walls = _maze_walls(rng, 30) - {(2, 2), (W - 3, W - 3)}
            self.pads = [(3, 12), (12, 3)]
        elif level == 5:                                  # teleporters
            self.walls = {(y, 8) for y in range(0, W)}    # impassable wall
            self.tele = {(7, 4): (7, 12), (7, 12): (7, 4)}
            self.pads = [(2, 12)]
        else:                                             # hazards + pads + maze
            self.walls = _maze_walls(rng, 24) - {(2, 2), (W - 3, W - 3)}
            self.hazards = {(5, 5), (9, 8), (4, 11), (11, 12)} - self.walls
            self.pads = [(13, 2), (2, 13)]
        for p in self.pads + [self.pos, self.goal]:
            self.walls.discard(p)
        for p in list(self.tele):
            self.walls.discard(p)
            self.walls.discard(self.tele[p])
        self.hazards -= {self.pos, self.goal, *self.pads}

    @property
    def armed(self):
        return len(self.pads_hit) == len(self.pads)

    def _apply(self, action, x, y):
        dy, dx = _MOVES[action]
        ny, nx = self.pos[0] + dy, self.pos[1] + dx
        if not (0 <= ny < W and 0 <= nx < W) or (ny, nx) in self.walls:
            return False
        if (ny, nx) in self.hazards:                      # hazard: level restart
            lvl = self.level
            self._load_level(lvl)
            return False
        if (ny, nx) in self.tele:
            ny, nx = self.tele[(ny, nx)]
        self.pos = (ny, nx)
        if self.pos in self.pads:
            self.pads_hit.add(self.pos)
        return self.pos == self.goal and self.armed

    def _render(self):
        g = canvas(0)
        rect(g, 0, 0, W * CELL, W * CELL, FLOOR)
        for (yy, xx) in self.walls:
            rect(g, yy * CELL, xx * CELL, CELL, CELL, WALL)
        for (yy, xx) in self.hazards:
            rect(g, yy * CELL, xx * CELL, CELL, CELL, HAZ)
        for p in self.tele:
            yy, xx = p
            rect(g, yy * CELL + 1, xx * CELL + 1, CELL - 2, CELL - 2, TELE)
        for p in self.pads:
            yy, xx = p
            c = FLOOR if p in self.pads_hit else PAD
            rect(g, yy * CELL + 1, xx * CELL + 1, CELL - 2, CELL - 2, c)
        gy, gx = self.goal
        rect(g, gy * CELL, gx * CELL, CELL, CELL, EXIT_ON if self.armed else EXIT_OFF)
        ay, ax = self.pos
        rect(g, ay * CELL, ax * CELL, CELL, CELL, AGENT)
        g[ay * CELL + 1, ax * CELL + 1] = 0               # inner marker: objectness
        return g
