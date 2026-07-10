"""Scripted reference solvers.

Used for (a) automated validation that every level is solvable (report 3.5),
and (b) calibrating the action-efficiency baseline used by RHAE scoring.
The real benchmark uses the upper-median best *first-run human*; humans spend
extra actions on exploration and mechanics learning (report 5.3.2), so the
surrogate baseline is
    h = ceil(1.75 * optimal) + 5
per level (documented deviation: no human cohort in this sandbox).
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np

from ..env.core import ACTION1, ACTION2, ACTION3, ACTION4, CLICK
from ..env.games import make
from ..env.games.rotor import TILE, GAP
from ..env.games.chroma import PALETTE, BTN, BTN_Y

_MOVES = {ACTION1: (-1, 0), ACTION2: (1, 0), ACTION3: (0, -1), ACTION4: (0, 1)}


# ------------------------------------------------------------------- passage
def solve_passage_level(env):
    W = 16
    start = (env.pos, frozenset(env.pads_hit))
    q = deque([start])
    prev = {start: None}
    goal_state = None
    while q and goal_state is None:
        (pos, hit) = q.popleft()
        for a, (dy, dx) in _MOVES.items():
            ny, nx = pos[0] + dy, pos[1] + dx
            if not (0 <= ny < W and 0 <= nx < W) or (ny, nx) in env.walls:
                continue
            if (ny, nx) in env.hazards:
                continue
            if (ny, nx) in env.tele:
                ny, nx = env.tele[(ny, nx)]
            nhit = hit | ({(ny, nx)} if (ny, nx) in env.pads else set())
            ns = ((ny, nx), frozenset(nhit))
            if ns in prev:
                continue
            prev[ns] = ((pos, hit), a)
            if (ny, nx) == env.goal and len(nhit) == len(env.pads):
                goal_state = ns
                break
            q.append(ns)
    assert goal_state is not None, "passage level unsolvable"
    seq = []
    s = goal_state
    while prev[s] is not None:
        s, a = prev[s]
        seq.append((a, None, None))
    return list(reversed(seq))


# -------------------------------------------------------------------- pusher
def solve_pusher_level(env):
    W = 12
    start = (env.pos, tuple(sorted(env.blocks)))
    q = deque([start])
    prev = {start: None}
    goal_state = None
    sockets = set(env.sockets)
    while q and goal_state is None:
        pos, blocks = q.popleft()
        bset = set(blocks)
        for a, (dy, dx) in _MOVES.items():
            n = (pos[0] + dy, pos[1] + dx)
            if not (0 <= n[0] < W and 0 <= n[1] < W) or n in env.walls or n in env.pits:
                continue
            nb = list(blocks)
            if n in bset:
                dest = (n[0] + dy, n[1] + dx)
                if (dest in env.pits or not (0 <= dest[0] < W and 0 <= dest[1] < W)
                        or dest in env.walls or dest in bset):
                    continue
                nb[nb.index(n)] = dest
            ns = (n, tuple(sorted(nb)))
            if ns in prev:
                continue
            prev[ns] = ((pos, blocks), a)
            if all(b in sockets for b in nb):
                goal_state = ns
                break
            q.append(ns)
    assert goal_state is not None, "pusher level unsolvable"
    seq = []
    s = goal_state
    while prev[s] is not None:
        s, a = prev[s]
        seq.append((a, None, None))
    return list(reversed(seq))


# --------------------------------------------------------------------- rotor
def solve_rotor_level(env):
    n = env.n
    best = None
    for combo in np.ndindex(*([4] * n)):
        rot = list(env.rot)
        for i, c in enumerate(combo):
            rot[i] = (rot[i] + c) % 4
            if i in env.linked:
                rot[env.linked[i]] = (rot[env.linked[i]] + c) % 4
        if all(r == 0 for r in rot):
            if best is None or sum(combo) < sum(best):
                best = combo
    assert best is not None, "rotor level unsolvable"
    seq = []
    for i, c in enumerate(best):
        x0 = 4 + i * (TILE + GAP) + TILE // 2
        for _ in range(c):
            seq.append((CLICK, x0, 26 + TILE // 2))
    return seq


# ------------------------------------------------------------------- conduit
def solve_conduit_level(env):
    Wd = 12
    # shortest open path source -> sink
    q = deque([env.src])
    prev = {env.src: None}
    while q:
        cur = q.popleft()
        if cur == env.snk:
            break
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (cur[0] + dy, cur[1] + dx)
            if (n in prev or not (0 <= n[0] < Wd and 0 <= n[1] < Wd)
                    or n in env.rocks):
                continue
            prev[n] = cur
            q.append(n)
    assert env.snk in prev, "conduit level unsolvable"
    path = []
    c = prev[env.snk]
    while c is not None and c != env.src:
        path.append(c)
        c = prev[c]
    path.reverse()
    seq = []
    cond = env.cond.copy()
    for (cy, cx) in path:
        if cond[cy, cx]:
            continue
        seq.append((CLICK, 2 + cx * 5 + 2, 2 + cy * 5 + 2))
        cond[cy, cx] = True
        if env.ripple:
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < Wd and 0 <= nx < Wd and (ny, nx) not in env.rocks \
                        and (ny, nx) not in (env.src, env.snk):
                    cond[ny, nx] = True
    return seq


# -------------------------------------------------------------------- chroma
def solve_chroma_level(env):
    seq = []
    for cue in env.seq:
        b = cue if env.legend is None else env.legend[cue]
        x0 = 4 + b * (BTN + 5) + BTN // 2
        seq.append((CLICK, x0, BTN_Y + BTN // 2))
    return seq


SOLVERS = {
    "xg01": solve_passage_level,
    "pb04": solve_pusher_level,
    "rt02": solve_rotor_level,
    "fl03": solve_conduit_level,
    "sq05": solve_chroma_level,
}


def optimal_actions(env_id: str) -> dict[int, int]:
    """Play every level with the scripted solver; return optimal action counts."""
    env = make(env_id)
    env.reset()
    out = {}
    solver = SOLVERS[env_id]
    while env.state == "NOT_FINISHED":
        level = env.level
        seq = solver(env)
        out[level] = len(seq)
        for (a, x, y) in seq:
            res = env.step(a, x, y)
        assert res.level_won, f"{env_id} L{level}: scripted solution failed"
        if res.env_won:
            break
    return out


def human_baseline(env_id: str) -> dict[int, int]:
    return {lvl: math.ceil(1.75 * n) + 5 for lvl, n in optimal_actions(env_id).items()}
