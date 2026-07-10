"""ARC-AGI-3 style environment core.

Format follows the ARC-AGI-3 technical report:
  * observation: 64x64 grid, 16 colors (ints 0..15), called a "frame";
  * turn based, deterministic: state only changes on agent actions;
  * small action space: up to five key actions (ACTION1..ACTION5),
    one coordinate action CLICK(x, y), and RESET;
  * an environment is a sequence of levels; a level ends on a win frame;
  * the agent is never told the goal - it must infer mechanics.

The real ARC-AGI-3 environments run behind three.arcprize.org and need an
ARC API key (not available in this sandbox), so this module hosts local
environments implemented in the same format; `drlqn/api/arc_client.py`
adapts the same agent to the remote API.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

GRID = 64
N_COLORS = 16

# canonical key actions
ACTION1, ACTION2, ACTION3, ACTION4, ACTION5 = 1, 2, 3, 4, 5
CLICK = 6
RESET = 0


@dataclass
class Frame:
    grid: np.ndarray                 # (64, 64) uint8
    state: str                       # NOT_PLAYED | NOT_FINISHED | WIN | GAME_OVER
    level: int
    score: int


@dataclass
class StepResult:
    frame: Frame
    level_won: bool = False
    env_won: bool = False


class ArcEnv:
    """Base class. Subclasses implement per-level logic on a 64x64 canvas."""

    env_id = "xx00"
    n_levels = 6
    #: subset of the action space this environment accepts
    action_space: tuple = (ACTION1, ACTION2, ACTION3, ACTION4)
    uses_click = False

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.level = 1
        self.action_count = 0          # first-contact action counter (never reset)
        self.state = "NOT_PLAYED"
        self._level_action_counts = {}

    # ------------------------------------------------------------ subclass API
    def _load_level(self, level: int) -> None:
        raise NotImplementedError

    def _render(self) -> np.ndarray:
        raise NotImplementedError

    def _apply(self, action: int, x: int | None, y: int | None) -> bool:
        """Mutate level state; return True if the level is now won."""
        raise NotImplementedError

    # ---------------------------------------------------------------- lifecycle
    def reset(self) -> Frame:
        """Full environment reset back to level 1 (counts as an action once playing)."""
        if self.state != "NOT_PLAYED":
            self._count_action()
        self.level = 1
        self._load_level(self.level)
        self.state = "NOT_FINISHED"
        return self._frame()

    def reset_level(self) -> Frame:
        self._count_action()
        self._load_level(self.level)
        return self._frame()

    def step(self, action: int, x: int | None = None, y: int | None = None) -> StepResult:
        assert self.state == "NOT_FINISHED", "call reset() first"
        if action == RESET:
            return StepResult(self.reset_level())
        assert action in self.action_space or (action == CLICK and self.uses_click), \
            f"invalid action {action} for {self.env_id}"
        self._count_action()
        won = self._apply(action, x, y)
        env_won = False
        if won:
            if self.level >= self.n_levels:
                self.state = "WIN"
                env_won = True
            else:
                self.level += 1
                self._load_level(self.level)
        return StepResult(self._frame(), level_won=won, env_won=env_won)

    def _count_action(self):
        self.action_count += 1
        self._level_action_counts[self.level] = self._level_action_counts.get(self.level, 0) + 1

    def _frame(self) -> Frame:
        return Frame(self._render(), self.state, self.level, self.action_count)

    # ------------------------------------------------------------------- misc
    def frame_hash(self) -> str:
        return hashlib.sha1(self._render().tobytes() + bytes([self.level])).hexdigest()[:16]

    @property
    def actions_per_level(self):
        return dict(self._level_action_counts)


# ------------------------------------------------------------------ drawing
def canvas(bg: int = 0) -> np.ndarray:
    return np.full((GRID, GRID), bg, dtype=np.uint8)


def rect(g, y, x, h, w, color):
    g[y:y + h, x:x + w] = color


def border(g, y, x, h, w, color):
    g[y, x:x + w] = color
    g[y + h - 1, x:x + w] = color
    g[y:y + h, x] = color
    g[y:y + h, x + w - 1] = color
