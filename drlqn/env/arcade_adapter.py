"""Adapter from the ARC Prize ``arc_agi.Arcade`` environment to DRLQN's API.

Imports of ``arc_agi`` and ``arcengine`` stay outside this module so the core
package and unit tests remain usable without the competition wheels. Pass the
``GameAction`` enum from the notebook when constructing the adapter.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .core import CLICK, RESET, Frame, StepResult


class ArcadeEnvAdapter:
    """Expose a competition Arcade environment as a local-style ``ArcEnv``.

    ``response.available_actions`` is authoritative and refreshed after every
    step. This matters because several official games expose ACTION7 and some
    games dynamically restrict the currently legal subset.
    """

    def __init__(self, env: Any, game_id: str, game_action_cls: Any):
        self.env = env
        self.env_id = str(game_id)
        self.GameAction = game_action_cls
        self.level = 1
        self.state = "NOT_PLAYED"
        self.action_count = 0
        self._level_action_counts: dict[int, int] = {}
        self._levels_completed = 0
        self.action_space: tuple[int, ...] = ()
        self.uses_click = False
        self._response = getattr(env, "_last_response", None)
        if self._response is not None:
            self._ingest(self._response)

    @staticmethod
    def _state_name(response: Any) -> str:
        state = getattr(response, "state", None)
        return str(getattr(state, "name", state or "NOT_PLAYED")).upper()

    @staticmethod
    def _last_grid(response: Any) -> np.ndarray:
        frames = getattr(response, "frame", None) or []
        if not frames:
            return np.zeros((64, 64), dtype=np.uint8)
        grid = np.asarray(frames[-1], dtype=np.uint8)
        if grid.shape != (64, 64):
            padded = np.zeros((64, 64), dtype=np.uint8)
            height = min(64, grid.shape[0])
            width = min(64, grid.shape[1])
            padded[:height, :width] = grid[:height, :width]
            grid = padded
        return grid

    @staticmethod
    def _action_id(value: Any) -> int | None:
        raw = getattr(value, "value", value)
        try:
            return int(raw)
        except (TypeError, ValueError):
            return None

    def _available_action_ids(self, response: Any) -> tuple[int, ...]:
        values = getattr(response, "available_actions", None)
        if values is None:
            values = getattr(self.env, "available_actions", None)
        if values is None:
            # Conservative compatibility fallback for older clients. Invalid
            # actions should be rejected by the environment, but current Arcade
            # responses expose ``available_actions`` so this path is exceptional.
            values = [action for action in self.GameAction]
        ids = {
            action_id
            for value in values
            for action_id in [self._action_id(value)]
            if action_id is not None and action_id != RESET
        }
        return tuple(sorted(ids))

    def _ingest(self, response: Any) -> Frame:
        self._response = response
        completed = int(getattr(response, "levels_completed", 0) or 0)
        state_name = self._state_name(response)
        if state_name == "WIN":
            self.state = "WIN"
            self.level = max(1, completed)
        elif state_name == "GAME_OVER":
            self.state = "GAME_OVER"
            self.level = max(1, completed + 1)
        elif state_name == "NOT_PLAYED":
            self.state = "NOT_PLAYED"
            self.level = max(1, completed + 1)
        else:
            self.state = "NOT_FINISHED"
            self.level = max(1, completed + 1)
        self._levels_completed = completed
        self.action_space = self._available_action_ids(response)
        self.uses_click = CLICK in self.action_space
        return Frame(
            grid=self._last_grid(response),
            state=self.state,
            level=self.level,
            score=completed,
        )

    def _game_action(self, action_id: int):
        if hasattr(self.GameAction, "from_id"):
            return self.GameAction.from_id(int(action_id))
        return self.GameAction(int(action_id))

    def reset(self) -> Frame:
        response = self._response or getattr(self.env, "_last_response", None)
        if response is None or self._state_name(response) in {"NOT_PLAYED", "GAME_OVER"}:
            response = self.env.step(self._game_action(RESET), {})
            if response is None:
                raise RuntimeError(f"{self.env_id}: RESET returned no response")
        self._levels_completed = int(getattr(response, "levels_completed", 0) or 0)
        return self._ingest(response)

    def reset_level(self) -> Frame:
        level_before = self.level
        self._count_action(level_before)
        response = self.env.step(self._game_action(RESET), {})
        if response is None:
            raise RuntimeError(f"{self.env_id}: RESET returned no response")
        return self._ingest(response)

    def step(self, action: int, x: int | None = None, y: int | None = None) -> StepResult:
        if self.state not in {"NOT_FINISHED", "PLAYING"}:
            raise RuntimeError(f"{self.env_id}: cannot step terminal state {self.state}")
        action = int(action)
        if self.action_space and action not in self.action_space:
            raise ValueError(
                f"{self.env_id}: ACTION{action} is not currently available; "
                f"valid={self.action_space}"
            )

        level_before = self.level
        completed_before = self._levels_completed
        data: dict[str, int] = {}
        if action == CLICK:
            if x is None or y is None:
                raise ValueError("ACTION6 requires x and y")
            data = {
                "x": int(np.clip(x, 0, 63)),
                "y": int(np.clip(y, 0, 63)),
            }

        self._count_action(level_before)
        response = self.env.step(self._game_action(action), data)
        if response is None:
            raise RuntimeError(f"{self.env_id}: ACTION{action} returned no response")
        frame = self._ingest(response)
        completed_after = self._levels_completed
        level_won = completed_after > completed_before
        env_won = self.state == "WIN"
        return StepResult(frame=frame, level_won=level_won, env_won=env_won)

    def _count_action(self, level: int) -> None:
        self.action_count += 1
        self._level_action_counts[int(level)] = (
            self._level_action_counts.get(int(level), 0) + 1
        )

    @property
    def actions_per_level(self) -> dict[int, int]:
        return dict(self._level_action_counts)
