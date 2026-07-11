from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

import numpy as np

from drlqn.config import drlqn_xs
from drlqn.env.arcade_adapter import ArcadeEnvAdapter
from drlqn.operators.library import (
    EXTRA_FIRST,
    is_click_lambda,
    lambda_for_action,
    to_env_action,
    valid_mask,
)
from drlqn.state import Obj
from drlqn.train.distill import match_teacher_action
from drlqn.train.submission_solver import SubmissionDRLQNSolver
from scripts.audit_environment_files import audit_tree


class StubEnv:
    env_id = "stub"
    level = 1
    action_count = 0
    action_space = (1, 6, 7)
    uses_click = True


def _anchor() -> Obj:
    return Obj(
        color=8,
        y0=8,
        x0=8,
        y1=10,
        x1=10,
        cy=9.0,
        cx=9.0,
        area=9,
        fill=1.0,
        perim=8 / 9,
        shape3=np.ones((3, 3)),
    )


def test_extra_action_reuses_reserved_logits_without_becoming_click():
    anchors = [_anchor()]
    mask = valid_mask(StubEnv(), anchors)
    assert mask[EXTRA_FIRST]
    assert not is_click_lambda(EXTRA_FIRST)
    assert lambda_for_action(7) == EXTRA_FIRST
    assert to_env_action(EXTRA_FIRST, anchors) == (7, None, None)
    assert match_teacher_action(7, anchors).lambda_id == EXTRA_FIRST


def test_short_history_distinguishes_same_rendered_frame():
    env = StubEnv()
    solver = SubmissionDRLQNSolver(drlqn_xs(), seed=0)
    grid = np.zeros((64, 64), dtype=np.uint8)
    grid[20:23, 20:23] = 8

    first = solver.perceive(env, grid)
    solver._pending_lambda = 0
    env.action_count = 1
    second = solver.perceive(env, grid)

    assert first.key != second.key
    assert second.trace[1] == (0 + 1) / 128.0
    assert second.trace[2] > 0.0
    assert second.raw_objects
    assert second.objs  # separate state-conditioned click anchors
    assert len(second.tokens) == 64


def test_bounded_history_revisits_state_after_window_saturates():
    env = StubEnv()
    solver = SubmissionDRLQNSolver(drlqn_xs(), seed=0, history_len=2)
    grid = np.zeros((64, 64), dtype=np.uint8)
    solver.perceive(env, grid)

    keys = []
    for count in range(1, 4):
        solver._pending_lambda = 0
        env.action_count = count
        keys.append(solver.perceive(env, grid).key)

    assert keys[-1] == keys[-2]
    assert solver._steps_in_level == 3


def test_known_winning_edge_preempts_new_exploration():
    env = StubEnv()
    solver = SubmissionDRLQNSolver(drlqn_xs(), seed=1)
    grid = np.zeros((64, 64), dtype=np.uint8)
    grid[30:34, 30:34] = 11
    node = solver.perceive(env, grid)
    solver.winning_actions[node.key] = EXTRA_FIRST

    assert solver.select(node, actions_taken=20) == EXTRA_FIRST
    assert solver.route_counts["known_win"] == 1


class FakeAction(IntEnum):
    RESET = 0
    ACTION1 = 1
    ACTION2 = 2
    ACTION3 = 3
    ACTION4 = 4
    ACTION5 = 5
    ACTION6 = 6
    ACTION7 = 7

    @classmethod
    def from_id(cls, value: int):
        return cls(value)


class FakeState(IntEnum):
    NOT_PLAYED = 0
    PLAYING = 1
    WIN = 2
    GAME_OVER = 3


@dataclass
class FakeResponse:
    state: FakeState
    levels_completed: int
    available_actions: list[FakeAction]
    frame: list[np.ndarray]


class FakeArcadeEnv:
    def __init__(self, initial: FakeResponse, responses: list[FakeResponse]):
        self._last_response = initial
        self.responses = list(responses)
        self.calls: list[tuple[FakeAction, dict]] = []

    def step(self, action, data):
        self.calls.append((action, dict(data)))
        self._last_response = self.responses.pop(0)
        return self._last_response


def test_arcade_adapter_uses_dynamic_available_actions_and_action7():
    grid = np.zeros((64, 64), dtype=np.uint8)
    initial = FakeResponse(FakeState.NOT_PLAYED, 0, [FakeAction.RESET], [grid])
    playing = FakeResponse(
        FakeState.PLAYING,
        0,
        [FakeAction.ACTION6, FakeAction.ACTION7],
        [grid],
    )
    won = FakeResponse(FakeState.WIN, 1, [], [grid])
    raw_env = FakeArcadeEnv(initial, [playing, won])
    env = ArcadeEnvAdapter(raw_env, "fake", FakeAction)

    frame = env.reset()
    assert frame.state == "NOT_FINISHED"
    assert env.action_space == (6, 7)
    result = env.step(7)
    assert result.env_won and result.level_won
    assert raw_env.calls[-1] == (FakeAction.ACTION7, {})


def test_official_environment_tree_fits_fixed_action_mapping():
    root = Path("arc-prize-2026-arc-agi-3/environment_files")
    results = audit_tree(root)
    assert len(results) == 25
    assert not [result for result in results if result.unsupported]
    action7_games = {
        result.game
        for result in results
        if 7 in result.action_ids
    }
    assert action7_games == {"ar25", "bp35", "lf52", "sb26", "sk48", "su15"}
