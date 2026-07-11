from __future__ import annotations

import numpy as np

from drlqn.config import drlqn_xs
from drlqn.env.core import Frame, StepResult
from drlqn.train.submission_solver import SubmissionDRLQNSolver


class NoOpEnvironment:
    env_id = "noop"
    level = 1
    state = "NOT_FINISHED"
    action_count = 0
    action_space = (7,)
    uses_click = False

    def __init__(self):
        self.grid = np.zeros((64, 64), dtype=np.uint8)
        self._level_action_counts: dict[int, int] = {}
        self.last_action = None

    def reset(self) -> Frame:
        self.state = "NOT_FINISHED"
        return Frame(self.grid.copy(), self.state, self.level, 0)

    def reset_level(self) -> Frame:
        self.action_count += 1
        self._level_action_counts[self.level] = (
            self._level_action_counts.get(self.level, 0) + 1
        )
        return Frame(self.grid.copy(), self.state, self.level, 0)

    def step(self, action: int, x=None, y=None) -> StepResult:
        self.last_action = int(action)
        self.action_count += 1
        self._level_action_counts[self.level] = (
            self._level_action_counts.get(self.level, 0) + 1
        )
        frame = Frame(self.grid.copy(), self.state, self.level, 0)
        return StepResult(frame=frame, level_won=False, env_won=False)

    @property
    def actions_per_level(self):
        return dict(self._level_action_counts)


def test_history_node_change_does_not_turn_visual_noop_into_novelty():
    env = NoOpEnvironment()
    solver = SubmissionDRLQNSolver(drlqn_xs(), seed=0, history_len=2)

    result = solver.run_env(env, {1: 1})

    assert result.total_actions == 1
    assert env.last_action == 7
    assert len(solver.replay.buf) == 1
    transition = solver.replay.buf[0]
    assert transition.key != transition.next_key  # history-conditioned graph node
    assert not transition.changed                 # rendered frame did not change
    assert not transition.stable_pair
    assert np.isclose(transition.reward, -0.32)
