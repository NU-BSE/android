from types import SimpleNamespace

import numpy as np

from drlqn.train.offline_experience import collect_trajectory, experience_summary


class FakeEnv:
    env_id = "fake"
    action_space = (1,)
    uses_click = False

    def __init__(self):
        self.level = 1
        self.state = "NOT_PLAYED"
        self.action_count = 0
        self._level_action_counts = {}
        self._levels_completed = 0
        self.grid = np.zeros((64, 64), dtype=np.uint8)

    def reset(self):
        self.state = "NOT_FINISHED"
        return SimpleNamespace(grid=self.grid.copy())

    def reset_level(self):
        return self.reset()

    def step(self, action, x=None, y=None):
        self.action_count += 1
        self._level_action_counts[1] = self.action_count
        if self.action_count == 1:
            frame = self.grid.copy()
            won = False
        else:
            frame = self.grid.copy()
            frame[0, 0] = 1
            self.grid = frame
            self.state = "WIN"
            self._levels_completed = 1
            won = True
        return SimpleNamespace(
            frame=SimpleNamespace(grid=frame),
            level_won=won,
            env_won=won,
        )


class FakeSolver:
    def __init__(self):
        self.rng = np.random.default_rng(0)
        self.graph = {}
        self.edge_stats = {}
        self._pending_lambda = None

    def perceive(self, env, grid):
        key = str(np.asarray(grid).sum())
        return SimpleNamespace(
            key=key,
            grid=np.asarray(grid).copy(),
            valid=np.asarray([True] + [False] * 127),
            objs=[],
            level=env.level,
        )

    def _update_priors(self, cur, lam, nxt, changed):
        stats = self.edge_stats.setdefault((cur.key, lam), [0.0, 0.0, 0.0])
        stats[0] += 1
        stats[1] += int(changed)


def test_offline_collector_retains_noop_negative_evidence_and_success():
    examples = collect_trajectory(
        FakeEnv(),
        FakeSolver(),
        policy="random",
        seed=0,
        max_actions_per_level=4,
    )
    assert len(examples) == 2
    assert examples[0].visual_changed is False
    assert "negative:no_change" in examples[0].reasoning
    assert examples[-1].success is True
    summary = experience_summary(examples)
    assert summary["no_change_negative_evidence"] == 1
    assert summary["level_advances"] == 1
