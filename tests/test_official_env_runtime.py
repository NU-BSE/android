from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

import numpy as np
import pytest

from drlqn.config import drlqn_xs
from drlqn.env.arcade_adapter import ArcadeEnvAdapter
from drlqn.train.submission_solver import SubmissionDRLQNSolver

arcengine = pytest.importorskip("arcengine")
ARCBaseGame = arcengine.ARCBaseGame
ActionInput = arcengine.ActionInput
GameAction = arcengine.GameAction


def _game_files() -> list[Path]:
    root = Path("arc-prize-2026-arc-agi-3/environment_files")
    return sorted(path for path in root.glob("*/*/*.py") if not path.name.startswith("_"))


def _load_game(path: Path):
    module_name = f"official_env_{path.parent.parent.name}_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    candidates = [
        value
        for value in vars(module).values()
        if inspect.isclass(value)
        and value.__module__ == module.__name__
        and issubclass(value, ARCBaseGame)
        and value is not ARCBaseGame
    ]
    assert len(candidates) == 1, f"{path}: expected one ARCBaseGame subclass"
    return candidates[0]()


class DirectArcadeEnvironment:
    """Minimal local equivalent of ``arc_agi.Arcade.make`` for runtime tests."""

    def __init__(self, game):
        self.game = game
        self._last_response = None

    def step(self, action, data):
        response = self.game.perform_action(
            ActionInput(id=action, data=dict(data)),
            raw=True,
        )
        self._last_response = response
        return response


@pytest.mark.parametrize("path", _game_files(), ids=lambda path: path.stem)
def test_official_environment_executes_one_submission_solver_action(path: Path):
    game = _load_game(path)
    assert str(game.game_id) == path.stem
    actions = getattr(game, "available_actions", None)
    if actions is None:
        actions = getattr(game, "_available_actions", None)
    assert actions is not None, f"{path}: no available action declaration"

    raw_env = DirectArcadeEnvironment(game)
    env = ArcadeEnvAdapter(raw_env, path.stem, GameAction)
    solver = SubmissionDRLQNSolver(drlqn_xs(), seed=0, history_len=2)
    budgets = {level: (1 if level == 1 else 0) for level in range(1, 129)}
    result = solver.run_env(env, budgets)

    assert result.total_actions == 1
    assert len(solver.replay.buf) == 1
    assert raw_env._last_response is not None
    grid = np.asarray(raw_env._last_response.frame[-1], dtype=np.uint8)
    assert grid.shape == (64, 64)
    assert env.state in {"NOT_FINISHED", "GAME_OVER", "WIN"}
    assert solver.replay.buf[0].lam in np.where(next(iter(solver.nodes.values())).valid)[0]
