from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

import numpy as np
import pytest

from drlqn.env.arcade_adapter import ArcadeEnvAdapter

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
def test_official_environment_executes_through_submission_adapter(path: Path):
    game = _load_game(path)
    assert str(game.game_id) == path.stem
    actions = getattr(game, "available_actions", None)
    if actions is None:
        actions = getattr(game, "_available_actions", None)
    assert actions is not None, f"{path}: no available action declaration"

    raw_env = DirectArcadeEnvironment(game)
    env = ArcadeEnvAdapter(raw_env, path.stem, GameAction)
    initial = env.reset()
    assert initial.grid.shape == (64, 64)
    assert initial.grid.dtype == np.uint8
    assert env.action_space

    action = int(env.action_space[0])
    if action == 6:
        result = env.step(action, x=32, y=32)
    else:
        result = env.step(action)
    assert result.frame.grid.shape == (64, 64)
    assert result.frame.state in {"NOT_FINISHED", "GAME_OVER", "WIN"}
    assert env.action_count == 1
