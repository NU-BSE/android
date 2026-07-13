"""Load checked-in ARC-AGI-3 environment implementations for replay/benchmarks."""
from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
import random
from typing import Any

import numpy as np

from .arcade_adapter import ArcadeEnvAdapter


def load_arcengine():
    try:
        import arcengine
    except ImportError as exc:
        raise RuntimeError("Install arcengine to replay checked-in official games") from exc
    return arcengine


def environment_files(root: str | Path, game_code: str | None = None) -> list[Path]:
    root = Path(root)
    if game_code:
        exact = sorted(root.glob(f"{game_code}/*/{game_code}.py"))
        if exact:
            return exact
    files = sorted(path for path in root.glob("*/*/*.py") if not path.name.startswith("_"))
    if game_code:
        files = [path for path in files if path.stem == game_code]
    return files


def load_game_class(path: str | Path, base_game_cls: type) -> type:
    path = Path(path)
    module_name = f"drlqn_official_{path.parent.parent.name}_{path.parent.name}_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load environment module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    candidates = [
        value
        for value in vars(module).values()
        if inspect.isclass(value)
        and value.__module__ == module.__name__
        and issubclass(value, base_game_cls)
        and value is not base_game_cls
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"{path}: expected one ARCBaseGame subclass, found {len(candidates)}"
        )
    return candidates[0]


class DirectArcadeEnvironment:
    """Local implementation of the competition Arcade step contract."""

    def __init__(self, game: Any, action_input_cls: Any):
        self.game = game
        self.ActionInput = action_input_cls
        self._last_response = None

    def step(self, action: Any, data: dict[str, int]):
        response = self.game.perform_action(
            self.ActionInput(id=action, data=dict(data)),
            raw=True,
        )
        self._last_response = response
        return response


def seed_process(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))


def make_arcade_env(path: str | Path, arcengine: Any, seed: int = 0) -> ArcadeEnvAdapter:
    """Instantiate one local official game under a reproducible process seed."""
    path = Path(path)
    seed_process(seed)
    game_cls = load_game_class(path, arcengine.ARCBaseGame)
    game = game_cls()
    raw_env = DirectArcadeEnvironment(game, arcengine.ActionInput)
    return ArcadeEnvAdapter(raw_env, path.stem, arcengine.GameAction)
