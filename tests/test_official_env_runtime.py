from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

import pytest

arcengine = pytest.importorskip("arcengine")
ARCBaseGame = arcengine.ARCBaseGame


def _game_files() -> list[Path]:
    root = Path("arc-prize-2026-arc-agi-3/environment_files")
    return sorted(path for path in root.glob("*/*/*.py") if not path.name.startswith("_"))


@pytest.mark.parametrize("path", _game_files(), ids=lambda path: path.stem)
def test_official_environment_imports_and_instantiates(path: Path):
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
    game = candidates[0]()

    game_id = str(getattr(game, "game_id", ""))
    assert game_id == path.stem
    actions = getattr(game, "available_actions", None)
    if actions is None:
        actions = getattr(game, "_available_actions", None)
    assert actions is not None, f"{path}: no available action declaration"
