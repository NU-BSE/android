"""Extract state/action supervision from Duck Harness trace exports.

The legacy loader keeps the first action after each observed grid. Dense replay
adds optional next-state and trajectory metadata to :class:`DuckExample` while
preserving that conservative behaviour for raw inference-trace JSON files.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


_ACTION_ALIASES = {
    "UP": 1,
    "ACTION1": 1,
    "DOWN": 2,
    "ACTION2": 2,
    "LEFT": 3,
    "ACTION3": 3,
    "RIGHT": 4,
    "ACTION4": 4,
    "SPACE": 5,
    "ENTER": 5,
    "SELECT": 5,
    "ACTION5": 5,
    "MOUSE": 6,
    "CLICK": 6,
    "ACTION6": 6,
    "ACTION7": 7,
    "ACTION8": 8,
    "ACTION9": 9,
}


@dataclass(frozen=True)
class DuckExample:
    game_id: str
    message_index: int
    grid: np.ndarray
    action: int
    x: int | None = None
    y: int | None = None
    reasoning: str = ""
    source: str = ""
    next_grid: np.ndarray | None = None
    visual_changed: bool | None = None
    level_advanced: bool | None = None
    terminal_result: str = ""
    remaining_path_length: int | None = None
    seed: int | None = None
    success: bool | None = None
    trajectory_id: str = ""
    level_before: int | None = None
    level_after: int | None = None
    action_index: int = 0


def _call_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return ""


def _literal_action_items(code: str) -> list[Any]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    calls = sorted(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and _call_name(node) == "action" and node.args
        ),
        key=lambda node: (getattr(node, "lineno", 0), getattr(node, "col_offset", 0)),
    )
    items: list[Any] = []
    for call in calls:
        try:
            value = ast.literal_eval(call.args[0])
        except (ValueError, TypeError, SyntaxError):
            continue
        if isinstance(value, list):
            items.extend(value)
        else:
            items.append(value)
    return items


def normalize_action(item: Any) -> tuple[int, int | None, int | None] | None:
    """Normalize a Duck/Arcade action representation.

    Duck click dictionaries use matrix coordinates ``row, col`` whereas the
    Arcade API expects ``x=col, y=row``.
    """
    if isinstance(item, str):
        action = _ACTION_ALIASES.get(item.strip().upper())
        return None if action is None else (action, None, None)
    if isinstance(item, (int, np.integer)):
        action = int(item)
        return (action, None, None) if 1 <= action <= 9 else None
    if not isinstance(item, dict):
        return None

    raw = item.get("action", item.get("name", item.get("id", item.get("action_id"))))
    if isinstance(raw, (int, np.integer)):
        action = int(raw)
    else:
        action = _ACTION_ALIASES.get(str(raw or "").strip().upper())
    if action is None or not 1 <= action <= 9:
        return None
    if action != 6:
        return action, None, None

    data = item.get("data") if isinstance(item.get("data"), dict) else item
    row = data.get("row", data.get("y"))
    col = data.get("col", data.get("x"))
    if row is None or col is None:
        return None
    try:
        return action, int(col), int(row)
    except (TypeError, ValueError):
        return None


def actions_from_python(code: str) -> list[tuple[int, int | None, int | None]]:
    result = []
    for item in _literal_action_items(code):
        normalized = normalize_action(item)
        if normalized is not None:
            result.append(normalized)
    return result


def _python_code(tool_call: dict[str, Any]) -> str | None:
    function = tool_call.get("function") or {}
    if str(function.get("name") or "") != "python":
        return None
    arguments = function.get("arguments")
    if isinstance(arguments, dict):
        payload = arguments
    else:
        try:
            payload = json.loads(str(arguments or "{}"))
        except json.JSONDecodeError:
            return None
    code = payload.get("code")
    return str(code) if code is not None else None


def load_duck_trace(path: str | Path) -> list[DuckExample]:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    trace_id = str(payload.get("id") or path.stem)
    game_id = trace_id.rsplit("/", 1)[-1]
    messages = payload.get("messages") or []

    current_grid: np.ndarray | None = None
    grid_consumed = True
    examples: list[DuckExample] = []
    for index, message in enumerate(messages):
        grid = message.get("grid") if isinstance(message, dict) else None
        if isinstance(grid, list):
            array = np.asarray(grid, dtype=np.uint8)
            if array.shape == (64, 64):
                current_grid = array
                grid_consumed = False

        if (
            current_grid is None
            or grid_consumed
            or not isinstance(message, dict)
            or message.get("role") != "assistant"
        ):
            continue
        for tool_call in message.get("tool_calls") or []:
            code = _python_code(tool_call)
            if not code:
                continue
            actions = actions_from_python(code)
            if not actions:
                continue
            action, x, y = actions[0]
            examples.append(
                DuckExample(
                    game_id=game_id,
                    message_index=index,
                    grid=current_grid.copy(),
                    action=action,
                    x=x,
                    y=y,
                    reasoning=str(message.get("reasoning") or ""),
                    source=str(path),
                    trajectory_id=trace_id,
                )
            )
            grid_consumed = True
            break
    return examples


def load_examples_npz(path: str | Path) -> list[DuckExample]:
    """Load dense replay output, accepting older conservative NPZ exports too."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        grids = archive["grids"]
        count = int(grids.shape[0])

        def values(name: str, default: Any) -> np.ndarray:
            if name in archive.files:
                return archive[name]
            return np.asarray([default] * count)

        actions = values("actions", -1)
        xs = values("xs", -1)
        ys = values("ys", -1)
        games = values("games", "")
        indices = values("message_indices", 0)
        sources = values("sources", str(path))
        reasoning = values("reasoning", "")
        has_next = values("has_next", False).astype(bool)
        next_grids = archive["next_grids"] if "next_grids" in archive.files else grids
        changed = values("visual_changed", -1)
        advanced = values("level_advanced", -1)
        terminal = values("terminal_results", "")
        remaining = values("remaining_path_lengths", -1)
        seeds = values("seeds", -1)
        success = values("success", -1)
        trajectories = values("trajectory_ids", "")
        levels_before = values("levels_before", -1)
        levels_after = values("levels_after", -1)
        action_indices = values("action_indices", 0)

        result: list[DuckExample] = []
        for i in range(count):
            result.append(
                DuckExample(
                    game_id=str(games[i]),
                    message_index=int(indices[i]),
                    grid=np.asarray(grids[i], dtype=np.uint8).copy(),
                    action=int(actions[i]),
                    x=None if int(xs[i]) < 0 else int(xs[i]),
                    y=None if int(ys[i]) < 0 else int(ys[i]),
                    reasoning=str(reasoning[i]),
                    source=str(sources[i]),
                    next_grid=(
                        np.asarray(next_grids[i], dtype=np.uint8).copy()
                        if bool(has_next[i])
                        else None
                    ),
                    visual_changed=None if int(changed[i]) < 0 else bool(changed[i]),
                    level_advanced=None if int(advanced[i]) < 0 else bool(advanced[i]),
                    terminal_result=str(terminal[i]),
                    remaining_path_length=(
                        None if int(remaining[i]) < 0 else int(remaining[i])
                    ),
                    seed=None if int(seeds[i]) < 0 else int(seeds[i]),
                    success=None if int(success[i]) < 0 else bool(success[i]),
                    trajectory_id=str(trajectories[i]),
                    level_before=(
                        None if int(levels_before[i]) < 0 else int(levels_before[i])
                    ),
                    level_after=(
                        None if int(levels_after[i]) < 0 else int(levels_after[i])
                    ),
                    action_index=int(action_indices[i]),
                )
            )
    return result


def load_duck_traces(paths: Iterable[str | Path]) -> list[DuckExample]:
    examples: list[DuckExample] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            files = sorted([*path.rglob("*.json"), *path.rglob("*.npz")])
        else:
            files = [path]
        for file_path in files:
            try:
                if file_path.suffix.lower() == ".npz":
                    examples.extend(load_examples_npz(file_path))
                else:
                    examples.extend(load_duck_trace(file_path))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError, OSError):
                continue
    return examples


def save_examples_npz(examples: list[DuckExample], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not examples:
        raise ValueError("cannot save an empty Duck example set")
    grids = np.stack([example.grid for example in examples]).astype(np.uint8)
    has_next = np.asarray([example.next_grid is not None for example in examples], dtype=bool)
    next_grids = np.stack(
        [example.next_grid if example.next_grid is not None else example.grid for example in examples]
    ).astype(np.uint8)
    payload = {
        "grids": grids,
        "next_grids": next_grids,
        "has_next": has_next,
        "actions": np.asarray([example.action for example in examples], dtype=np.int16),
        "xs": np.asarray([example.x if example.x is not None else -1 for example in examples], dtype=np.int16),
        "ys": np.asarray([example.y if example.y is not None else -1 for example in examples], dtype=np.int16),
        "games": np.asarray([example.game_id for example in examples]),
        "message_indices": np.asarray([example.message_index for example in examples], dtype=np.int32),
        "sources": np.asarray([example.source for example in examples]),
        "reasoning": np.asarray([example.reasoning for example in examples]),
        "visual_changed": np.asarray([
            -1 if example.visual_changed is None else int(example.visual_changed)
            for example in examples
        ], dtype=np.int8),
        "level_advanced": np.asarray([
            -1 if example.level_advanced is None else int(example.level_advanced)
            for example in examples
        ], dtype=np.int8),
        "terminal_results": np.asarray([example.terminal_result for example in examples]),
        "remaining_path_lengths": np.asarray([
            -1 if example.remaining_path_length is None else example.remaining_path_length
            for example in examples
        ], dtype=np.int32),
        "seeds": np.asarray([
            -1 if example.seed is None else example.seed for example in examples
        ], dtype=np.int32),
        "success": np.asarray([
            -1 if example.success is None else int(example.success) for example in examples
        ], dtype=np.int8),
        "trajectory_ids": np.asarray([example.trajectory_id for example in examples]),
        "levels_before": np.asarray([
            -1 if example.level_before is None else example.level_before
            for example in examples
        ], dtype=np.int16),
        "levels_after": np.asarray([
            -1 if example.level_after is None else example.level_after
            for example in examples
        ], dtype=np.int16),
        "action_indices": np.asarray([example.action_index for example in examples], dtype=np.int32),
    }
    np.savez_compressed(path, **payload)
    return path
