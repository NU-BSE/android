"""Extract state/action supervision from Duck Harness trace exports.

Duck's trace exporter stores a 64x64 grid on user messages and Python tool calls
on assistant messages. A Python call may execute several environment actions,
but only the first action after a fresh grid is a fully observed state/action
pair. This loader deliberately keeps that conservative subset for SFT.
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


def _normalize_action(item: Any) -> tuple[int, int | None, int | None] | None:
    if isinstance(item, str):
        action = _ACTION_ALIASES.get(item.strip().upper())
        return None if action is None else (action, None, None)
    if isinstance(item, int):
        return (int(item), None, None) if 1 <= int(item) <= 9 else None
    if not isinstance(item, dict):
        return None

    raw = item.get("action", item.get("name", item.get("id")))
    if isinstance(raw, int):
        action = int(raw)
    else:
        action = _ACTION_ALIASES.get(str(raw or "").strip().upper())
    if action is None or not 1 <= action <= 9:
        return None
    if action != 6:
        return action, None, None

    # Duck uses matrix coordinates (row, col); Arcade uses (x=col, y=row).
    row = item.get("row", item.get("y"))
    col = item.get("col", item.get("x"))
    if row is None or col is None:
        return None
    try:
        return action, int(col), int(row)
    except (TypeError, ValueError):
        return None


def actions_from_python(code: str) -> list[tuple[int, int | None, int | None]]:
    result = []
    for item in _literal_action_items(code):
        normalized = _normalize_action(item)
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
                )
            )
            grid_consumed = True
            break
    return examples


def load_duck_traces(paths: Iterable[str | Path]) -> list[DuckExample]:
    examples: list[DuckExample] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            files = sorted(path.rglob("*.json"))
        else:
            files = [path]
        for file_path in files:
            try:
                examples.extend(load_duck_trace(file_path))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    return examples


def save_examples_npz(examples: list[DuckExample], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    grids = np.stack([example.grid for example in examples]).astype(np.uint8)
    actions = np.asarray([example.action for example in examples], dtype=np.int16)
    xs = np.asarray([example.x if example.x is not None else -1 for example in examples], dtype=np.int16)
    ys = np.asarray([example.y if example.y is not None else -1 for example in examples], dtype=np.int16)
    games = np.asarray([example.game_id for example in examples])
    np.savez_compressed(path, grids=grids, actions=actions, xs=xs, ys=ys, games=games)
    return path
