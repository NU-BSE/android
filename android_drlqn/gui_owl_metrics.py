"""Free-running GUI-Owl correctness metrics shared by trainers and diagnostics."""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Mapping, Sequence


_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
_COORDINATE_SCALE = 1000.0
_MISSING_COORDINATE_ERROR = math.sqrt(2.0) * _COORDINATE_SCALE

_ACTION_ALIASES = {
    "tap": "click",
    "scroll": "swipe",
    "key": "system_button",
    "terminate": "answer",
    "ask_confirmation": "interact",
}


@dataclass(frozen=True)
class ParsedToolCall:
    name: str
    arguments: Mapping[str, object]
    coordinate: tuple[float, float] | None


@dataclass(frozen=True)
class GUICorrectness:
    target_action: str | None
    generated_action: str | None
    action_match: bool
    target_coordinate: tuple[float, float] | None
    generated_coordinate: tuple[float, float] | None
    coordinate_error: float | None


def assistant_content_text(content: object) -> str:
    """Extract textual assistant content from Qwen chat-template content shapes."""

    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, Sequence) or isinstance(content, (str, bytes)):
        return ""
    parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            if item.strip():
                parts.append(item.strip())
            continue
        if not isinstance(item, Mapping):
            continue
        if str(item.get("type") or "") == "text" or "text" in item:
            text = str(item.get("text") or "").strip()
            if text:
                parts.append(text)
    return "\n".join(parts).strip()


def _canonical_action_name(value: object) -> str:
    name = str(value or "").strip().casefold().rsplit(".", 1)[-1]
    return _ACTION_ALIASES.get(name, name)


def _coordinate(arguments: Mapping[str, object]) -> tuple[float, float] | None:
    raw = arguments.get("coordinate") or arguments.get("point")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) != 2:
        return None
    try:
        x = max(0.0, min(_COORDINATE_SCALE, float(raw[0])))
        y = max(0.0, min(_COORDINATE_SCALE, float(raw[1])))
    except (TypeError, ValueError):
        return None
    return (x, y)


def parse_tool_call(text: str) -> ParsedToolCall | None:
    """Parse the first GUI-Owl tool call without collapsing action classes."""

    match = _TOOL_CALL_RE.search(text)
    if not match:
        return None
    try:
        value = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    if not isinstance(value, Mapping):
        return None
    name = _canonical_action_name(value.get("name"))
    if not name:
        return None
    arguments = value.get("arguments") or {}
    if not isinstance(arguments, Mapping):
        arguments = {}
    return ParsedToolCall(name=name, arguments=dict(arguments), coordinate=_coordinate(arguments))


def compare_gui_outputs(generated: str, target: str) -> GUICorrectness:
    """Compare action identity and point grounding against the supervised target.

    Coordinate-bearing targets are never allowed to disappear from the metric:
    a missing generated point receives the maximum 0..1000-space diagonal error.
    This prevents a malformed/wrong-action output from making the median look
    artificially good by being dropped from the coordinate denominator.
    """

    generated_call = parse_tool_call(generated)
    target_call = parse_tool_call(target)
    target_action = target_call.name if target_call else None
    generated_action = generated_call.name if generated_call else None
    action_match = bool(target_action and generated_action == target_action)
    target_coordinate = target_call.coordinate if target_call else None
    generated_coordinate = generated_call.coordinate if generated_call else None

    coordinate_error: float | None = None
    if target_coordinate is not None:
        if generated_coordinate is None:
            coordinate_error = _MISSING_COORDINATE_ERROR
        else:
            coordinate_error = math.dist(target_coordinate, generated_coordinate)

    return GUICorrectness(
        target_action=target_action,
        generated_action=generated_action,
        action_match=action_match,
        target_coordinate=target_coordinate,
        generated_coordinate=generated_coordinate,
        coordinate_error=coordinate_error,
    )


__all__ = [
    "GUICorrectness",
    "ParsedToolCall",
    "assistant_content_text",
    "compare_gui_outputs",
    "parse_tool_call",
]
