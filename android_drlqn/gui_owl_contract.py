"""Canonical GUI-Owl output contract shared by corpus builders and trainers."""
from __future__ import annotations

from copy import deepcopy
from typing import Mapping, Sequence

GUI_OWL_SYSTEM_PROMPT = """You are GUI-Owl-1.5 operating Android.
Return exactly one concise visible action conclusion followed immediately by exactly one tool call:
Action: <what to do next>
<tool_call>{\"name\":\"mobile.<action>\",\"arguments\":{...}}</tool_call>
Allowed actions: mobile.click, mobile.long_press, mobile.swipe, mobile.type, mobile.system_button, mobile.wait, mobile.answer, mobile.ask_user, mobile.terminate.
Use integer screen coordinates on a 0..1000 scale. Do not stop after the Action line. Do not emit user, assistant, robot, or other role markers inside the answer."""


def normalize_gui_messages(messages: Sequence[object]) -> list[object]:
    """Return a copy of one GUI-policy conversation with the canonical contract.

    Existing training artifacts predate the explicit grammar above.  Rewriting
    only the system message lets old JSONL corpora train/evaluate under the same
    output contract without mutating user observations or assistant targets.
    """

    values = deepcopy(list(messages))
    if not values:
        return values
    first = values[0]
    if isinstance(first, Mapping) and str(first.get("role") or "") == "system":
        replacement = dict(first)
        replacement["content"] = GUI_OWL_SYSTEM_PROMPT
        values[0] = replacement
    else:
        values.insert(0, {"role": "system", "content": GUI_OWL_SYSTEM_PROMPT})
    return values


__all__ = ["GUI_OWL_SYSTEM_PROMPT", "normalize_gui_messages"]
