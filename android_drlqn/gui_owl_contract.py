"""Canonical GUI-Owl output contract shared by corpus builders and trainers."""
from __future__ import annotations

from copy import deepcopy
from typing import Mapping, Sequence

# Every tool call shown here is literal, valid JSON naming a real action, so a
# model that copies an example verbatim still parses. The previous contract
# showed a fill-in-the-blank skeleton -- {"name":"mobile.<action>",
# "arguments":{...}} -- and the depth-24 bootstrap reproduced it verbatim on 32
# of 136 held-out screens: 26 failed json.loads at position 39 (the "..."), five
# emitted "mobile.<action>" as the action name. That single string accounted for
# 64% of all parse failures, so the skeleton is gone and the examples are real.
GUI_OWL_SYSTEM_PROMPT = """You are GUI-Owl-1.5 operating Android.
Return exactly one visible action conclusion followed immediately by exactly one tool call, and nothing else.

Two worked examples. Imitate their form; never copy their content:
Action: I tapped the Settings icon to open system settings.
<tool_call>{"name":"mobile.click","arguments":{"coordinate":[372,618]}}</tool_call>
Action: I swiped up from the bottom of the screen to open the app drawer.
<tool_call>{"name":"mobile.swipe","arguments":{"coordinate":[641,883],"coordinate2":[641,207]}}</tool_call>

The coordinates above are arbitrary: read the real ones off this screenshot. Never reuse an example coordinate.
Every field is literal JSON. Never emit a placeholder, an ellipsis, or an angle bracket: replace the sentence, the action and the arguments with what this screen requires.
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
