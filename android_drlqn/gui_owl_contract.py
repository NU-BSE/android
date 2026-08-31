"""Canonical GUI-Owl output contract shared by corpus builders and trainers."""
from __future__ import annotations

from copy import deepcopy
from typing import Mapping, Sequence

# Do not add a tool-call example to this prompt. Both attempts cost more than
# they bought, measured against the 2B teacher with its GUI adapter over one
# fixed 96-record holdout (median coordinate error, and share within 50px):
#
#   no example, actions declared (this prompt)      4.0px   61.7%
#   + "0..1000 scale" line                          5.0px   60.5%
#   examples using coordinate-free actions         13.0px   54.3%
#   examples containing real coordinates          128.0px   44.4%
#
# The last row is not a rounding difference: the teacher copied the example's
# point verbatim on 27 of 88 screens, and at 128px it failed the <=100px
# grounding gate its own students are held to. An earlier revision instead
# showed a fill-in-the-blank skeleton -- {"name":"mobile.<action>",
# "arguments":{...}} -- which the depth-24 bootstrap reproduced literally on 32
# of 136 screens, 64% of all parse failures. A placeholder gets copied and a
# concrete value gets copied; the corpus prompt, which shows neither, grounds
# best.
#
# Declaring the action vocabulary is the one addition that pays: it costs
# ~2px of grounding and the teacher emits six distinct action types against
# four without it, which is the lever on action-name accuracy. It is also what
# makes mobile.type and mobile.long_press reachable at all.
GUI_OWL_SYSTEM_PROMPT = """You are GUI-Owl-1.5 operating Android. Return one concise Action conclusion and one bounded mobile tool call.
Allowed actions: mobile.click, mobile.long_press, mobile.swipe, mobile.type, mobile.system_button, mobile.wait, mobile.answer, mobile.ask_user, mobile.terminate."""


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
