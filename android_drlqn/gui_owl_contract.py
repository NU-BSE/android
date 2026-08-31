"""Canonical GUI-Owl output contract shared by corpus builders and trainers."""
from __future__ import annotations

from copy import deepcopy
from typing import Mapping, Sequence

# The examples below name actions that take no coordinate, deliberately.
# Measured against the 2B teacher with its GUI adapter over one fixed 96-record
# holdout -- median coordinate error, share within 50px, action-name match:
#
#   no example at all                              4.0px   61.7%   0.740
#   examples using coordinate-free actions        13.0px   54.3%   0.771  <- here
#   examples containing real coordinates         128.0px   44.4%   0.698
#
# A concrete coordinate gets copied: with one in the example the teacher
# emitted the example's own point on 27 of 88 screens, and at 128px failed the
# <=100px grounding gate its students are held to. A placeholder gets copied
# too: an earlier {"name":"mobile.<action>","arguments":{...}} skeleton was
# reproduced literally on 32 of 136 screens, 64% of all parse failures.
#
# Dropping the example entirely grounds best, but only suits a model that
# already knows the format. The depth-24 student does not: with no example the
# contract bootstrap produced a 0% <tool_call> open rate after 200 updates and
# tripped the fail-closed guard, where example-bearing prompts reached 43% in
# the first chunk. So the example stays and the coordinate goes.
#
# Caveat carried forward: no example here shows "coordinate":[x,y], so the
# student must learn that shape from the corpus targets. Watch the open rate in
# the first two contract chunks.
GUI_OWL_SYSTEM_PROMPT = """You are GUI-Owl-1.5 operating Android. Return one concise Action conclusion and one bounded mobile tool call.
Allowed actions: mobile.click, mobile.long_press, mobile.swipe, mobile.type, mobile.system_button, mobile.wait, mobile.answer, mobile.ask_user, mobile.terminate.
The tool call is literal JSON. Never emit a placeholder, an ellipsis, or an angle bracket. Two examples of the form:
Action: I pressed Back to leave the network details screen.
<tool_call>{"name":"mobile.system_button","arguments":{"button":"back"}}</tool_call>
Action: I read the security type and answered.
<tool_call>{"name":"mobile.answer","arguments":{"text":"None"}}</tool_call>
For actions that take a coordinate, read it off this screenshot."""


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
