"""Canonical GUI-Owl output contract shared by corpus builders and trainers."""
from __future__ import annotations

from copy import deepcopy
from typing import Mapping, Sequence

# Do not "improve" this prompt without running both measurements below. Six
# variants were tried; four of them cost a production run each.
#
# Teacher control -- 2B teacher with its GUI adapter, one fixed 96-record
# holdout, the same metric the students are gated on:
#
#   prompt                              med err   <50px   action
#   corpus prompt, no example             2.2px   63.0%    0.792
#   + allowed actions                     4.0px   61.7%    0.740
#   + coordinate-free examples           13.0px   54.3%    0.771
#   + one click coordinate              100.0px   44.4%    0.604
#   this prompt (click + swipe)         128.0px   44.4%    0.698
#   + three varied coordinates          234.1px   37.0%    0.510
#
# Student contract bootstrap, <tool_call> open rate:
#
#   no example                     0% after 200 updates, tripped fail-closed
#   coordinate-free examples       peaked 76.2% at chunk 2, then declined
#   this prompt                    95.3% at chunk 6 -- the only one to pass
#
# The two axes disagree and the student's axis wins, because a prompt the
# student cannot learn the format from is worth nothing however well the
# teacher grounds under it. Teacher ablation mispredicted the student twice.
#
# Verbatim copying is NOT the mechanism: the one-click variant copied its own
# example point 0 times out of 86 and still landed at 100px, reverting to the
# corpus's own (500,800) attractor 28 times. Any coordinate in the context
# degrades grounding whether or not that value is reused, so there is no
# coordinate-bearing prompt below ~100px to find.
#
# The ~128px is therefore treated as a prompt-induced offset that applies
# equally to teacher and student. It matters at the depth-24 recovery gate
# (<=100px), not at the contract bootstrap, which gates only open/close/parse
# -- so grounding is a recovery-stage problem, and that gate should be
# calibrated against the teacher measured under THIS prompt, not against the
# 2.2px the teacher reaches under a prompt nothing ships.
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
