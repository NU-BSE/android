#!/usr/bin/env python3
"""Add compressed prior-action history to a single-step GUI-policy corpus.

Both GUI-Owl-2B and Qwen3.5-0.8B score 0.000 recall on `mobile.system_button`
and `mobile.answer`, and the teacher was fine-tuned on this corpus. Those two
actions are the ones that depend on episode state rather than on the current
screen: "answer" means the goal's question is now satisfied, "Back" means this
screen is a dead end. A single screenshot plus a goal does not say either.

The interaction contract already carries history — `project_gui_owl_train.py`
keeps the latest turns in full and compresses older ones to their visible
`Action:` conclusions. The SFT corpus does not. This restores the compressed
half: prior steps of the same episode, in order, as text. Screenshots are not
duplicated, so a record grows by tens of tokens rather than by thousands.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path
from typing import Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from android_drlqn.gui_owl_metrics import assistant_content_text  # noqa: E402

_ACTION_LINE = re.compile(r"(?:^|\n)Action:\s*(.*?)(?=\n<tool_call>|\Z)", re.S)
_ANCHOR = "Current screen:"
_HEADING = "Steps so far:"


def _conclusion(record: Mapping[str, object]) -> str:
    messages = record.get("messages") or ()
    if not messages:
        return ""
    text = assistant_content_text(messages[-1].get("content"))
    found = _ACTION_LINE.search(text or "")
    return " ".join((found.group(1) if found else "").split())


def _with_history(record: dict[str, object], history: Sequence[str]) -> dict[str, object]:
    if not history:
        return record
    block = _HEADING + "\n" + "\n".join(
        f"{index}. {line}" for index, line in enumerate(history, 1)
    )
    messages = [dict(message) for message in record["messages"]]
    user = next(
        (message for message in reversed(messages) if message.get("role") == "user"), None
    )
    if user is None:
        return record
    content = user.get("content")
    if not isinstance(content, Sequence) or isinstance(content, (str, bytes)):
        return record
    parts = [dict(part) if isinstance(part, Mapping) else part for part in content]
    target = next(
        (part for part in parts if isinstance(part, Mapping) and part.get("type") == "text"),
        None,
    )
    if target is None:
        return record
    text = str(target.get("text") or "")
    if _HEADING in text:
        return record
    if _ANCHOR in text:
        text = text.replace(_ANCHOR, block + "\n" + _ANCHOR, 1)
    else:
        text = text.rstrip() + "\n" + block
    target["text"] = text
    user["content"] = parts
    record = dict(record)
    record["messages"] = messages
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--history-steps",
        type=int,
        default=4,
        help="how many prior conclusions to carry (0 disables, writing a copy)",
    )
    parser.add_argument(
        "--only-ids",
        type=Path,
        help="JSON list, or an object with holdout_ids, restricting the output",
    )
    args = parser.parse_args(argv)
    if args.history_steps < 0:
        parser.error("--history-steps must not be negative")

    records = [
        json.loads(line)
        for line in args.corpus.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    keep: set[str] | None = None
    if args.only_ids:
        value = json.loads(args.only_ids.read_text(encoding="utf-8"))
        raw = value.get("holdout_ids") if isinstance(value, Mapping) else value
        keep = {str(item) for item in raw}

    episodes: dict[str, list[dict[str, object]]] = collections.defaultdict(list)
    for record in records:
        episodes[str(record["episode_id"])].append(record)
    for steps in episodes.values():
        steps.sort(key=lambda item: int(item.get("step_index") or 0))

    written = with_history = 0
    lengths: collections.Counter[int] = collections.Counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            if keep is not None and str(record["id"]) not in keep:
                continue
            history: list[str] = []
            if args.history_steps:
                index = int(record.get("step_index") or 0)
                prior = [
                    item
                    for item in episodes[str(record["episode_id"])]
                    if int(item.get("step_index") or 0) < index
                ]
                history = [
                    line
                    for line in (_conclusion(item) for item in prior[-args.history_steps :])
                    if line
                ]
            updated = _with_history(dict(record), history)
            lengths[len(history)] += 1
            written += 1
            with_history += bool(history)
            handle.write(json.dumps(updated, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "corpus": str(args.corpus),
                "output": str(args.output),
                "records": written,
                "records_with_history": with_history,
                "history_steps": args.history_steps,
                "history_length_distribution": dict(sorted(lengths.items())),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
