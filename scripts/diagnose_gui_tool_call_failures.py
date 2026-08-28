#!/usr/bin/env python3
"""Explain why well-formed GUI tool calls still fail to parse.

The contract bootstrap gates on open/close marker rates and parse rate. When
close sits at 86-97% while parse plateaus near 62%, the markers are already
landing in the right place and something inside them is wrong. This buckets
those rows by the parser's own decision points.

`parse_gui_owl_output` is the oracle: it decides pass/fail, and the staged
checks here only explain the failure. A row the oracle rejects that no bucket
explains is reported as `unclassified`, so a taxonomy gap is visible rather
than silently mislabelled.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project_gui_owl_train import _TOOL_CALL_RE, parse_gui_owl_output  # noqa: E402

# Mirrors the name branches in parse_gui_owl_output; anything else returns None.
KNOWN_ACTIONS = frozenset(
    {
        "click", "tap",
        "swipe", "scroll",
        "system_button", "key",
        "wait",
        "interact", "ask_confirmation",
        "answer", "terminate",
    }
)


def classify(text: str) -> tuple[str, str]:
    """Return (bucket, detail) for a text the parser rejected."""
    if "<tool_call>" not in text:
        return "no_open_marker", ""
    if "</tool_call>" not in text:
        return "no_close_marker", ""
    match = _TOOL_CALL_RE.search(text)
    if not match:
        # The parser requires <tool_call> {...} </tool_call> with only
        # whitespace around the braces.
        return "payload_not_brace_delimited", text[:160]
    payload = match.group(1)
    try:
        call = json.loads(payload)
    except json.JSONDecodeError as exc:
        return "malformed_json", f"{exc.msg} at pos {exc.pos}"
    if not isinstance(call, Mapping):
        return "payload_not_object", type(call).__name__
    raw_name = str(call.get("name") or "")
    name = raw_name.casefold().rsplit(".", 1)[-1]
    if name not in KNOWN_ACTIONS:
        return "unsupported_action_name", raw_name or "<empty>"
    return "unclassified", payload[:160]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("generations", help="jsonl written by evaluate_gui_owl_policy_checkpoint.py")
    parser.add_argument("--show", type=int, default=3, help="examples per bucket")
    args = parser.parse_args(argv)

    rows = [json.loads(line) for line in Path(args.generations).read_text().splitlines() if line.strip()]
    buckets: dict[str, list[dict[str, object]]] = collections.defaultdict(list)
    marked_but_unparsed = 0
    parsed = 0

    for row in rows:
        text = row.get("generation") or ""
        if parse_gui_owl_output(text) is not None:
            parsed += 1
            continue
        bucket, detail = classify(text)
        if row.get("has_tool_call_open") and row.get("has_tool_call_close"):
            marked_but_unparsed += 1
        buckets[bucket].append({"index": row.get("index"), "detail": detail, "generation": text})

    total = len(rows)
    print(f"records {total}   parsed {parsed} ({parsed/total:.1%})   "
          f"markers present but unparsed {marked_but_unparsed}")
    print()
    for bucket, items in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
        print(f"{bucket:32} {len(items):>3}  ({len(items)/total:.1%})")
        details = collections.Counter(str(item["detail"])[:70] for item in items)
        for detail, count in details.most_common(args.show):
            if detail:
                print(f"    {count:>3}x {detail}")

    # A target the parser itself rejects can never be scored as correct, so a
    # student that emits it faithfully is punished for being right.
    unwinnable = collections.Counter()
    for row in rows:
        target = row.get("target")
        if isinstance(target, str) and parse_gui_owl_output(target) is None:
            bucket, detail = classify(target)
            unwinnable[f"{bucket}:{detail}" if detail else bucket] += 1
    if unwinnable:
        count = sum(unwinnable.values())
        print()
        print(f"!! targets the parser itself rejects: {count}/{total} ({count/total:.1%}) "
              f"-- these cap the achievable parse rate")
        for key, value in unwinnable.most_common(10):
            print(f"    {value:>3}x {key}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
