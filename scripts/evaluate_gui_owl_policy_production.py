#!/usr/bin/env python3
"""Evaluate GUI-Owl checkpoints under the canonical production system prompt."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from android_drlqn.gui_owl_contract import normalize_gui_messages  # noqa: E402
from scripts import evaluate_gui_owl_policy_checkpoint as base  # noqa: E402

_ORIGINAL_LOAD = base._load_gui_records


def _load(path):
    records = _ORIGINAL_LOAD(path)
    for record in records:
        messages = record.get("messages")
        if messages:
            record["messages"] = normalize_gui_messages(messages)
    return records


base._load_gui_records = _load


if __name__ == "__main__":
    raise SystemExit(base.main())
