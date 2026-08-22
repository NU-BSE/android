#!/usr/bin/env python3
"""Run the base GUI-Owl trainer with action/coordinate correctness evaluation."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import distill_gui_owl_student as base  # noqa: E402
from scripts.gui_owl_grounded_evaluation import evaluate_generation  # noqa: E402


def _grounded_evaluate_generation(**kwargs):
    return evaluate_generation(base, **kwargs)


base._evaluate_generation = _grounded_evaluate_generation


if __name__ == "__main__":
    raise SystemExit(base.main())
