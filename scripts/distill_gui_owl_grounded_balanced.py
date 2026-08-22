#!/usr/bin/env python3
"""Grounded structural recovery with damped GUI action balancing."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import distill_gui_owl_student as base  # noqa: E402
from scripts.gui_owl_grounded_evaluation import evaluate_generation  # noqa: E402
from scripts.gui_owl_grounding_revision import balanced_corpus_records  # noqa: E402

_ORIGINAL_CORPUS_RECORDS = base._corpus_records


def _grounded_evaluate(**kwargs):
    return evaluate_generation(base, **kwargs)


def _balanced_records(sources, max_records):
    return balanced_corpus_records(_ORIGINAL_CORPUS_RECORDS, sources, max_records)


base._evaluate_generation = _grounded_evaluate
base._corpus_records = _balanced_records


if __name__ == "__main__":
    raise SystemExit(base.main())
