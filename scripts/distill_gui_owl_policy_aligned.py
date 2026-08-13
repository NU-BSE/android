#!/usr/bin/env python3
"""Run GUI-Owl distillation with a GUI-stratified free-generation holdout.

The base trainer evaluates the first N records of a mixed holdout. During the
on-policy stage that holdout also contains the large broad-recovery corpus, so a
small GUI corpus can be absent or represented by only one or two examples. A
GUI parse-rate gate is meaningless in that case. This entrypoint keeps the base
trainer/losses unchanged but stratifies each generation evaluation so GUI policy
records receive a material share of the evaluation budget.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import distill_gui_owl_student as base  # noqa: E402

_BASE_EVALUATE = base._evaluate_generation
MIN_GUI_EVAL_RECORDS = 8


def _stratified_evaluate_generation(**kwargs):
    records: Sequence[base.TrainingRecord] = kwargs["records"]
    max_records = int(kwargs["max_records"])
    if max_records < 1:
        return _BASE_EVALUATE(**kwargs)

    gui = [record for record in records if record.mode == "gui_policy"]
    other = [record for record in records if record.mode != "gui_policy"]
    if not gui:
        return _BASE_EVALUATE(**kwargs)

    # Reserve at least eight GUI examples where available and otherwise use up
    # to half the evaluation budget. This keeps EOS/CE/KL metrics mixed while
    # making the GUI parse-rate gate statistically meaningful.
    gui_quota = min(
        len(gui),
        max_records,
        max(MIN_GUI_EVAL_RECORDS, max_records // 2),
    )
    other_quota = min(len(other), max_records - gui_quota)
    selected = gui[:gui_quota] + other[:other_quota]
    if len(selected) < max_records:
        used_gui = gui_quota
        used_other = other_quota
        selected.extend(gui[used_gui : used_gui + max_records - len(selected)])
        if len(selected) < max_records:
            selected.extend(other[used_other : used_other + max_records - len(selected)])

    forwarded = dict(kwargs)
    forwarded["records"] = selected
    forwarded["max_records"] = len(selected)
    return _BASE_EVALUATE(**forwarded)


base._evaluate_generation = _stratified_evaluate_generation


if __name__ == "__main__":
    raise SystemExit(base.main())
