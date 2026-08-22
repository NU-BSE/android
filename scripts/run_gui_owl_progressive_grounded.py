#!/usr/bin/env python3
"""Run progressive GUI-Owl compression with the grounding revision enabled."""
from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_gui_owl_progressive_distillation as base  # noqa: E402
from scripts.gui_owl_grounding_revision import text_layer_selection_is_safe  # noqa: E402

REVISION_CONFIG = os.environ.get(
    "GUI_OWL_REVISION_CONFIG", "config/gui_owl_students_grounded.json"
)
CANDIDATE_OUTPUT = os.environ.get("GUI_OWL_DISTILLED_OUTPUT", "").strip()

base.RECOVERY_TRAINER = "scripts/distill_gui_owl_grounded_balanced.py"
base.POLICY_TRAINER = "scripts/distill_gui_owl_policy_grounded_balanced.py"
base.text_layer_selection_is_safe = text_layer_selection_is_safe

_original_load_plan = base.load_student_plan
_original_run = base._run


def _load_plan(path):
    plan = copy.deepcopy(_original_load_plan(path))
    if CANDIDATE_OUTPUT:
        students = plan.get("students")
        if isinstance(students, dict) and isinstance(students.get("1.5b"), dict):
            students["1.5b"]["distilled"] = CANDIDATE_OUTPUT
    return plan


def _run(command, *, dry_run):
    values = list(command)
    for index, value in enumerate(values):
        if value == "scripts/build_gui_owl_student.py":
            values[index] = "scripts/build_gui_owl_student_grounded.py"
    if len(values) >= 2 and values[1] in {
        "scripts/build_gui_owl_student_grounded.py",
        "scripts/distill_gui_owl_grounded_balanced.py",
        "scripts/distill_gui_owl_policy_grounded_balanced.py",
    }:
        if "--config" not in values:
            values += ["--config", REVISION_CONFIG]
    _original_run(values, dry_run=dry_run)


base.load_student_plan = _load_plan
base._run = _run


if __name__ == "__main__":
    raise SystemExit(base.main())
