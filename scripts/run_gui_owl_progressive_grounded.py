#!/usr/bin/env python3
"""Run progressive GUI-Owl compression with the grounding revision enabled."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _argv_value(name: str, default: str) -> str:
    try:
        index = sys.argv.index(name)
    except ValueError:
        return default
    if index + 1 >= len(sys.argv):
        return default
    return sys.argv[index + 1]


def _revision_plan(source: Path, candidate_output: str) -> dict[str, object]:
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"GUI-Owl plan must be an object: {source}")
    value["grounding_revision"] = {
        "name": "coverage24-prefix3-action-balance-v1",
        "text_coverage_max_depth": 24,
        "visual_integration_prefix_layers": 3,
        "gui_action_balance_power": 0.5,
        "gui_action_balance_max_factor": 3.0,
        "policy_first_token_weight": 2.0,
    }
    stages = value.get("stages")
    if not isinstance(stages, dict):
        raise ValueError("GUI-Owl plan lacks stages")
    overrides = {
        "recovery": {
            "supervised_weight": 0.35,
            "forward_kl_weight": 1.0,
            "hidden_state_weight": 0.30,
            "learning_rate": 0.000002,
            "eval_records": 64,
            "generation_gate": {
                "first_token_eos_rate_max": 0.05,
                "empty_generation_rate_max": 0.05,
                "median_generated_tokens_min": 8,
                "gui_parse_rate_min": 0.70,
                "gui_action_name_match_rate_min": 0.70,
                "gui_median_coordinate_error_max": 100.0,
            },
        },
        "on_policy": {
            "supervised_weight": 0.50,
            "forward_kl_weight": 0.50,
            "k3_weight": 0.50,
            "learning_rate": 0.0000015,
            "eval_records": 64,
            "generation_gate": {
                "first_token_eos_rate_max": 0.05,
                "empty_generation_rate_max": 0.05,
                "median_generated_tokens_min": 8,
                "gui_parse_rate_min": 0.70,
                "gui_action_name_match_rate_min": 0.70,
                "gui_median_coordinate_error_max": 100.0,
            },
        },
        "specialize": {
            "supervised_weight": 0.60,
            "forward_kl_weight": 0.60,
            "hidden_state_weight": 0.10,
            "k3_weight": 0.10,
            "learning_rate": 0.0000008,
            "eval_records": 64,
            "generation_gate": {
                "first_token_eos_rate_max": 0.05,
                "empty_generation_rate_max": 0.05,
                "median_generated_tokens_min": 8,
                "gui_parse_rate_min": 0.70,
                "gui_action_name_match_rate_min": 0.70,
                "gui_median_coordinate_error_max": 100.0,
            },
        },
    }
    for stage, patch in overrides.items():
        current = stages.get(stage)
        if not isinstance(current, dict):
            raise ValueError(f"GUI-Owl plan lacks stage: {stage}")
        current.update(patch)

    if candidate_output:
        students = value.get("students")
        if not isinstance(students, dict) or not isinstance(students.get("1.5b"), dict):
            raise ValueError("GUI-Owl plan lacks 1.5b student")
        students["1.5b"]["distilled"] = candidate_output
    return value


SOURCE_CONFIG = Path(_argv_value("--config", "config/gui_owl_students.json"))
CANDIDATE_OUTPUT = os.environ.get("GUI_OWL_DISTILLED_OUTPUT", "").strip()
REVISION_CONFIG = Path(
    os.environ.get(
        "GUI_OWL_REVISION_CONFIG",
        f"artifacts/training/gui-owl-grounding-revision-{os.getpid()}.json",
    )
)
REVISION_CONFIG.parent.mkdir(parents=True, exist_ok=True)
REVISION_CONFIG.write_text(
    json.dumps(_revision_plan(SOURCE_CONFIG, CANDIDATE_OUTPUT), indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
os.environ.setdefault("GUI_OWL_FIRST_TOKEN_WEIGHT", "2.0")

# Make the base runner itself load the same plan passed to every subprocess.
if "--config" in sys.argv:
    config_index = sys.argv.index("--config")
    if config_index + 1 >= len(sys.argv):
        raise SystemExit("--config requires a path")
    sys.argv[config_index + 1] = str(REVISION_CONFIG)
else:
    sys.argv += ["--config", str(REVISION_CONFIG)]

from scripts import run_gui_owl_progressive_distillation as base  # noqa: E402
from scripts.gui_owl_grounding_revision import text_layer_selection_is_safe  # noqa: E402

base.RECOVERY_TRAINER = "scripts/distill_gui_owl_grounded_balanced.py"
base.POLICY_TRAINER = "scripts/distill_gui_owl_policy_grounded_balanced.py"
base.text_layer_selection_is_safe = text_layer_selection_is_safe

_original_run = base._run


def _run(command, *, dry_run):
    values = list(command)
    for index, value in enumerate(values):
        if value == "scripts/build_gui_owl_student.py":
            values[index] = "scripts/build_gui_owl_student_grounded.py"
    if len(values) >= 2 and values[1] in {
        "scripts/build_gui_owl_student_grounded.py",
        "scripts/distill_gui_owl_grounded_balanced.py",
        "scripts/distill_gui_owl_policy_grounded_balanced.py",
    } and "--config" not in values:
        values += ["--config", str(REVISION_CONFIG)]
    _original_run(values, dry_run=dry_run)


base._run = _run


if __name__ == "__main__":
    raise SystemExit(base.main())
