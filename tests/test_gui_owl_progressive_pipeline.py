from __future__ import annotations

import json
from pathlib import Path

from android_drlqn.gui_owl_student import load_student_plan
from scripts.run_gui_owl_progressive_distillation import _accepted


def _plan() -> dict[str, object]:
    return load_student_plan("config/gui_owl_students.json")


def test_plan_replaces_one_epoch_distillation_with_update_stages():
    plan = _plan()
    assert plan["schema_version"] == 2
    stages = plan["stages"]
    assert set(stages) == {"recovery", "on_policy", "specialize"}
    for stage in stages.values():
        assert "epochs" not in stage
        assert int(stage["max_updates"]) > 0
        assert int(stage["gradient_accumulation"]) > 0
        gate = stage["generation_gate"]
        assert float(gate["first_token_eos_rate_max"]) <= 0.05
        assert float(gate["empty_generation_rate_max"]) <= 0.05
        assert float(gate["median_generated_tokens_min"]) >= 8


def test_recovery_requires_broad_data_and_on_policy_uses_k3():
    plan = _plan()
    recovery = plan["recovery_corpora"]["broad_instruction_multimodal"]
    assert recovery["required"] is True
    assert recovery["teacher_mode"] == "base"
    assert recovery["path"].endswith("gui-owl-recovery.jsonl")
    assert float(plan["stages"]["recovery"]["k3_weight"]) == 0.0
    assert float(plan["stages"]["on_policy"]["k3_weight"]) > 0.0
    assert int(plan["stages"]["on_policy"]["on_policy_min_new_tokens"]) >= 1


def test_progressive_sources_avoid_one_shot_depth_and_bad_half_b_cascade():
    plan = _plan()
    progressive = plan["progressive"]
    assert progressive["1.5b"]["text_depth_schedule"] == [24, 20, 16]
    assert progressive["1b"]["text_depth_schedule"] == [12, 9, 6]
    assert progressive["1b"]["source"].endswith("gui-owl-1.5b-distilled")
    assert progressive["0.5b"]["source"].endswith("gui-owl-1.5b-distilled")
    assert progressive["0.5b"]["text_depth_schedule"] == []
    assert progressive["0.5b"]["structured_width_pruning"] is True


def test_new_outputs_are_versioned_away_from_collapsed_checkpoints():
    plan = _plan()
    for entry in plan["students"].values():
        assert "/v2/" in entry["initialized"]
        assert "/v2/" in entry["distilled"]


def test_resume_only_accepts_generation_gated_checkpoint(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    manifest = checkpoint / "distillation-manifest.json"
    manifest.write_text(json.dumps({"accepted": False}), encoding="utf-8")
    assert _accepted(checkpoint) is False
    manifest.write_text(json.dumps({"accepted": True}), encoding="utf-8")
    assert _accepted(checkpoint) is True
