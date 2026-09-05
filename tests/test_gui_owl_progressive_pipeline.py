from __future__ import annotations

import argparse
import json
from pathlib import Path

from android_drlqn.gui_owl_student import load_student_plan
from scripts.normalize_gui_owl_calibration import (
    SOURCE_DEEPSTACK_VISUAL_INDEXES,
    TARGET_DEEPSTACK_VISUAL_INDEXES,
    deepstack_aligned_vision_selection,
    grouped_query_kv_selection,
)
from scripts.prepare_gui_owl_recovery import DEFAULT_REVISION, _clean_messages
from scripts.run_gui_owl_progressive_distillation import (
    POLICY_TRAINER,
    RECOVERY_TRAINER,
    _accepted,
    _tail_protected_lineage,
)


def _plan() -> dict[str, object]:
    return load_student_plan("config/gui_owl_students.json")


def test_all_training_stages_run_through_correctness_aware_entrypoints():
    assert POLICY_TRAINER == "scripts/distill_gui_owl_policy_aligned.py"
    assert RECOVERY_TRAINER == "scripts/distill_gui_owl_grounded.py"

    source = Path("scripts/run_gui_owl_progressive_distillation.py").read_text(encoding="utf-8")
    for stage in ('"on_policy"', '"specialize"'):
        head = source.split(stage)[0]
        assert head.rstrip().endswith('"--stage",'), stage
        invocation = head.rsplit("POLICY_TRAINER", 1)
        assert len(invocation) == 2, f"{stage} must be launched with POLICY_TRAINER"
        assert "RECOVERY_TRAINER" not in invocation[1], stage
    assert source.count('"scripts/distill_gui_owl_grounded.py"') == 1
    assert source.count('"scripts/distill_gui_owl_policy_aligned.py"') == 1


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
        # Calibrated at 85% retention of the 2B teacher measured under the
        # shipping contract prompt: action 0.762, within-50px 0.452, median
        # 128.3px at n=256. The previous 0.70 / 100px came from a teacher
        # measured under a prompt nothing ships (2.2px), and under the shipping
        # one the teacher failed its own students' coordinate gate outright.
        assert 0.60 <= float(gate["gui_action_name_match_rate_min"]) <= 0.70
        # The within-50px rate is the binding coordinate gate; the median is a
        # loose backstop because it bootstraps to a 130px-wide CI on a bimodal
        # error distribution.
        assert 0 < float(gate["gui_coordinate_within_50_rate_min"]) <= 0.50
        assert float(gate["gui_median_coordinate_error_max"]) <= 200.0


def test_recovery_requires_broad_data_and_on_policy_uses_k3():
    plan = _plan()
    recovery = plan["recovery_corpora"]["broad_instruction_multimodal"]
    assert recovery["required"] is True
    assert recovery["teacher_mode"] == "base"
    assert recovery["path"].endswith("gui-owl-recovery.jsonl")
    assert float(plan["stages"]["recovery"]["k3_weight"]) == 0.0
    assert float(plan["stages"]["on_policy"]["k3_weight"]) > 0.0
    assert int(plan["stages"]["on_policy"]["on_policy_min_new_tokens"]) == 0


def test_recovery_preparer_is_pinned_and_filters_incomplete_dialogues():
    assert len(DEFAULT_REVISION) == 40
    good = _clean_messages(
        [
            {"role": "user", "content": "Explain gravity simply."},
            {"role": "assistant", "content": "Gravity attracts masses."},
        ],
        max_turn_chars=100,
        max_total_chars=200,
    )
    assert good is not None
    assert good[-1]["role"] == "assistant"
    incomplete = _clean_messages(
        [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Answer"},
            {"role": "user", "content": "Follow-up"},
        ],
        max_turn_chars=100,
        max_total_chars=200,
    )
    assert incomplete is None


def test_progressive_sources_stop_1_5b_at_depth_20():
    plan = _plan()
    progressive = plan["progressive"]
    assert progressive["1.5b"]["text_depth_schedule"] == [24, 20]
    assert plan["students"]["1.5b"]["text_layers"] == 20
    assert progressive["1b"]["text_depth_schedule"] == [12, 9, 6]
    assert progressive["1b"]["source"].endswith("gui-owl-1.5b-distilled")
    assert progressive["0.5b"]["source"].endswith("gui-owl-1.5b-distilled")
    assert progressive["0.5b"]["text_depth_schedule"] == []
    assert progressive["0.5b"]["structured_width_pruning"] is True
    assert progressive["0.5b"]["source_deepstack_visual_indexes"] == [5, 11, 17]
    assert progressive["0.5b"]["target_deepstack_visual_indexes"] == [2, 5, 8]
    assert plan["students"]["0.5b"]["deepstack_visual_indexes"] == [2, 5, 8]


def test_grouped_query_pruning_keeps_query_heads_with_their_kv_head():
    q_scores = [
        100.0, 0.0, 1.0, 1.0, 90.0, 0.0, 1.0, 1.0,
        80.0, 0.0, 1.0, 1.0, 70.0, 0.0, 1.0, 1.0,
    ]
    kv_scores = [0.0, 1000.0, 0.0, 900.0, 0.0, 800.0, 0.0, 700.0]
    q_heads, kv_heads = grouped_query_kv_selection(
        q_scores,
        kv_scores,
        target_q_heads=8,
        target_kv_heads=4,
    )
    assert kv_heads == (1, 3, 5, 7)
    assert q_heads == (2, 3, 6, 7, 10, 11, 14, 15)


def _tiny_calibration(layers: int = 2) -> dict[str, object]:
    return {
        "schema_version": 2,
        "text": {
            "layers": layers,
            "hidden_size": 8,
            "attention_heads": 16,
            "key_value_heads": 8,
            "head_dim": 2,
            "layer_scores": [1.0] * layers,
            "hidden_scores": [float(index) for index in range(8)],
            "mlp_scores": {str(i): [float(j) for j in range(12)] for i in range(layers)},
            "q_head_scores": {str(i): [float(j) for j in range(16)] for i in range(layers)},
            "kv_head_scores": {str(i): [float(j) for j in range(8)] for i in range(layers)},
        },
        "vision": {"ready": False, "hidden_size": 1024, "attention_heads": 16},
        "recommended": {"text_layers_by_depth": {}},
    }


def test_normalize_derives_width_selections_for_every_plan_student(tmp_path: Path):
    from scripts.normalize_gui_owl_calibration import normalize_manifest

    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(_tiny_calibration()), encoding="utf-8")
    plan = {
        "students": {
            "1.5b": {
                "text_layers": 2,
                "text_hidden_size": 8,
                "text_intermediate_size": 6,
                "text_attention_heads": 12,
                "text_key_value_heads": 6,
                "text_head_dim": 2,
            }
        }
    }
    manifest = normalize_manifest(path, plan=plan)
    recommended = manifest["recommended"]
    # Legacy 0.5B GQA normalization still runs...
    assert recommended["text_kv_heads_4"]["0"] == [4, 5, 6, 7]
    # ...and the plan's own widths are derived from the same raw scores.
    assert recommended["text_mlp_indices_6"]["1"] == [6, 7, 8, 9, 10, 11]
    assert recommended["text_kv_heads_6"]["0"] == [2, 3, 4, 5, 6, 7]
    assert recommended["text_q_heads_12"]["0"] == list(range(4, 16))
    assert "text_hidden_indices_8" not in recommended
    assert manifest["text_mlp_score_source"] == {"1.5b": "mlp_scores"}
    assert json.loads(path.read_text(encoding="utf-8"))["recommended"] == recommended


def test_runner_width_path_builds_at_full_depth_then_recovers(tmp_path: Path, capsys):
    from scripts.run_gui_owl_progressive_distillation import run

    corpus = tmp_path / "recovery.jsonl"
    corpus.write_text('{"messages": []}\n', encoding="utf-8")
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps(_tiny_calibration()), encoding="utf-8")
    plan = _plan()
    plan["recovery_corpora"] = {
        "broad": {"path": str(corpus), "required": True, "min_records": 1}
    }
    plan["progressive"]["1.5b"] = {
        "source": "mPLUG/GUI-Owl-1.5-2B-Instruct",
        "text_depth_schedule": [],
        "structured_width_pruning": True,
    }
    plan["students"]["1.5b"]["initialized"] = str(tmp_path / "init")
    config = tmp_path / "plan.json"
    config.write_text(json.dumps(plan), encoding="utf-8")

    manifest = run(
        argparse.Namespace(
            student="1.5b",
            config=str(config),
            teacher_model=None,
            calibration_manifest=str(calibration),
            device="cpu",
            work_root=str(tmp_path / "work"),
            intermediate_recovery_updates=1,
            max_updates=None,
            max_records_per_corpus=None,
            resume=False,
            skip_on_policy=True,
            skip_specialize=True,
            dry_run=True,
        )
    )
    commands = [line for line in capsys.readouterr().out.splitlines() if line.startswith("+ ")]
    assert "build_gui_owl_student.py" in commands[0]
    assert "--target-text-layers" not in commands[0]
    assert RECOVERY_TRAINER in commands[1]
    assert manifest["stages"][0] == {
        "width_pruned": True,
        "checkpoint": str(tmp_path / "work" / "width-recovered"),
    }


def test_compact_vision_keeps_source_deepstack_blocks_at_matching_target_taps():
    scores = [float((index * 17) % 23) for index in range(24)]
    selected = deepstack_aligned_vision_selection(scores)
    assert len(selected) == 12
    assert tuple(sorted(selected)) == selected
    for source, target in zip(
        SOURCE_DEEPSTACK_VISUAL_INDEXES,
        TARGET_DEEPSTACK_VISUAL_INDEXES,
    ):
        assert selected[target] == source
    assert any(index > 17 for index in selected)


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


def test_final_recovery_resume_requires_gui_evidence(tmp_path: Path):
    checkpoint = tmp_path / "depth-20-recovered"
    checkpoint.mkdir()
    manifest = checkpoint / "distillation-manifest.json"
    legacy = {
        "accepted": True,
        "stage": "recovery",
        "final_evaluation": {"gui_records": 0.0},
    }
    manifest.write_text(json.dumps(legacy), encoding="utf-8")
    assert _accepted(checkpoint) is True
    assert _accepted(checkpoint, require_gui_recovery=True) is False

    repaired = {
        **legacy,
        "final_evaluation": {"gui_records": 8.0, "gui_parse_rate": 0.75},
    }
    manifest.write_text(json.dumps(repaired), encoding="utf-8")
    assert _accepted(checkpoint, require_gui_recovery=True) is True


def test_resume_requires_measured_action_and_coordinate_correctness(tmp_path: Path):
    checkpoint = tmp_path / "accepted"
    checkpoint.mkdir()
    manifest = checkpoint / "distillation-manifest.json"
    legacy = {
        "accepted": True,
        "stage": "specialize",
        "final_evaluation": {"gui_records": 16.0, "gui_parse_rate": 0.984},
    }
    manifest.write_text(json.dumps(legacy), encoding="utf-8")
    assert _accepted(checkpoint) is True
    assert _accepted(checkpoint, require_gui_correctness=True) is False

    grounded = {
        **legacy,
        "final_evaluation": {
            **legacy["final_evaluation"],
            "gui_action_records": 16.0,
            "gui_action_name_match_rate": 0.75,
            "gui_coordinate_records": 10.0,
            "gui_median_coordinate_error": 42.0,
        },
    }
    manifest.write_text(json.dumps(grounded), encoding="utf-8")
    assert _accepted(checkpoint, require_gui_correctness=True) is True


def test_sparse_depth_6_lineage_is_not_resume_safe(tmp_path: Path):
    checkpoint = tmp_path / "depth-6-recovered"
    checkpoint.mkdir()
    unsafe = {
        "source_text_original_layer_ids": list(range(28)),
        "text_original_layer_ids": [0, 23, 24, 25, 26, 27],
    }
    (checkpoint / "compression-lineage.json").write_text(json.dumps(unsafe), encoding="utf-8")
    assert _tail_protected_lineage(checkpoint) is False

    safe = {
        "source_text_original_layer_ids": list(range(28)),
        "text_original_layer_ids": [0, 5, 10, 16, 26, 27],
    }
    (checkpoint / "compression-lineage.json").write_text(json.dumps(safe), encoding="utf-8")
    assert _tail_protected_lineage(checkpoint) is True


def test_k3_delta_clip_bounds_the_exponential_loss_term():
    import math

    clip = float(_plan()["stages"]["on_policy"]["k3_delta_clip"])
    assert 0 < clip <= 6
    assert math.exp(clip) < 1_000


def _plan_with_corpora(tmp_path: Path, plan: dict[str, object]) -> dict[str, object]:
    record = '{"messages": [{"role": "user", "content": "hi"}]}\n'
    plan = json.loads(json.dumps(plan))
    for group in ("recovery_corpora", "specialized_corpora"):
        for name, entry in plan[group].items():
            path = tmp_path / f"{group}-{name}.jsonl"
            path.write_text(record * 3, encoding="utf-8")
            entry["path"] = str(path)
            entry.pop("min_records", None)
    return plan


def test_1_5b_and_1b_recovery_mix_broad_and_multimodal_sources(tmp_path: Path):
    from scripts.distill_gui_owl_student import _stage_sources

    plan = _plan_with_corpora(tmp_path, _plan())
    recovery = plan["stages"]["recovery"]
    assert recovery["include_specialized"] is False
    specialized = set(plan["specialized_corpora"])

    for student_name in ("1.5b", "1b"):
        student = plan["students"][student_name]
        assert student["recovery_include_specialized"] is True
        sources = _stage_sources(
            plan,
            "recovery",
            recovery,
            student_entry=student,
            vision_trainable=False,
        )
        names = {source.name for source in sources}
        assert specialized.issubset(names)
        assert "broad_instruction_multimodal" in names
        assert any(source.text_only for source in sources)
        assert any(not source.text_only for source in sources)

    stage_c = {
        source.name
        for source in _stage_sources(plan, "specialize", plan["stages"]["specialize"])
    }
    assert stage_c == specialized


def test_text_stack_refuses_a_text_only_recovery_mix_even_with_frozen_vision(tmp_path: Path):
    import pytest

    from scripts.distill_gui_owl_student import _stage_sources

    plan = _plan_with_corpora(tmp_path, _plan())
    with pytest.raises(ValueError, match="trainable text stack"):
        _stage_sources(
            plan,
            "recovery",
            plan["stages"]["recovery"],
            student_entry={},
            vision_trainable=False,
        )


def test_trainable_vision_refuses_a_text_only_recovery_mix(tmp_path: Path):
    import pytest

    from scripts.distill_gui_owl_student import _stage_sources

    plan = _plan_with_corpora(tmp_path, _plan())
    with pytest.raises(ValueError, match="text and vision stacks"):
        _stage_sources(
            plan,
            "recovery",
            plan["stages"]["recovery"],
            student_entry={},
            vision_trainable=True,
        )

    assert plan["students"]["1.5b"]["recovery_include_specialized"] is True
    assert plan["students"]["1b"]["recovery_include_specialized"] is True
    assert plan["students"]["0.5b"]["recovery_include_specialized"] is True
    assert plan["students"]["0.5b"]["freeze_vision_during_kd"] is False
    sources = _stage_sources(
        plan,
        "recovery",
        plan["stages"]["recovery"],
        student_entry=plan["students"]["0.5b"],
        vision_trainable=True,
    )
    assert not {source.name for source in sources}.isdisjoint(
        set(plan["specialized_corpora"])
    )


def test_recovery_guard_rejects_a_truncated_corpus(tmp_path: Path):
    import pytest

    from scripts.run_gui_owl_progressive_distillation import _require_recovery_corpora

    corpus = tmp_path / "recovery.jsonl"
    corpus.write_text('{"messages": []}\n' * 5, encoding="utf-8")
    plan = {
        "recovery_corpora": {
            "broad": {"path": str(corpus), "required": True, "min_records": 20_000}
        }
    }
    with pytest.raises(ValueError, match="too small"):
        _require_recovery_corpora(plan)

    plan["recovery_corpora"]["broad"]["min_records"] = 5
    _require_recovery_corpora(plan)


def test_configured_recovery_corpus_declares_a_minimum_size():
    corpus = _plan()["recovery_corpora"]["broad_instruction_multimodal"]
    assert int(corpus["min_records"]) >= 10_000
    assert corpus["text_only"] is True
