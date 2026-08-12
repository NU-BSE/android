from __future__ import annotations

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


def test_progressive_sources_avoid_one_shot_depth_and_bad_half_b_cascade():
    plan = _plan()
    progressive = plan["progressive"]
    assert progressive["1.5b"]["text_depth_schedule"] == [24, 20, 16]
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


def test_compact_vision_keeps_source_deepstack_blocks_at_matching_target_taps():
    # Make high-saliency non-deepstack layers compete for every free slot. The
    # source taps 5/11/17 must still be retained exactly at compact 2/5/8.
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


def test_k3_delta_clip_bounds_the_exponential_loss_term():
    """K3 is exp(delta) - delta - 1, so the clip bounds the loss exponentially.

    A clip of 20 admits terms near exp(20) ~ 4.9e8, where a single token the
    teacher strongly prefers dictates the update direction; grad clipping caps
    the step length but not where it points.
    """

    import math

    clip = float(_plan()["stages"]["on_policy"]["k3_delta_clip"])
    assert 0 < clip <= 6
    assert math.exp(clip) < 1_000


def _plan_with_corpora(tmp_path: Path, plan: dict[str, object]) -> dict[str, object]:
    """Point the plan's corpora at real temp files so source selection can run."""

    record = '{"messages": [{"role": "user", "content": "hi"}]}\n'
    plan = json.loads(json.dumps(plan))
    for group in ("recovery_corpora", "specialized_corpora"):
        for name, entry in plan[group].items():
            path = tmp_path / f"{group}-{name}.jsonl"
            path.write_text(record * 3, encoding="utf-8")
            entry["path"] = str(path)
            entry.pop("min_records", None)
    return plan


def test_stage_a_is_broad_only_unless_the_student_trains_vision(tmp_path: Path):
    from scripts.distill_gui_owl_student import _stage_sources

    plan = _plan_with_corpora(tmp_path, _plan())
    recovery = plan["stages"]["recovery"]
    assert recovery["include_specialized"] is False

    specialized = set(plan["specialized_corpora"])
    broad_only = _stage_sources(plan, "recovery", recovery)
    assert specialized.isdisjoint({source.name for source in broad_only})

    # Stage C still specializes.
    names = {
        source.name
        for source in _stage_sources(plan, "specialize", plan["stages"]["specialize"])
    }
    assert names == specialized


def test_trainable_vision_refuses_a_text_only_recovery_mix(tmp_path: Path):
    """A pruned, unfrozen vision tower gets no gradient from text-only records.

    The broad recovery corpus is text-only by construction, so excluding the
    multimodal specialization sets would leave 0.5b's pruned tower untrained
    for the whole of stage A.
    """

    import pytest

    from scripts.distill_gui_owl_student import _stage_sources

    plan = _plan_with_corpora(tmp_path, _plan())
    with pytest.raises(ValueError, match="text_only"):
        _stage_sources(
            plan,
            "recovery",
            plan["stages"]["recovery"],
            vision_trainable=True,
        )

    # 0.5b is the only student that trains vision, and it opts back in.
    assert plan["students"]["0.5b"]["freeze_vision_during_kd"] is False
    assert plan["students"]["0.5b"]["recovery_include_specialized"] is True
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
    """Existence is not enough: a half-written corpus must not start a paid run."""

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
