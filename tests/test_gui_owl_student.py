from __future__ import annotations

import pytest

from android_drlqn.gui_owl_student import (
    DEFAULT_DEPTH_SCHEDULES,
    DEFAULT_STUDENT_SPECS,
    GenerationGate,
    build_student_config_dict,
    evenly_spaced_layer_map,
    expand_head_indices,
    expand_spatial_merge_indices,
    generation_gate_failures,
    protected_text_tail_layers,
    ranked_indices,
    relative_parameter_error,
    resolve_peft_adapter_dir,
    select_salient_layer_ids,
    unreachable_gate_projection,
    source_positions_for_layer_ids,
    teacher_key_for_student,
    text_layer_gap_limit,
    text_layer_selection_is_safe,
)


TEACHER_CONFIG = {
    "architectures": ["Qwen3VLForConditionalGeneration"],
    "model_type": "qwen3_vl",
    "image_token_id": 151655,
    "video_token_id": 151656,
    "vision_start_token_id": 151652,
    "vision_end_token_id": 151653,
    "tie_word_embeddings": True,
    "text_config": {
        "model_type": "qwen3_vl_text",
        "vocab_size": 151936,
        "hidden_size": 2048,
        "intermediate_size": 6144,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "rope_theta": 5000000,
        "layer_types": ["full_attention"] * 28,
    },
    "vision_config": {
        "model_type": "qwen3_vl",
        "depth": 24,
        "hidden_size": 1024,
        "intermediate_size": 4096,
        "num_heads": 16,
        "out_hidden_size": 2048,
        "patch_size": 16,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
        "deepstack_visual_indexes": [5, 11, 17],
    },
}


def test_legacy_even_map_is_diagnostic_not_progressive_contract():
    mapping = evenly_spaced_layer_map(28, 6)
    assert mapping == (0, 5, 11, 16, 22, 27)
    assert DEFAULT_STUDENT_SPECS["1.5b"].text_layers == 20
    assert DEFAULT_DEPTH_SCHEDULES["1.5b"] == (24, 20)
    assert DEFAULT_DEPTH_SCHEDULES["1b"] == (12, 9, 6)


def test_saliency_selection_is_nested_and_preserves_boundaries():
    scores = [float(index) for index in range(28)]
    first = select_salient_layer_ids(scores, tuple(range(28)), 24)
    second = select_salient_layer_ids(scores, first, 20)
    third = select_salient_layer_ids(scores, second, 16)
    assert len(first) == 24
    assert len(second) == 20
    assert len(third) == 16
    assert set(third).issubset(second)
    assert set(second).issubset(first)
    assert third[0] == 0
    assert third[-1] == 27
    assert source_positions_for_layer_ids(first, second) == tuple(
        first.index(value) for value in second
    )


def test_ranked_channel_and_group_expansion_are_not_prefix_selection():
    scores = [0.1, 9.0, 0.2, 8.0, 0.3]
    assert ranked_indices(scores, 2) == (1, 3)
    assert expand_head_indices((1, 3), 2) == (2, 3, 6, 7)
    assert expand_spatial_merge_indices((1, 3), 4, 2) == (1, 3, 5, 7)


def test_student_config_preserves_token_contract_and_changes_architecture():
    one_b = build_student_config_dict(
        TEACHER_CONFIG,
        DEFAULT_STUDENT_SPECS["1b"],
        text_layers=12,
    )
    assert one_b["image_token_id"] == TEACHER_CONFIG["image_token_id"]
    assert one_b["text_config"]["vocab_size"] == 151936
    assert one_b["text_config"]["num_hidden_layers"] == 12
    assert one_b["text_config"]["hidden_size"] == 2048
    assert one_b["vision_config"]["depth"] == 24
    assert one_b["vision_config"]["out_hidden_size"] == 2048

    half_b = build_student_config_dict(TEACHER_CONFIG, DEFAULT_STUDENT_SPECS["0.5b"])
    assert half_b["text_config"]["hidden_size"] == 1024
    assert half_b["text_config"]["num_attention_heads"] == 8
    assert half_b["vision_config"]["depth"] == 12
    assert half_b["vision_config"]["hidden_size"] == 768
    assert half_b["vision_config"]["out_hidden_size"] == 1024
    # Source taps 5/11/17 land at compact positions 2/5/8; the plan and the
    # deepstack-aligned normalizer agree, so the fallback spec must too.
    assert half_b["vision_config"]["deepstack_visual_indexes"] == [2, 5, 8]


def test_head_pruning_keeps_residual_width_and_inherited_head_dim():
    from dataclasses import replace

    from android_drlqn.gui_owl_student import width_pruned_axes

    # Qwen3 maps hidden -> heads * head_dim, so 12 heads of 128 on a 2048-wide
    # residual is a valid student even though 2048 % 12 != 0.
    narrow = replace(
        DEFAULT_STUDENT_SPECS["1.5b"],
        text_layers=28,
        text_attention_heads=12,
        text_key_value_heads=6,
        text_intermediate_size=4096,
    )
    config = build_student_config_dict(TEACHER_CONFIG, narrow)
    assert config["text_config"]["hidden_size"] == 2048
    assert config["text_config"]["num_attention_heads"] == 12
    assert config["text_config"]["num_key_value_heads"] == 6
    assert config["text_config"]["head_dim"] == 128
    assert config["text_config"]["intermediate_size"] == 4096
    assert width_pruned_axes(narrow, TEACHER_CONFIG) == {
        "text_mlp": (6144, 4096),
        "text_q_heads": (16, 12),
        "text_kv_heads": (8, 6),
    }
    assert width_pruned_axes(replace(narrow, text_attention_heads=16, text_key_value_heads=8, text_intermediate_size=6144), TEACHER_CONFIG) == {}

    with pytest.raises(ValueError, match="head_dim"):
        build_student_config_dict(TEACHER_CONFIG, replace(narrow, text_head_dim=64))
    with pytest.raises(ValueError, match="exceeds"):
        width_pruned_axes(replace(narrow, text_intermediate_size=8192), TEACHER_CONFIG)


def _width_calibration() -> dict[str, object]:
    layers = 3
    return {
        "schema_version": 2,
        "text": {
            "layers": layers,
            "hidden_size": 8,
            "attention_heads": 4,
            "key_value_heads": 2,
            "head_dim": 2,
            "hidden_scores": [0.1, 9.0, 0.2, 8.0, 0.3, 7.0, 0.4, 6.0],
            "mlp_scores": {str(i): [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] for i in range(layers)},
            "mlp_activation_scores": {
                str(i): [6.0, 5.0, 4.0, 3.0, 2.0, 1.0] for i in range(layers)
            },
            "q_head_scores": {str(i): [1.0, 1.0, 9.0, 9.0] for i in range(layers)},
            "kv_head_scores": {str(i): [0.0, 5.0] for i in range(layers)},
        },
        # The tower shape is known but no image ever exercised it.
        "vision": {"ready": False, "hidden_size": 1024, "attention_heads": 16},
        "recommended": {},
    }


def test_width_selections_are_derived_only_for_narrowed_axes():
    from dataclasses import replace

    from android_drlqn.gui_owl_student import derive_width_selections

    base = replace(
        DEFAULT_STUDENT_SPECS["1.5b"],
        text_layers=3,
        text_hidden_size=8,
        text_intermediate_size=6,
        text_attention_heads=4,
        text_key_value_heads=2,
        text_head_dim=2,
    )
    assert derive_width_selections(_width_calibration(), base) == {}

    mlp_only = replace(base, text_intermediate_size=3)
    derived = derive_width_selections(_width_calibration(), mlp_only)
    # The gated activation that enters down_proj outranks the gate pre-activation.
    assert derived["text_mlp_indices_3"] == {"0": [0, 1, 2], "1": [0, 1, 2], "2": [0, 1, 2]}
    assert derived["text_mlp_score_source"] == "mlp_activation_scores"
    assert "text_hidden_indices_8" not in derived
    assert not any(key.startswith("text_q_heads") for key in derived)

    legacy = _width_calibration()
    del legacy["text"]["mlp_activation_scores"]
    fallback = derive_width_selections(legacy, mlp_only)
    assert fallback["text_mlp_indices_3"]["0"] == [3, 4, 5]
    assert fallback["text_mlp_score_source"] == "mlp_scores"

    heads = replace(base, text_attention_heads=2, text_key_value_heads=1)
    derived = derive_width_selections(_width_calibration(), heads)
    # Whole GQA groups: KV head 1 keeps query heads 2 and 3.
    assert derived["text_kv_heads_1"]["0"] == [1]
    assert derived["text_q_heads_2"]["0"] == [2, 3]

    residual = replace(base, text_hidden_size=4)
    derived = derive_width_selections(_width_calibration(), residual)
    assert derived["text_hidden_indices_4"] == [1, 3, 5, 7]

    with pytest.raises(ValueError, match="vision"):
        derive_width_selections(_width_calibration(), replace(base, vision_hidden_size=512, vision_heads=8))


def test_builder_refuses_gate_energy_neuron_ranking_for_text_mlp():
    from dataclasses import replace

    from scripts.build_gui_owl_student import _WidthSelectors

    source_config = {
        "text_config": {
            "hidden_size": 8,
            "intermediate_size": 6,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 2,
        },
        "vision_config": {
            "hidden_size": 1024,
            "intermediate_size": 4096,
            "num_heads": 16,
            "spatial_merge_size": 2,
        },
    }
    spec = replace(
        DEFAULT_STUDENT_SPECS["1.5b"],
        text_layers=3,
        text_hidden_size=8,
        text_intermediate_size=3,
        text_attention_heads=4,
        text_key_value_heads=2,
        text_head_dim=2,
    )
    legacy = _width_calibration()
    del legacy["text"]["mlp_activation_scores"]
    # A cached recommendation from the retired ranking must not rescue it either.
    legacy["recommended"]["text_mlp_indices_3"] = {str(i): [3, 4, 5] for i in range(3)}
    with pytest.raises(RuntimeError, match="mlp_activation_scores"):
        _WidthSelectors(
            calibration=legacy,
            source_config=source_config,
            target_spec=spec,
            text_original_ids=(0, 1, 2),
            vision_original_ids=tuple(range(24)),
        )

    current = _width_calibration()
    current["recommended"]["text_mlp_indices_3"] = {str(i): [3, 4, 5] for i in range(3)}
    selectors = _WidthSelectors(
        calibration=current,
        source_config=source_config,
        target_spec=spec,
        text_original_ids=(0, 1, 2),
        vision_original_ids=tuple(range(24)),
    )
    # Derived from the gated activation, not the stale cached ranking.
    assert selectors.text_mlp(1) == (0, 1, 2)
    assert selectors.selection_sources == {"text_mlp": "derived_from_calibration_scores"}
    assert selectors.text_mlp_score_source == "mlp_activation_scores"
    # Untouched axes map to identity, so unchanged tensors copy exactly.
    assert selectors.text_q(0) == tuple(range(8))
    assert selectors.text_hidden == tuple(range(8))
    assert selectors.vision_mlp(0) == tuple(range(4096))


def test_state_dict_layer_routing_supports_text_and_vision():
    text_map = (0, 3, 7)
    vision_map = (1, 5)
    assert teacher_key_for_student(
        "model.language_model.layers.2.self_attn.q_proj.weight",
        text_map,
        vision_map,
    ) == "model.language_model.layers.7.self_attn.q_proj.weight"
    assert teacher_key_for_student(
        "model.visual.blocks.1.attn.qkv.weight",
        text_map,
        vision_map,
    ) == "model.visual.blocks.5.attn.qkv.weight"


def test_generation_gate_catches_eos_collapse_even_with_good_ce():
    gate = GenerationGate(
        first_token_eos_rate_max=0.05,
        empty_generation_rate_max=0.05,
        median_generated_tokens_min=8,
        mean_supervised_ce_max=3.0,
    )
    metrics = {
        "first_token_eos_rate": 1.0,
        "empty_generation_rate": 1.0,
        "median_generated_tokens": 0,
        "mean_teacher_kl": 2.0,
        "mean_supervised_ce": 2.36,
        "gui_parse_rate": 0.0,
    }
    failures = generation_gate_failures(metrics, gate)
    assert any("first_token_eos_rate" in value for value in failures)
    assert any("empty_generation_rate" in value for value in failures)
    assert any("median_generated_tokens" in value for value in failures)
    assert not any("mean_supervised_ce" in value for value in failures)


def test_tail_protection_is_scaled_before_small_depth_coverage_selection():
    scores = [10.0, 0.9, 0.8, 0.7, 0.6, 0.5] + [0.1, 0.1, 0.1, 0.1]
    source = tuple(range(10))

    unprotected = select_salient_layer_ids(scores, source, 6)
    assert unprotected == (0, 1, 2, 3, 4, 9)

    protected = select_salient_layer_ids(scores, source, 6, protect_tail=4)
    assert protected == (0, 1, 2, 4, 8, 9)
    assert {8, 9}.issubset(protected)


def test_scaled_tail_does_not_consume_small_depth_budget():
    assert protected_text_tail_layers(24) == 6
    assert protected_text_tail_layers(20) == 6
    assert protected_text_tail_layers(16) == 5
    assert protected_text_tail_layers(12) == 4
    assert protected_text_tail_layers(9) == 3
    assert protected_text_tail_layers(6) == 2


def test_depth_6_selection_cannot_collapse_to_first_plus_last_five():
    source = tuple(range(28))
    scores = [1.0] * 23 + [1000.0 + index for index in range(5)]
    selected = select_salient_layer_ids(
        scores,
        source,
        6,
        protect_tail=protected_text_tail_layers(6),
    )
    assert selected != (0, 23, 24, 25, 26, 27)
    assert {26, 27}.issubset(selected)
    assert text_layer_selection_is_safe(source, selected)
    limit = text_layer_gap_limit(source, 6)
    assert limit is not None
    assert max(b - a for a, b in zip(selected, selected[1:])) <= limit


def test_stale_recommended_map_cannot_bypass_tail_protection():
    from scripts.build_gui_owl_student import _select_text_layers

    source = tuple(range(28))
    unsafe = tuple(range(19)) + (27,)
    calibration = {
        "recommended": {"text_layers_by_depth": {"20": list(unsafe)}},
        "text": {"layer_scores": [100.0 - index for index in range(28)]},
    }
    selected = _select_text_layers(calibration, source, 20)
    assert selected != unsafe
    assert len(selected) == 20
    assert {22, 23, 24, 25, 26, 27}.issubset(selected)


def test_stale_depth_6_recommendation_cannot_bypass_coverage_guard():
    from scripts.build_gui_owl_student import _select_text_layers

    source = tuple(range(28))
    unsafe = (0, 23, 24, 25, 26, 27)
    calibration = {
        "recommended": {"text_layers_by_depth": {"6": list(unsafe)}},
        "text": {"layer_scores": [float(index) for index in range(28)]},
    }
    selected = _select_text_layers(calibration, source, 6)
    assert selected != unsafe
    assert len(selected) == 6
    assert text_layer_selection_is_safe(source, selected)


def test_tail_request_is_capped_when_depth_is_tight():
    scores = [10.0] + [0.1] * 9
    source = tuple(range(10))
    assert select_salient_layer_ids(scores, source, 3, protect_tail=4) == (0, 1, 9)
    with pytest.raises(ValueError, match="protect_tail"):
        select_salient_layer_ids(scores, source, 3, protect_tail=-1)


def test_scaled_tail_selections_stay_nested_across_the_depth_schedule():
    scores = [10.0] + [1.0 / (index + 1) for index in range(1, 27)] + [5.0]
    selected = tuple(range(28))
    previous = set(selected)
    for depth in (24, 20, 16, 12, 9, 6):
        selected = select_salient_layer_ids(
            scores,
            selected,
            depth,
            protect_tail=protected_text_tail_layers(depth),
        )
        assert len(selected) == depth
        assert set(selected).issubset(previous)
        assert text_layer_selection_is_safe(tuple(sorted(previous)), selected)
        previous = set(selected)


def test_gui_parse_gate_rejects_an_unmeasured_parse_rate():
    gate = GenerationGate(
        first_token_eos_rate_max=0.05,
        empty_generation_rate_max=0.05,
        median_generated_tokens_min=8,
        gui_parse_rate_min=0.5,
    )
    healthy = {
        "first_token_eos_rate": 0.0,
        "empty_generation_rate": 0.0,
        "median_generated_tokens": 64,
        "mean_teacher_kl": 2.49,
        "mean_supervised_ce": 2.31,
    }
    failures = generation_gate_failures({**healthy, "gui_records": 0.0, "gui_parse_rate": 0.0}, gate)
    assert any("unmeasured" in value for value in failures)

    measured = generation_gate_failures(
        {**healthy, "gui_records": 8.0, "gui_parse_rate": 0.75}, gate
    )
    assert not measured


def test_grounding_gate_rejects_syntactically_valid_center_tap_policy():
    gate = GenerationGate(
        gui_parse_rate_min=0.5,
        gui_action_name_match_rate_min=0.70,
        gui_median_coordinate_error_max=100.0,
    )
    common = {
        "first_token_eos_rate": 0.0,
        "empty_generation_rate": 0.0,
        "median_generated_tokens": 27.5,
        "gui_records": 64.0,
        "gui_action_records": 64.0,
        "gui_coordinate_records": 41.0,
    }
    student = generation_gate_failures(
        {
            **common,
            "gui_parse_rate": 0.984,
            "gui_action_name_match_rate": 0.641,
            "gui_median_coordinate_error": 340.0,
        },
        gate,
    )
    assert any("gui_action_name_match_rate" in value for value in student)
    assert any("gui_median_coordinate_error" in value for value in student)
    assert not any("gui_parse_rate" in value for value in student)

    teacher_like = generation_gate_failures(
        {
            **common,
            "gui_parse_rate": 0.938,
            "gui_action_name_match_rate": 0.828,
            "gui_median_coordinate_error": 4.0,
        },
        gate,
    )
    assert not teacher_like


def test_parameter_error_and_invalid_teacher_config():
    assert relative_parameter_error(1_050_000_000, 1_000_000_000) == pytest.approx(0.05)
    with pytest.raises(ValueError, match="qwen3_vl"):
        build_student_config_dict({"model_type": "other"}, DEFAULT_STUDENT_SPECS["1b"])


def test_resolve_peft_adapter_dir_supports_named_and_flat_layouts(tmp_path):
    named_root = tmp_path / "gui"
    named = named_root / "gui_policy"
    named.mkdir(parents=True)
    (named / "adapter_config.json").write_text("{}", encoding="utf-8")
    assert resolve_peft_adapter_dir(named_root, "gui_policy") == named

    flat = tmp_path / "chat"
    flat.mkdir()
    (flat / "adapter_config.json").write_text("{}", encoding="utf-8")
    assert resolve_peft_adapter_dir(flat, "chat") == flat


def test_resolve_peft_adapter_dir_requires_adapter_config(tmp_path):
    root = tmp_path / "missing"
    root.mkdir()
    with pytest.raises(FileNotFoundError, match="adapter_config.json"):
        resolve_peft_adapter_dir(root, "gui_policy")


def test_grounding_gates_are_calibrated_against_the_shipping_teacher():
    """The gate must be reachable by the teacher the student is distilled from.

    0.70 / 100px was calibrated against the teacher measured under a prompt
    nothing ships (2.2px). Under the shipping contract the same teacher scores
    action 0.762 and a 128.3px median over 256 records -- it failed both of its
    own students' gates, so no student could pass without beating its teacher.
    These are 85% retention of that measurement.
    """
    gate = GenerationGate(
        gui_action_name_match_rate_min=0.65,
        gui_median_coordinate_error_max=200.0,
        gui_coordinate_within_50_rate_min=0.38,
    )
    base = {
        "first_token_eos_rate": 0.0,
        "empty_generation_rate": 0.0,
        "median_generated_tokens": 41,
        "mean_teacher_kl": 1.6,
        "mean_supervised_ce": 1.2,
        "gui_action_records": 256.0,
        "gui_coordinate_records": 217.0,
    }
    teacher = {
        **base,
        "gui_action_name_match_rate": 0.762,
        "gui_median_coordinate_error": 128.3,
        "gui_coordinate_within_50_rate": 0.452,
    }
    assert not generation_gate_failures(teacher, gate)

    # depth-24 recovery, same prompt: short on action and grounding alike.
    student = {
        **base,
        "gui_action_name_match_rate": 0.609,
        "gui_median_coordinate_error": 279.2,
        "gui_coordinate_within_50_rate": 0.023,
    }
    failures = generation_gate_failures(student, gate)
    assert any("gui_action_name_match_rate" in value for value in failures)
    assert any("gui_coordinate_within_50_rate" in value for value in failures)


def test_within_50_gate_fails_when_no_coordinate_was_measured():
    """An unmeasured rate must not read as a perfect score.

    gui_coordinate_within_50_rate is 0.0 both when every point missed and when
    nothing was scored; gui_coordinate_records is what separates them.
    """
    gate = GenerationGate(gui_coordinate_within_50_rate_min=0.38)
    metrics = {
        "first_token_eos_rate": 0.0,
        "empty_generation_rate": 0.0,
        "median_generated_tokens": 41,
        "mean_teacher_kl": 1.6,
        "mean_supervised_ce": 1.2,
        "gui_coordinate_records": 0.0,
        "gui_coordinate_within_50_rate": 0.0,
    }
    assert any("unmeasured" in value for value in generation_gate_failures(metrics, gate))


def _history(trajectory, *names):
    rows = []
    for point in trajectory:
        metrics = {
            "first_token_eos_rate": 0.0,
            "empty_generation_rate": 0.0,
            "median_generated_tokens": 35,
        }
        metrics.update(dict(zip(names, point[1:])))
        rows.append({"update": point[0], "metrics": metrics})
    return rows


def test_stall_detection_catches_the_depth24_recovery_plateau():
    """The real trajectory that burned 1700 updates for nothing.

    depth-24 recovery reached the best-constant-guess coordinate error inside
    100 updates and held it for the rest of the stage:
    gui_coordinate_within_50_rate sat at 0.000-0.009 against a 0.38 bar for all
    1800 updates. The projection must call that before the budget is spent.
    """
    gate = GenerationGate(
        gui_action_name_match_rate_min=0.65,
        gui_coordinate_within_50_rate_min=0.38,
    )
    observed = [
        (0, 0.398, 0.000), (100, 0.516, 0.009), (200, 0.531, 0.000),
        (300, 0.547, 0.009), (400, 0.555, 0.000), (500, 0.578, 0.000),
    ]
    history = _history(observed, "gui_action_name_match_rate", "gui_coordinate_within_50_rate")
    reason = unreachable_gate_projection(history[:5], gate, final_update=1800)
    assert reason is not None and "gui_coordinate_within_50_rate" in reason


def test_stall_detection_leaves_a_climbing_stage_alone():
    """The contract bootstrap climbed ~13 points a chunk and passed at chunk 6.

    A stage improving toward its bar must never be cut short, which is the
    failure mode that would make early stopping worse than the waste it saves.
    """
    gate = GenerationGate(gui_parse_rate_min=0.65)
    climbing = [(100, 0.426), (200, 0.563), (300, 0.688), (400, 0.805), (500, 0.860)]
    assert unreachable_gate_projection(
        _history(climbing, "gui_parse_rate"), gate, final_update=800
    ) is None

    # Slow but real progress, and noise around an upward trend, both continue.
    slow = [(100, 0.30), (200, 0.34), (300, 0.38), (400, 0.42), (500, 0.46)]
    noisy = [(100, 0.30), (200, 0.36), (300, 0.33), (400, 0.41), (500, 0.39)]
    for trajectory in (slow, noisy):
        assert unreachable_gate_projection(
            _history(trajectory, "gui_parse_rate"), gate, final_update=1800
        ) is None


def test_stall_detection_needs_history_budget_and_a_gate():
    gate = GenerationGate(gui_parse_rate_min=0.65)
    flat = [(100, 0.30), (200, 0.31), (300, 0.30), (400, 0.31), (500, 0.30)]
    history = _history(flat, "gui_parse_rate")
    # Too few evaluations to fit a trend.
    assert unreachable_gate_projection(history[:3], gate, final_update=1800) is None
    # No budget left to project into.
    assert unreachable_gate_projection(history, gate, final_update=500) is None
    # Nothing gated on this metric.
    assert unreachable_gate_projection(history, GenerationGate(), final_update=1800) is None
    # Flat and far below the bar, with budget remaining: report it.
    assert unreachable_gate_projection(history, gate, final_update=1800) is not None
