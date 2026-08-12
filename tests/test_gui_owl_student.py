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
    ranked_indices,
    relative_parameter_error,
    resolve_peft_adapter_dir,
    select_salient_layer_ids,
    source_positions_for_layer_ids,
    teacher_key_for_student,
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
    assert DEFAULT_DEPTH_SCHEDULES["1.5b"] == (24, 20, 16)
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
    assert half_b["vision_config"]["deepstack_visual_indexes"] == [3, 7, 11]


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
