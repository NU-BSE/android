from __future__ import annotations

import pytest

from android_drlqn.gui_owl_student import (
    DEFAULT_STUDENT_SPECS,
    build_student_config_dict,
    evenly_spaced_layer_map,
    prefix_copy_shape,
    relative_parameter_error,
    resolve_peft_adapter_dir,
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


def test_layer_maps_cover_first_and_last_teacher_blocks():
    mapping = evenly_spaced_layer_map(28, 16)
    assert len(mapping) == 16
    assert mapping[0] == 0
    assert mapping[-1] == 27
    assert tuple(sorted(set(mapping))) == mapping

    one_b = evenly_spaced_layer_map(28, 6)
    assert one_b == (0, 5, 11, 16, 22, 27)


def test_student_config_preserves_token_contract_and_changes_architecture():
    one_b = build_student_config_dict(TEACHER_CONFIG, DEFAULT_STUDENT_SPECS["1b"])
    assert one_b["image_token_id"] == TEACHER_CONFIG["image_token_id"]
    assert one_b["text_config"]["vocab_size"] == 151936
    assert one_b["text_config"]["num_hidden_layers"] == 6
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


def test_state_dict_language_layers_route_to_selected_teacher_blocks():
    mapping = evenly_spaced_layer_map(28, 6)
    assert teacher_key_for_student(
        "model.language_model.layers.2.self_attn.q_proj.weight", mapping
    ) == "model.language_model.layers.11.self_attn.q_proj.weight"
    assert teacher_key_for_student("model.visual.patch_embed.proj.weight", mapping) == (
        "model.visual.patch_embed.proj.weight"
    )


def test_prefix_copy_only_allows_narrower_equal_rank_tensors():
    slices = prefix_copy_shape((1024, 2048), (2048, 2048))
    assert slices == (slice(0, 1024), slice(0, 2048))
    assert prefix_copy_shape((2048, 2048), (1024, 2048)) is None
    assert prefix_copy_shape((1024,), (1024, 1)) is None


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
