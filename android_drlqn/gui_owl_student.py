"""Architecture planning helpers for GUI-Owl Qwen3-VL student distillation.

This module is intentionally dependency-light: configuration shaping, layer
routing, and checkpoint-key remapping can be tested without downloading model
weights or importing PyTorch/Transformers.
"""
from __future__ import annotations

import copy
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence


_LAYER_KEY = re.compile(r"^model\.language_model\.layers\.(\d+)\.(.+)$")


@dataclass(frozen=True)
class StudentSpec:
    """Target architecture for one compact Qwen3-VL student."""

    name: str
    target_parameters: int
    text_layers: int
    text_hidden_size: int
    text_intermediate_size: int
    text_attention_heads: int
    text_key_value_heads: int
    vision_depth: int
    vision_hidden_size: int
    vision_intermediate_size: int
    vision_heads: int
    deepstack_visual_indexes: tuple[int, ...]
    freeze_vision_during_kd: bool
    max_relative_parameter_error: float = 0.15

    def __post_init__(self) -> None:
        if self.target_parameters <= 0:
            raise ValueError("target_parameters must be positive")
        positive = (
            self.text_layers,
            self.text_hidden_size,
            self.text_intermediate_size,
            self.text_attention_heads,
            self.text_key_value_heads,
            self.vision_depth,
            self.vision_hidden_size,
            self.vision_intermediate_size,
            self.vision_heads,
        )
        if min(positive) <= 0:
            raise ValueError("student dimensions must be positive")
        if self.text_hidden_size % self.text_attention_heads:
            raise ValueError("text hidden size must be divisible by attention heads")
        if self.text_attention_heads % self.text_key_value_heads:
            raise ValueError("attention heads must be divisible by key/value heads")
        if self.vision_hidden_size % self.vision_heads:
            raise ValueError("vision hidden size must be divisible by vision heads")
        if any(index < 0 or index >= self.vision_depth for index in self.deepstack_visual_indexes):
            raise ValueError("deepstack visual indexes must fall inside the vision depth")
        if not 0 < self.max_relative_parameter_error < 1:
            raise ValueError("parameter tolerance must be in (0, 1)")


DEFAULT_STUDENT_SPECS: dict[str, StudentSpec] = {
    # Keep the complete GUI-Owl vision tower and shrink only the language depth.
    # With the published 2B architecture, 16 text blocks lands close to 1.5B.
    "1.5b": StudentSpec(
        name="1.5b",
        target_parameters=1_500_000_000,
        text_layers=16,
        text_hidden_size=2048,
        text_intermediate_size=6144,
        text_attention_heads=16,
        text_key_value_heads=8,
        vision_depth=24,
        vision_hidden_size=1024,
        vision_intermediate_size=4096,
        vision_heads=16,
        deepstack_visual_indexes=(5, 11, 17),
        freeze_vision_during_kd=True,
    ),
    # Six 2048-wide text blocks preserve the original tokenizer/vision interface
    # while taking the full checkpoint to roughly the 1B class.
    "1b": StudentSpec(
        name="1b",
        target_parameters=1_000_000_000,
        text_layers=6,
        text_hidden_size=2048,
        text_intermediate_size=6144,
        text_attention_heads=16,
        text_key_value_heads=8,
        vision_depth=24,
        vision_hidden_size=1024,
        vision_intermediate_size=4096,
        vision_heads=16,
        deepstack_visual_indexes=(5, 11, 17),
        freeze_vision_during_kd=True,
    ),
    # A genuine ~0.5B student cannot retain the 2B model's 2048-wide embedding
    # and full vision tower. Both modalities are narrowed; the 1B distilled
    # student is the recommended logit teacher for the final KD stage.
    "0.5b": StudentSpec(
        name="0.5b",
        target_parameters=500_000_000,
        text_layers=16,
        text_hidden_size=1024,
        text_intermediate_size=3072,
        text_attention_heads=8,
        text_key_value_heads=4,
        vision_depth=12,
        vision_hidden_size=768,
        vision_intermediate_size=3072,
        vision_heads=12,
        deepstack_visual_indexes=(3, 7, 11),
        freeze_vision_during_kd=False,
        max_relative_parameter_error=0.20,
    ),
}


def evenly_spaced_layer_map(teacher_layers: int, student_layers: int) -> tuple[int, ...]:
    """Map every student block to a monotonic, evenly spaced teacher block."""

    if teacher_layers < 1 or student_layers < 1:
        raise ValueError("layer counts must be positive")
    if student_layers > teacher_layers:
        raise ValueError("student depth cannot exceed initialization teacher depth")
    if student_layers == 1:
        return (teacher_layers - 1,)
    result = tuple(
        round(index * (teacher_layers - 1) / (student_layers - 1))
        for index in range(student_layers)
    )
    if len(set(result)) != len(result):
        raise ValueError("layer mapping produced duplicate teacher blocks")
    return result


def build_student_config_dict(
    teacher_config: Mapping[str, object],
    spec: StudentSpec,
) -> dict[str, object]:
    """Return a Qwen3-VL config dictionary shaped for ``spec``.

    Special-token IDs, vocabulary, RoPE settings, patch sizes and processor
    contracts remain inherited from the teacher. Only architecture dimensions
    needed for compression are changed.
    """

    if str(teacher_config.get("model_type") or "") != "qwen3_vl":
        raise ValueError("GUI-Owl student builder requires a qwen3_vl teacher config")
    text = teacher_config.get("text_config")
    vision = teacher_config.get("vision_config")
    if not isinstance(text, Mapping) or not isinstance(vision, Mapping):
        raise ValueError("teacher config must contain text_config and vision_config")

    output = copy.deepcopy(dict(teacher_config))
    text_out = copy.deepcopy(dict(text))
    vision_out = copy.deepcopy(dict(vision))

    head_dim = int(text_out.get("head_dim") or 0)
    if head_dim and spec.text_hidden_size != spec.text_attention_heads * head_dim:
        raise ValueError(
            "student hidden size must equal attention_heads * inherited head_dim"
        )

    text_out.update(
        {
            "num_hidden_layers": spec.text_layers,
            "hidden_size": spec.text_hidden_size,
            "intermediate_size": spec.text_intermediate_size,
            "num_attention_heads": spec.text_attention_heads,
            "num_key_value_heads": spec.text_key_value_heads,
        }
    )
    vision_out.update(
        {
            "depth": spec.vision_depth,
            "hidden_size": spec.vision_hidden_size,
            "intermediate_size": spec.vision_intermediate_size,
            "num_heads": spec.vision_heads,
            "out_hidden_size": spec.text_hidden_size,
            "deepstack_visual_indexes": list(spec.deepstack_visual_indexes),
        }
    )
    output["text_config"] = text_out
    output["vision_config"] = vision_out
    output["tie_word_embeddings"] = True
    return output


def teacher_key_for_student(student_key: str, layer_map: Sequence[int]) -> str:
    """Resolve a student state-dict key to the teacher key used for initialization."""

    match = _LAYER_KEY.match(student_key)
    if not match:
        return student_key
    student_index = int(match.group(1))
    if student_index < 0 or student_index >= len(layer_map):
        raise IndexError("student language-layer key falls outside layer mapping")
    return f"model.language_model.layers.{layer_map[student_index]}.{match.group(2)}"


def prefix_copy_shape(student_shape: Sequence[int], teacher_shape: Sequence[int]) -> tuple[slice, ...] | None:
    """Return safe prefix slices when a narrower student tensor fits in a teacher tensor."""

    if len(student_shape) != len(teacher_shape):
        return None
    if any(int(dst) > int(src) for dst, src in zip(student_shape, teacher_shape)):
        return None
    return tuple(slice(0, int(size)) for size in student_shape)


def relative_parameter_error(actual: int, target: int) -> float:
    if actual <= 0 or target <= 0:
        raise ValueError("parameter counts must be positive")
    return abs(actual - target) / target


def load_student_plan(path: str | Path) -> dict[str, object]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("student plan must be a JSON object")
    return dict(value)


def spec_from_plan(plan: Mapping[str, object], name: str) -> StudentSpec:
    students = plan.get("students")
    if not isinstance(students, Mapping):
        return DEFAULT_STUDENT_SPECS[name]
    raw = students.get(name)
    if not isinstance(raw, Mapping):
        return DEFAULT_STUDENT_SPECS[name]
    defaults = asdict(DEFAULT_STUDENT_SPECS[name])
    overrides = {key: value for key, value in raw.items() if key in defaults}
    merged = {**defaults, **overrides, "name": name}
    merged["deepstack_visual_indexes"] = tuple(
        int(value) for value in merged["deepstack_visual_indexes"]
    )
    return StudentSpec(**merged)


__all__ = [
    "DEFAULT_STUDENT_SPECS",
    "StudentSpec",
    "build_student_config_dict",
    "evenly_spaced_layer_map",
    "load_student_plan",
    "prefix_copy_shape",
    "relative_parameter_error",
    "spec_from_plan",
    "teacher_key_for_student",
]
