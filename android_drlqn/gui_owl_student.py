"""Planning helpers for calibrated, progressive GUI-Owl student compression.

The first distillation pipeline proved that one-shot depth cuts and prefix
truncation of hidden dimensions can yield valid checkpoints that nevertheless
collapse at generation time.  This module therefore keeps architecture
planning dependency-light while making the production compression contract
explicit:

* depth reductions are progressive and use calibration-derived layer saliency;
* narrow students require explicit activation-ranked channel/head selections;
* width pruning never falls back to taking the first N hidden dimensions; and
* generation gates are part of checkpoint acceptance rather than an optional
  post-export smoke test.
"""
from __future__ import annotations

import copy
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence


_TEXT_LAYER_KEY = re.compile(r"^model\.language_model\.layers\.(\d+)\.(.+)$")
_VISION_LAYER_KEY = re.compile(r"^model\.visual\.blocks\.(\d+)\.(.+)$")


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


@dataclass(frozen=True)
class GenerationGate:
    """Acceptance limits for free-running student generation."""

    first_token_eos_rate_max: float = 0.05
    empty_generation_rate_max: float = 0.05
    median_generated_tokens_min: float = 8.0
    mean_teacher_kl_max: float | None = None
    mean_supervised_ce_max: float | None = None
    gui_parse_rate_min: float | None = None
    gui_action_name_match_rate_min: float | None = None
    gui_median_coordinate_error_max: float | None = None

    def __post_init__(self) -> None:
        probabilities = (
            self.first_token_eos_rate_max,
            self.empty_generation_rate_max,
        )
        if any(not 0 <= value <= 1 for value in probabilities):
            raise ValueError("generation probability gates must be in [0, 1]")
        if self.median_generated_tokens_min < 0:
            raise ValueError("median generated token gate must be non-negative")
        if self.mean_teacher_kl_max is not None and self.mean_teacher_kl_max < 0:
            raise ValueError("teacher KL gate must be non-negative")
        if self.mean_supervised_ce_max is not None and self.mean_supervised_ce_max < 0:
            raise ValueError("supervised CE gate must be non-negative")
        if self.gui_parse_rate_min is not None and not 0 <= self.gui_parse_rate_min <= 1:
            raise ValueError("GUI parse-rate gate must be in [0, 1]")
        if (
            self.gui_action_name_match_rate_min is not None
            and not 0 <= self.gui_action_name_match_rate_min <= 1
        ):
            raise ValueError("GUI action-name match gate must be in [0, 1]")
        if (
            self.gui_median_coordinate_error_max is not None
            and self.gui_median_coordinate_error_max < 0
        ):
            raise ValueError("GUI coordinate-error gate must be non-negative")


DEFAULT_STUDENT_SPECS: dict[str, StudentSpec] = {
    "1.5b": StudentSpec(
        name="1.5b",
        target_parameters=1_500_000_000,
        text_layers=20,
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

# Depth-only students are not built in one jump.  The 1B schedule deliberately
# starts from the recovered 20-layer 1.5B checkpoint, not from the 28-layer 2B.
DEFAULT_DEPTH_SCHEDULES: dict[str, tuple[int, ...]] = {
    "1.5b": (24, 20),
    "1b": (12, 9, 6),
    "0.5b": (),
}


def evenly_spaced_layer_map(teacher_layers: int, student_layers: int) -> tuple[int, ...]:
    """Legacy deterministic map retained for diagnostics/tests only.

    Production builders require a calibration manifest.  This helper remains
    useful when comparing a saliency-selected map with the old one-shot map.
    """

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


def _score_sequence(scores: Mapping[object, object] | Sequence[object]) -> dict[int, float]:
    if isinstance(scores, Mapping):
        parsed = {int(index): float(value) for index, value in scores.items()}
    else:
        parsed = {index: float(value) for index, value in enumerate(scores)}
    if not parsed:
        raise ValueError("saliency scores cannot be empty")
    if any(value != value or value in (float("inf"), float("-inf")) for value in parsed.values()):
        raise ValueError("saliency scores must be finite")
    return parsed


TEXT_TAIL_PROTECTED_LAYERS = 6
"""Maximum final text blocks retained regardless of saliency.

Six tail blocks are necessary around the 20-layer regime that fixed the EOS
collapse, but treating six as a constant at every target depth is pathological:
at depth 6 it consumes five slots in addition to the first boundary and yields
maps such as [0, 23, 24, 25, 26, 27]. Production selection therefore scales
the protected tail with target depth while keeping six as the upper bound.
"""

TEXT_COVERAGE_BALANCED_MAX_DEPTH = 12
"""At or below this depth, enforce an explicit maximum original-layer gap."""

TEXT_LAYER_GAP_FACTOR = 1.75
"""Maximum small-depth gap relative to ideal even spacing, before inherited gaps."""


def protected_text_tail_layers(target_layers: int) -> int:
    """Scale tail protection to roughly 30% of the target, capped at six blocks."""

    if target_layers < 1:
        raise ValueError("target layer count must be positive")
    return min(TEXT_TAIL_PROTECTED_LAYERS, max(1, (3 * target_layers + 9) // 10))


def protected_text_layer_ids(
    source_layer_ids: Sequence[int], target_layers: int
) -> set[int]:
    """Return boundary and scaled-tail IDs that a structural map must retain."""

    source = tuple(int(value) for value in source_layer_ids)
    if not source or tuple(sorted(source)) != source or len(set(source)) != len(source):
        raise ValueError("source layer IDs must be unique, monotonic, and non-empty")
    if not 1 <= target_layers <= len(source):
        raise ValueError("target layer count must fit inside source depth")
    keep: set[int] = {source[0]}
    if target_layers > 1:
        keep.add(source[-1])
    protect_tail = protected_text_tail_layers(target_layers)
    for layer_id in reversed(source[len(source) - protect_tail :]):
        if len(keep) >= target_layers:
            break
        keep.add(layer_id)
    return keep


def text_layer_gap_limit(source_layer_ids: Sequence[int], target_layers: int) -> int | None:
    """Return a fail-closed max original-layer gap for small target depths.

    The bound never demands a gap smaller than one already present in the
    recovered predecessor, because a nested selector cannot reintroduce layers
    removed upstream. It does prevent later stages from creating a new giant
    hole such as the 23-layer depth-6 gap.
    """

    source = tuple(int(value) for value in source_layer_ids)
    if target_layers > TEXT_COVERAGE_BALANCED_MAX_DEPTH or target_layers <= 1:
        return None
    if len(source) < target_layers or tuple(sorted(source)) != source:
        raise ValueError("target layer count must fit inside a monotonic source")
    inherited = max((b - a for a, b in zip(source, source[1:])), default=1)
    span = max(1, source[-1] - source[0])
    ideal = span / max(1, target_layers - 1)
    return max(inherited, int(math.ceil(ideal * TEXT_LAYER_GAP_FACTOR)))


def text_layer_selection_is_safe(
    source_layer_ids: Sequence[int], selected_layer_ids: Sequence[int]
) -> bool:
    """Validate scaled tail protection and coverage for a nested structural map."""

    source = tuple(int(value) for value in source_layer_ids)
    selected = tuple(int(value) for value in selected_layer_ids)
    if not source or not selected or len(selected) > len(source):
        return False
    if tuple(sorted(source)) != source or tuple(sorted(selected)) != selected:
        return False
    if len(set(source)) != len(source) or len(set(selected)) != len(selected):
        return False
    if not set(selected).issubset(set(source)):
        return False
    try:
        required = protected_text_layer_ids(source, len(selected))
        gap_limit = text_layer_gap_limit(source, len(selected))
    except ValueError:
        return False
    if not required.issubset(set(selected)):
        return False
    if gap_limit is not None:
        max_gap = max((b - a for a, b in zip(selected, selected[1:])), default=0)
        if max_gap > gap_limit:
            return False
    return True


def _coverage_constrained_saliency_selection(
    source: tuple[int, ...],
    score_by_id: Mapping[int, float],
    target_layers: int,
    required: set[int],
    gap_limit: int,
) -> tuple[int, ...]:
    """Maximize saliency subject to exact depth, required IDs, and max gap.

    Production text pruning protects both source boundaries, so the dynamic
    program starts at the first source block and ends at the last. Transitions
    may not jump over another required block. With at most 28 source layers and
    target depth at most 12 this O(K*N^2) search is negligible compared with
    loading a checkpoint and gives an actual coverage guarantee rather than a
    heuristic bucket preference.
    """

    if source[0] not in required or source[-1] not in required:
        raise ValueError("coverage-constrained selection requires protected boundaries")
    required_positions = {
        index for index, layer_id in enumerate(source) if layer_id in required
    }
    # (source_position, selected_count) -> (saliency_sum, selected_original_ids)
    dp: dict[tuple[int, int], tuple[float, tuple[int, ...]]] = {
        (0, 1): (score_by_id[source[0]], (source[0],))
    }
    for end in range(1, len(source)):
        for count in range(2, target_layers + 1):
            best: tuple[float, tuple[int, ...]] | None = None
            for previous in range(end):
                state = dp.get((previous, count - 1))
                if state is None:
                    continue
                if source[end] - source[previous] > gap_limit:
                    continue
                if any(pos in required_positions for pos in range(previous + 1, end)):
                    continue
                candidate_path = state[1] + (source[end],)
                candidate = (state[0] + score_by_id[source[end]], candidate_path)
                if best is None:
                    best = candidate
                    continue
                if candidate[0] > best[0] or (
                    candidate[0] == best[0] and candidate[1] < best[1]
                ):
                    best = candidate
            if best is not None:
                dp[(end, count)] = best

    result = dp.get((len(source) - 1, target_layers))
    if result is None:
        raise RuntimeError(
            "no small-depth text-layer map satisfies the required tail/boundary "
            f"layers and max gap {gap_limit}"
        )
    selected = result[1]
    if not required.issubset(set(selected)):
        raise RuntimeError("coverage-constrained selection dropped a required layer")
    return selected


def select_salient_layer_ids(
    scores: Mapping[object, object] | Sequence[object],
    source_layer_ids: Sequence[int],
    target_layers: int,
    *,
    protect_boundaries: bool = True,
    protect_tail: int = 0,
) -> tuple[int, ...]:
    """Choose a nested subset of original layer IDs by calibration saliency.

    ``source_layer_ids`` is the lineage of the current checkpoint. Returning
    original IDs rather than current positions keeps repeated progressive
    reductions stable across intermediate checkpoints. A requested tail count
    is treated as a maximum and automatically scaled down with target depth, so
    legacy callers that still pass ``protect_tail=6`` cannot recreate the
    depth-6 [0,23,24,25,26,27] pathology. At depth <=12, selection maximizes
    saliency subject to an explicit max-gap coverage constraint.
    """

    source = tuple(int(value) for value in source_layer_ids)
    if not source or len(set(source)) != len(source):
        raise ValueError("source layer IDs must be unique and non-empty")
    if tuple(sorted(source)) != source:
        raise ValueError("source layer IDs must be monotonic")
    if not 1 <= target_layers <= len(source):
        raise ValueError("target layer count must fit inside source depth")
    if target_layers == len(source):
        return source

    score_by_id = _score_sequence(scores)
    missing = [layer_id for layer_id in source if layer_id not in score_by_id]
    if missing:
        raise ValueError(f"missing saliency scores for layers: {missing}")

    if protect_tail < 0:
        raise ValueError("protect_tail must not be negative")
    effective_tail = min(protect_tail, protected_text_tail_layers(target_layers))

    keep: set[int] = set()
    if protect_boundaries:
        keep.add(source[0])
        if target_layers > 1:
            keep.add(source[-1])
    for layer_id in reversed(source[len(source) - effective_tail :] if effective_tail else ()):
        if len(keep) >= target_layers:
            break
        keep.add(layer_id)

    gap_limit = text_layer_gap_limit(source, target_layers)
    if (
        gap_limit is not None
        and protect_boundaries
        and target_layers <= TEXT_COVERAGE_BALANCED_MAX_DEPTH
    ):
        result = _coverage_constrained_saliency_selection(
            source,
            score_by_id,
            target_layers,
            keep,
            gap_limit,
        )
    else:
        ranked = sorted(source, key=lambda layer_id: (-score_by_id[layer_id], layer_id))
        for layer_id in ranked:
            if len(keep) >= target_layers:
                break
            keep.add(layer_id)
        if len(keep) != target_layers:
            raise RuntimeError("failed to select the requested number of layers")
        result = tuple(sorted(keep))

    if gap_limit is not None:
        max_gap = max((b - a for a, b in zip(result, result[1:])), default=0)
        if max_gap > gap_limit:
            raise RuntimeError(
                f"small-depth layer selection has max gap {max_gap} > allowed {gap_limit}: {result}"
            )
    return result


def source_positions_for_layer_ids(
    source_layer_ids: Sequence[int], selected_layer_ids: Sequence[int]
) -> tuple[int, ...]:
    source = tuple(int(value) for value in source_layer_ids)
    positions = {layer_id: index for index, layer_id in enumerate(source)}
    selected = tuple(int(value) for value in selected_layer_ids)
    if any(layer_id not in positions for layer_id in selected):
        raise ValueError("selected layer ID is absent from source lineage")
    return tuple(positions[layer_id] for layer_id in selected)


def ranked_indices(scores: Sequence[object], count: int) -> tuple[int, ...]:
    """Return activation-ranked indices, sorted for deterministic tensor slicing."""

    values = tuple(float(value) for value in scores)
    if not 1 <= count <= len(values):
        raise ValueError("selection count must fit inside score vector")
    if any(value != value or value in (float("inf"), float("-inf")) for value in values):
        raise ValueError("channel scores must be finite")
    selected = sorted(range(len(values)), key=lambda index: (-values[index], index))[:count]
    return tuple(sorted(selected))


def expand_head_indices(head_indices: Sequence[int], head_dim: int) -> tuple[int, ...]:
    if head_dim <= 0:
        raise ValueError("head_dim must be positive")
    heads = tuple(int(value) for value in head_indices)
    if any(value < 0 for value in heads) or len(set(heads)) != len(heads):
        raise ValueError("head indices must be unique non-negative integers")
    return tuple(
        head * head_dim + offset
        for head in heads
        for offset in range(head_dim)
    )


def expand_spatial_merge_indices(
    hidden_indices: Sequence[int], source_hidden_size: int, merge_units: int
) -> tuple[int, ...]:
    """Expand residual-channel indices through Qwen3-VL's patch-merger flattening."""

    if source_hidden_size <= 0 or merge_units <= 0:
        raise ValueError("source hidden size and merge units must be positive")
    hidden = tuple(int(value) for value in hidden_indices)
    if any(value < 0 or value >= source_hidden_size for value in hidden):
        raise ValueError("hidden index falls outside source hidden size")
    return tuple(
        unit * source_hidden_size + channel
        for unit in range(merge_units)
        for channel in hidden
    )


def build_student_config_dict(
    teacher_config: Mapping[str, object],
    spec: StudentSpec,
    *,
    text_layers: int | None = None,
) -> dict[str, object]:
    """Return a Qwen3-VL config shaped for a final or intermediate student."""

    if str(teacher_config.get("model_type") or "") != "qwen3_vl":
        raise ValueError("GUI-Owl student builder requires a qwen3_vl teacher config")
    text = teacher_config.get("text_config")
    vision = teacher_config.get("vision_config")
    if not isinstance(text, Mapping) or not isinstance(vision, Mapping):
        raise ValueError("teacher config must contain text_config and vision_config")

    output = copy.deepcopy(dict(teacher_config))
    text_out = copy.deepcopy(dict(text))
    vision_out = copy.deepcopy(dict(vision))
    target_text_layers = int(text_layers if text_layers is not None else spec.text_layers)

    head_dim = int(text_out.get("head_dim") or 0)
    if head_dim and spec.text_hidden_size != spec.text_attention_heads * head_dim:
        raise ValueError(
            "student hidden size must equal attention_heads * inherited head_dim"
        )

    text_out.update(
        {
            "num_hidden_layers": target_text_layers,
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


def teacher_key_for_student(
    student_key: str,
    text_layer_map: Sequence[int],
    vision_layer_map: Sequence[int] | None = None,
) -> str:
    """Resolve target state-dict keys to current-source block positions."""

    match = _TEXT_LAYER_KEY.match(student_key)
    if match:
        student_index = int(match.group(1))
        if student_index < 0 or student_index >= len(text_layer_map):
            raise IndexError("student language-layer key falls outside layer mapping")
        return f"model.language_model.layers.{text_layer_map[student_index]}.{match.group(2)}"
    match = _VISION_LAYER_KEY.match(student_key)
    if match and vision_layer_map is not None:
        student_index = int(match.group(1))
        if student_index < 0 or student_index >= len(vision_layer_map):
            raise IndexError("student vision-layer key falls outside layer mapping")
        return f"model.visual.blocks.{vision_layer_map[student_index]}.{match.group(2)}"
    return student_key


def relative_parameter_error(actual: int, target: int) -> float:
    if actual <= 0 or target <= 0:
        raise ValueError("parameter counts must be positive")
    return abs(actual - target) / target


def resolve_peft_adapter_dir(root: str | Path, adapter_name: str) -> Path:
    """Resolve a PEFT adapter saved either flat or below its named adapter."""

    root_path = Path(root)
    candidates = (root_path, root_path / adapter_name)
    for candidate in candidates:
        if (candidate / "adapter_config.json").is_file():
            return candidate
    if not root_path.is_dir():
        raise FileNotFoundError(f"teacher {adapter_name} adapter directory not found: {root_path}")
    raise FileNotFoundError(
        f"teacher {adapter_name} adapter_config.json not found under "
        f"{root_path} or {root_path / adapter_name}"
    )


def load_student_plan(path: str | Path) -> dict[str, object]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("student plan must be a JSON object")
    return dict(value)


def load_calibration_manifest(path: str | Path) -> dict[str, object]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("compression calibration manifest must be a JSON object")
    if int(value.get("schema_version") or 0) < 2:
        raise ValueError("compression calibration manifest must use schema_version >= 2")
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


def gate_from_mapping(value: Mapping[str, object] | None) -> GenerationGate:
    raw = dict(value or {})
    allowed = {field for field in GenerationGate.__dataclass_fields__}
    return GenerationGate(**{key: raw[key] for key in raw if key in allowed})


def generation_gate_failures(
    metrics: Mapping[str, object], gate: GenerationGate
) -> tuple[str, ...]:
    """Return human-readable gate failures; empty tuple means accepted."""

    failures: list[str] = []

    def number(name: str) -> float:
        if name not in metrics:
            raise ValueError(f"generation metrics missing required field: {name}")
        return float(metrics[name])

    first_eos = number("first_token_eos_rate")
    if first_eos > gate.first_token_eos_rate_max:
        failures.append(
            f"first_token_eos_rate={first_eos:.4f} > {gate.first_token_eos_rate_max:.4f}"
        )
    empty = number("empty_generation_rate")
    if empty > gate.empty_generation_rate_max:
        failures.append(
            f"empty_generation_rate={empty:.4f} > {gate.empty_generation_rate_max:.4f}"
        )
    median_tokens = number("median_generated_tokens")
    if median_tokens < gate.median_generated_tokens_min:
        failures.append(
            f"median_generated_tokens={median_tokens:.2f} < {gate.median_generated_tokens_min:.2f}"
        )
    if gate.mean_teacher_kl_max is not None:
        value = number("mean_teacher_kl")
        if value > gate.mean_teacher_kl_max:
            failures.append(f"mean_teacher_kl={value:.4f} > {gate.mean_teacher_kl_max:.4f}")
    if gate.mean_supervised_ce_max is not None:
        value = number("mean_supervised_ce")
        if value > gate.mean_supervised_ce_max:
            failures.append(
                f"mean_supervised_ce={value:.4f} > {gate.mean_supervised_ce_max:.4f}"
            )
    if gate.gui_parse_rate_min is not None:
        measured = metrics.get("gui_records")
        if measured is not None and float(measured) <= 0:
            failures.append(
                "gui_parse_rate is unmeasured: the evaluation holdout contains "
                "no gui_policy records"
            )
        else:
            value = number("gui_parse_rate")
            if value < gate.gui_parse_rate_min:
                failures.append(f"gui_parse_rate={value:.4f} < {gate.gui_parse_rate_min:.4f}")
    if gate.gui_action_name_match_rate_min is not None:
        measured = number("gui_action_records")
        if measured <= 0:
            failures.append("gui_action_name_match_rate is unmeasured: no parseable target actions")
        else:
            value = number("gui_action_name_match_rate")
            if value < gate.gui_action_name_match_rate_min:
                failures.append(
                    f"gui_action_name_match_rate={value:.4f} < "
                    f"{gate.gui_action_name_match_rate_min:.4f}"
                )
    if gate.gui_median_coordinate_error_max is not None:
        measured = number("gui_coordinate_records")
        if measured <= 0:
            failures.append("gui_median_coordinate_error is unmeasured: no coordinate targets")
        else:
            value = number("gui_median_coordinate_error")
            if value > gate.gui_median_coordinate_error_max:
                failures.append(
                    f"gui_median_coordinate_error={value:.2f} > "
                    f"{gate.gui_median_coordinate_error_max:.2f}"
                )
    return tuple(failures)


__all__ = [
    "DEFAULT_DEPTH_SCHEDULES",
    "DEFAULT_STUDENT_SPECS",
    "GenerationGate",
    "StudentSpec",
    "TEXT_COVERAGE_BALANCED_MAX_DEPTH",
    "TEXT_LAYER_GAP_FACTOR",
    "TEXT_TAIL_PROTECTED_LAYERS",
    "build_student_config_dict",
    "evenly_spaced_layer_map",
    "expand_head_indices",
    "expand_spatial_merge_indices",
    "gate_from_mapping",
    "generation_gate_failures",
    "load_calibration_manifest",
    "load_student_plan",
    "protected_text_layer_ids",
    "protected_text_tail_layers",
    "ranked_indices",
    "relative_parameter_error",
    "resolve_peft_adapter_dir",
    "select_salient_layer_ids",
    "source_positions_for_layer_ids",
    "spec_from_plan",
    "teacher_key_for_student",
    "text_layer_gap_limit",
    "text_layer_selection_is_safe",
]
