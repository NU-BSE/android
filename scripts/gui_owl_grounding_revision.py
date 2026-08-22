"""Grounding-preserving helpers for GUI-Owl compression and distillation.

This revision is intentionally additive so old artifacts remain reproducible.
It strengthens two contracts exposed by the 1.5B grounding diagnostics:

* text-depth pruning must preserve the first three language blocks, the scaled
  tail, and bounded original-layer coverage from 28->24 onward; and
* GUI training records are damped inverse-frequency reweighted by target action
  so the click majority cannot dominate the entire GUI gradient.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import replace
from typing import Mapping, Sequence

from android_drlqn import gui_owl_student as core
from android_drlqn.gui_owl_metrics import assistant_content_text, parse_tool_call

TEXT_COVERAGE_BALANCED_MAX_DEPTH = 24
TEXT_VISUAL_INTEGRATION_PREFIX_LAYERS = 3
TEXT_LAYER_GAP_FACTOR = 1.75
GUI_ACTION_BALANCE_POWER = 0.5
GUI_ACTION_BALANCE_MAX_FACTOR = 3.0


def _validated_source(source_layer_ids: Sequence[int], target_layers: int) -> tuple[int, ...]:
    source = tuple(int(value) for value in source_layer_ids)
    if not source or tuple(sorted(source)) != source or len(set(source)) != len(source):
        raise ValueError("source layer IDs must be unique, monotonic, and non-empty")
    if not 1 <= target_layers <= len(source):
        raise ValueError("target layer count must fit inside source depth")
    return source


def protected_text_layer_ids(
    source_layer_ids: Sequence[int], target_layers: int
) -> set[int]:
    """Protect visual-integration prefix, boundaries, and a scaled text tail.

    Qwen3-VL consumes three DeepStack visual features in the first three
    language blocks. For production depths (>=4), a lineage starting at
    original layer 0 must therefore still contain original layers 0, 1 and 2;
    a later nested build cannot repair an upstream deletion of either layer.
    """

    source = _validated_source(source_layer_ids, target_layers)
    keep: set[int] = {source[0]}
    if target_layers > 1:
        keep.add(source[-1])

    prefix_budget = min(
        TEXT_VISUAL_INTEGRATION_PREFIX_LAYERS,
        max(1, target_layers - (1 if len(source) > 1 else 0)),
    )
    if source[0] == 0 and source[-1] >= 2 and target_layers >= 4:
        missing = [layer_id for layer_id in range(3) if layer_id not in source]
        if missing:
            raise ValueError(
                "source lineage already lost required visual-integration text layers: "
                + ", ".join(str(value) for value in missing)
            )
    keep.update(source[:prefix_budget])

    protect_tail = core.protected_text_tail_layers(target_layers)
    for layer_id in reversed(source[len(source) - protect_tail :]):
        if len(keep) >= target_layers:
            break
        keep.add(layer_id)
    return keep


def text_layer_gap_limit(source_layer_ids: Sequence[int], target_layers: int) -> int | None:
    """Bound new original-layer holes for every production depth <=24."""

    source = _validated_source(source_layer_ids, target_layers)
    if target_layers > TEXT_COVERAGE_BALANCED_MAX_DEPTH or target_layers <= 1:
        return None
    inherited = max((b - a for a, b in zip(source, source[1:])), default=1)
    span = max(1, source[-1] - source[0])
    ideal = span / max(1, target_layers - 1)
    return max(inherited, int(math.ceil(ideal * TEXT_LAYER_GAP_FACTOR)))


def text_layer_selection_is_safe(
    source_layer_ids: Sequence[int], selected_layer_ids: Sequence[int]
) -> bool:
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


def _score_sequence(scores: Mapping[object, object] | Sequence[object]) -> dict[int, float]:
    if isinstance(scores, Mapping):
        parsed = {int(index): float(value) for index, value in scores.items()}
    else:
        parsed = {index: float(value) for index, value in enumerate(scores)}
    if not parsed:
        raise ValueError("saliency scores cannot be empty")
    if any(not math.isfinite(value) for value in parsed.values()):
        raise ValueError("saliency scores must be finite")
    return parsed


def _coverage_constrained_selection(
    source: tuple[int, ...],
    score_by_id: Mapping[int, float],
    target_layers: int,
    required: set[int],
    gap_limit: int,
) -> tuple[int, ...]:
    """Maximize saliency subject to exact depth, anchors, and max gap."""

    if source[0] not in required or source[-1] not in required:
        raise ValueError("coverage-constrained selection requires protected boundaries")
    required_positions = {
        index for index, layer_id in enumerate(source) if layer_id in required
    }
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
                if best is None or candidate[0] > best[0] or (
                    candidate[0] == best[0] and candidate[1] < best[1]
                ):
                    best = candidate
            if best is not None:
                dp[(end, count)] = best

    result = dp.get((len(source) - 1, target_layers))
    if result is None:
        raise RuntimeError(
            "no grounding-safe text-layer map satisfies prefix/tail anchors "
            f"and max gap {gap_limit}"
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
    """Grounding-safe replacement for text-layer saliency selection."""

    source = _validated_source(source_layer_ids, target_layers)
    if target_layers == len(source):
        return source
    if protect_tail < 0:
        raise ValueError("protect_tail must not be negative")

    score_by_id = _score_sequence(scores)
    missing = [layer_id for layer_id in source if layer_id not in score_by_id]
    if missing:
        raise ValueError(f"missing saliency scores for layers: {missing}")

    if not protect_boundaries or protect_tail == 0:
        return core.select_salient_layer_ids(
            scores,
            source,
            target_layers,
            protect_boundaries=protect_boundaries,
            protect_tail=protect_tail,
        )

    required = protected_text_layer_ids(source, target_layers)
    gap_limit = text_layer_gap_limit(source, target_layers)
    if gap_limit is None:
        ranked = sorted(source, key=lambda layer_id: (-score_by_id[layer_id], layer_id))
        keep = set(required)
        for layer_id in ranked:
            if len(keep) >= target_layers:
                break
            keep.add(layer_id)
        result = tuple(sorted(keep))
    else:
        result = _coverage_constrained_selection(
            source, score_by_id, target_layers, required, gap_limit
        )

    if not text_layer_selection_is_safe(source, result):
        raise RuntimeError(f"grounding-safe text layer selection failed: {result}")
    return result


def _target_action(record: object) -> str | None:
    if str(getattr(record, "mode", "")) != "gui_policy":
        return None
    messages = getattr(record, "messages", None)
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)) or not messages:
        return None
    final = messages[-1]
    if not isinstance(final, Mapping):
        return None
    call = parse_tool_call(assistant_content_text(final.get("content")))
    return call.name if call is not None else None


def balance_gui_action_records(
    records: Sequence[object],
    *,
    power: float = GUI_ACTION_BALANCE_POWER,
    max_factor: float = GUI_ACTION_BALANCE_MAX_FACTOR,
) -> list[object]:
    """Damped inverse-frequency reweight GUI target actions without oversampling.

    The factors are normalized back to mean 1 over parsed GUI records, so the
    configured GUI corpus weight stays unchanged. This reduces click dominance
    while avoiding extreme weights for one-off long_press/wait examples.
    """

    if not 0 <= power <= 1:
        raise ValueError("GUI action balance power must be in [0, 1]")
    if max_factor < 1:
        raise ValueError("GUI action balance max_factor must be >= 1")

    actions = [_target_action(record) for record in records]
    counts = Counter(action for action in actions if action)
    if len(counts) <= 1:
        return list(records)

    largest = max(counts.values())
    raw = {
        action: min(max_factor, (largest / count) ** power)
        for action, count in counts.items()
    }
    normalizer = sum(counts[action] * raw[action] for action in counts) / sum(counts.values())
    factors = {action: raw[action] / normalizer for action in counts}

    output: list[object] = []
    for record, action in zip(records, actions):
        if action is None:
            output.append(record)
            continue
        output.append(
            replace(record, loss_weight=float(getattr(record, "loss_weight")) * factors[action])
        )
    return output


def balanced_corpus_records(original, sources, max_records):
    records = original(sources, max_records)
    balanced = balance_gui_action_records(records)
    counts = Counter(action for action in (_target_action(record) for record in records) if action)
    if counts:
        print(
            "GUI action-balanced targets: "
            + ", ".join(f"{name}={count}" for name, count in sorted(counts.items()))
        )
    return balanced


__all__ = [
    "TEXT_COVERAGE_BALANCED_MAX_DEPTH",
    "TEXT_VISUAL_INTEGRATION_PREFIX_LAYERS",
    "balance_gui_action_records",
    "balanced_corpus_records",
    "protected_text_layer_ids",
    "select_salient_layer_ids",
    "text_layer_gap_limit",
    "text_layer_selection_is_safe",
]
