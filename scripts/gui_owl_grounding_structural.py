"""Anchor-feasible structural selection for grounded GUI-Owl students."""
from __future__ import annotations

import math
from typing import Mapping, Sequence

from android_drlqn import gui_owl_student as core
from scripts import gui_owl_grounding_revision as common

TEXT_COVERAGE_BALANCED_MAX_DEPTH = 24
TEXT_LAYER_GAP_FACTOR = 1.75
protected_text_layer_ids = common.protected_text_layer_ids


def _validated_source(source_layer_ids: Sequence[int], target_layers: int) -> tuple[int, ...]:
    source = tuple(int(value) for value in source_layer_ids)
    if not source or tuple(sorted(source)) != source or len(set(source)) != len(source):
        raise ValueError("source layer IDs must be unique, monotonic, and non-empty")
    if not 1 <= target_layers <= len(source):
        raise ValueError("target layer count must fit inside source depth")
    return source


def _feasible_with_gap(
    source: tuple[int, ...], target_layers: int, required: set[int], gap_limit: int
) -> bool:
    required_positions = {
        index for index, layer_id in enumerate(source) if layer_id in required
    }
    states: set[tuple[int, int]] = {(0, 1)}
    for end in range(1, len(source)):
        for count in range(2, target_layers + 1):
            for previous in range(end):
                if (previous, count - 1) not in states:
                    continue
                if source[end] - source[previous] > gap_limit:
                    continue
                if any(pos in required_positions for pos in range(previous + 1, end)):
                    continue
                states.add((end, count))
                break
    return (len(source) - 1, target_layers) in states


def text_layer_gap_limit(source_layer_ids: Sequence[int], target_layers: int) -> int | None:
    """Return the tightest coverage bound compatible with required anchors.

    The ordinary 1.75x-even-spacing bound is retained whenever feasible. If the
    protected prefix/tail themselves make it impossible (notably depth 6), the
    bound grows only to the smallest integer that admits an exact nested map.
    """

    source = _validated_source(source_layer_ids, target_layers)
    if target_layers > TEXT_COVERAGE_BALANCED_MAX_DEPTH or target_layers <= 1:
        return None
    inherited = max((b - a for a, b in zip(source, source[1:])), default=1)
    span = max(1, source[-1] - source[0])
    ideal = span / max(1, target_layers - 1)
    base_limit = max(inherited, int(math.ceil(ideal * TEXT_LAYER_GAP_FACTOR)))
    required = protected_text_layer_ids(source, target_layers)
    for candidate in range(base_limit, span + 1):
        if _feasible_with_gap(source, target_layers, required, candidate):
            return candidate
    raise RuntimeError("required text-layer anchors cannot form an exact nested map")


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
    except (ValueError, RuntimeError):
        return False
    if not required.issubset(set(selected)):
        return False
    if gap_limit is not None:
        max_gap = max((b - a for a, b in zip(selected, selected[1:])), default=0)
        if max_gap > gap_limit:
            return False
    return True


def select_salient_layer_ids(
    scores: Mapping[object, object] | Sequence[object],
    source_layer_ids: Sequence[int],
    target_layers: int,
    *,
    protect_boundaries: bool = True,
    protect_tail: int = 0,
) -> tuple[int, ...]:
    """Select saliency-maximizing text layers under the grounded contract."""

    source = _validated_source(source_layer_ids, target_layers)
    if target_layers == len(source):
        return source
    if protect_tail < 0:
        raise ValueError("protect_tail must not be negative")
    if not protect_boundaries or protect_tail == 0:
        return core.select_salient_layer_ids(
            scores,
            source,
            target_layers,
            protect_boundaries=protect_boundaries,
            protect_tail=protect_tail,
        )

    score_by_id = common._score_sequence(scores)
    missing = [layer_id for layer_id in source if layer_id not in score_by_id]
    if missing:
        raise ValueError(f"missing saliency scores for layers: {missing}")
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
        result = common._coverage_constrained_selection(
            source, score_by_id, target_layers, required, gap_limit
        )
    if not text_layer_selection_is_safe(source, result):
        raise RuntimeError(f"grounding-safe text layer selection failed: {result}")
    return result


__all__ = [
    "protected_text_layer_ids",
    "select_salient_layer_ids",
    "text_layer_gap_limit",
    "text_layer_selection_is_safe",
]
