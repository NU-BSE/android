#!/usr/bin/env python3
"""Normalize GUI-Owl calibration for GQA and Qwen3-VL deepstack pruning.

Two structural invariants need preservation before the 0.5B width/vision build:

* text grouped-query attention keeps each retained KV head with its original
  query-head group; and
* the three Qwen3-VL deepstack feature taps keep their corresponding source
  vision blocks and remain aligned with `deepstack_merger_list` ordering.

The source GUI-Owl vision tower taps blocks 5/11/17. The 12-block compact tower
places those same retained blocks at compact positions 2/5/8, leaving capacity
both before, between, and after the deepstack taps. Other retained blocks are
chosen by calibration saliency within those intervals.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

from android_drlqn.gui_owl_student import (
    derive_width_selections,
    grouped_query_kv_selection,
    load_student_plan,
    plan_student_specs,
)


SOURCE_DEEPSTACK_VISUAL_INDEXES = (5, 11, 17)
TARGET_DEEPSTACK_VISUAL_INDEXES = (2, 5, 8)
TARGET_VISION_DEPTH = 12


def _floats(value: object, *, name: str) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be an array")
    result = [float(item) for item in value]
    if not result:
        raise ValueError(f"{name} cannot be empty")
    return result


def _top_scored_indices(scores: Sequence[float], candidates: Sequence[int], count: int) -> tuple[int, ...]:
    if count < 0 or count > len(candidates):
        raise ValueError("requested interval retention does not fit source vision blocks")
    ranked = sorted(candidates, key=lambda index: (-float(scores[index]), index))[:count]
    return tuple(sorted(ranked))


def deepstack_aligned_vision_selection(
    layer_scores: Sequence[object],
    *,
    source_deepstack: Sequence[int] = SOURCE_DEEPSTACK_VISUAL_INDEXES,
    target_deepstack: Sequence[int] = TARGET_DEEPSTACK_VISUAL_INDEXES,
    target_depth: int = TARGET_VISION_DEPTH,
) -> tuple[int, ...]:
    """Select vision blocks while preserving source deepstack tap semantics.

    The retained source block at ``source_deepstack[i]`` is placed at compact
    position ``target_deepstack[i]``. Saliency fills the intervals around those
    mandatory taps. This preserves the order of `deepstack_merger_list` while
    still allowing activation-based pruning of the other blocks.
    """

    scores = [float(value) for value in layer_scores]
    source = tuple(int(value) for value in source_deepstack)
    target = tuple(int(value) for value in target_deepstack)
    if not scores:
        raise ValueError("vision layer scores cannot be empty")
    if len(source) != len(target) or not source:
        raise ValueError("source/target deepstack index lists must have equal non-zero length")
    if tuple(sorted(source)) != source or tuple(sorted(target)) != target:
        raise ValueError("deepstack indexes must be strictly monotonic")
    if len(set(source)) != len(source) or len(set(target)) != len(target):
        raise ValueError("deepstack indexes must be unique")
    if source[-1] >= len(scores) or target[-1] >= target_depth:
        raise ValueError("deepstack index falls outside source/target vision depth")
    if target_depth > len(scores):
        raise ValueError("target vision depth cannot exceed source depth")

    boundaries_source = (-1, *source, len(scores))
    boundaries_target = (-1, *target, target_depth)
    selected: list[int] = []
    for interval in range(len(boundaries_source) - 1):
        source_start = boundaries_source[interval] + 1
        source_end = boundaries_source[interval + 1]
        target_start = boundaries_target[interval] + 1
        target_end = boundaries_target[interval + 1]
        count = target_end - target_start
        candidates = tuple(range(source_start, source_end))
        selected.extend(_top_scored_indices(scores, candidates, count))
        if interval < len(source):
            selected.append(source[interval])

    result = tuple(selected)
    if len(result) != target_depth or tuple(sorted(result)) != result:
        raise RuntimeError("deepstack-aligned vision selection produced an invalid layer map")
    for source_index, target_index in zip(source, target):
        if result[target_index] != source_index:
            raise RuntimeError("deepstack source block did not land at its target compact position")
    return result


def normalize_manifest(
    path: str | Path, *, plan: Mapping[str, object] | None = None
) -> dict[str, object]:
    """Rewrite ``path`` with GQA-consistent, deepstack-aligned, plan-complete selections.

    ``plan`` lists the student architectures whose width selections the
    manifest must carry; every narrowed axis of every student is derived from
    the recorded raw scores, so a manifest calibrated before a width target
    existed still serves it without re-running the teacher.
    """

    target = Path(path)
    value = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("calibration manifest must be an object")
    manifest = dict(value)
    text = manifest.get("text")
    vision = manifest.get("vision")
    recommended = manifest.get("recommended")
    if not isinstance(text, Mapping) or not isinstance(recommended, Mapping):
        raise ValueError("calibration manifest lacks text/recommended sections")

    source_q_heads = int(text.get("attention_heads") or 0)
    source_kv_heads = int(text.get("key_value_heads") or 0)
    q_scores_raw = text.get("q_head_scores")
    kv_scores_raw = text.get("kv_head_scores")
    if not isinstance(q_scores_raw, Mapping) or not isinstance(kv_scores_raw, Mapping):
        raise ValueError("calibration manifest lacks per-layer query/KV head scores")

    target_q_heads = 8
    target_kv_heads = 4
    if source_q_heads < target_q_heads or source_kv_heads < target_kv_heads:
        raise ValueError("source calibration has too few attention heads for the 0.5B target")

    q_output: dict[str, list[int]] = {}
    kv_output: dict[str, list[int]] = {}
    layer_ids = sorted(
        {str(key) for key in q_scores_raw} & {str(key) for key in kv_scores_raw},
        key=int,
    )
    if not layer_ids:
        raise ValueError("calibration manifest has no aligned query/KV layer scores")
    for layer_id in layer_ids:
        q_scores = _floats(q_scores_raw[layer_id], name=f"q_head_scores[{layer_id}]")
        kv_scores = _floats(kv_scores_raw[layer_id], name=f"kv_head_scores[{layer_id}]")
        if len(q_scores) != source_q_heads or len(kv_scores) != source_kv_heads:
            raise ValueError(f"head-score shape mismatch for layer {layer_id}")
        selected_q, selected_kv = grouped_query_kv_selection(
            q_scores,
            kv_scores,
            target_q_heads=target_q_heads,
            target_kv_heads=target_kv_heads,
        )
        q_output[layer_id] = list(selected_q)
        kv_output[layer_id] = list(selected_kv)

    normalized_recommended = dict(recommended)
    normalized_recommended[f"text_q_heads_{target_q_heads}"] = q_output
    normalized_recommended[f"text_kv_heads_{target_kv_heads}"] = kv_output

    deepstack_selection: tuple[int, ...] | None = None
    if isinstance(vision, Mapping) and bool(vision.get("ready")):
        vision_scores = _floats(vision.get("layer_scores"), name="vision.layer_scores")
        deepstack_selection = deepstack_aligned_vision_selection(vision_scores)
        vision_layers = normalized_recommended.get("vision_layers_by_depth")
        normalized_layers = dict(vision_layers) if isinstance(vision_layers, Mapping) else {}
        normalized_layers[str(TARGET_VISION_DEPTH)] = list(deepstack_selection)
        normalized_recommended["vision_layers_by_depth"] = normalized_layers

    # Width selections for every student the plan declares. derive_width_selections
    # ranks whole GQA groups the same way as above, so the two agree wherever they
    # overlap, and it only emits the axes a student actually narrows.
    derived_sources: dict[str, str] = {}
    for spec in plan_student_specs(plan):
        try:
            derived = derive_width_selections(manifest, spec)
        except ValueError:
            # A manifest without vision evidence cannot serve a vision-narrowing
            # student; the builder refuses that build explicitly.
            continue
        source = derived.pop("text_mlp_score_source", None)
        derived.pop("vision_mlp_score_source", None)
        if isinstance(source, str):
            derived_sources[spec.name] = source
        normalized_recommended.update(derived)

    manifest["recommended"] = normalized_recommended
    if derived_sources:
        manifest["text_mlp_score_source"] = derived_sources
    manifest["gqa_group_consistent"] = True
    manifest["gqa_source_group_size"] = source_q_heads // source_kv_heads
    manifest["gqa_target_group_size"] = target_q_heads // target_kv_heads
    manifest["gqa_selection_method"] = (
        "rank_whole_kv_groups_by_kv_plus_mean_query_activation"
    )
    manifest["source_deepstack_visual_indexes"] = list(SOURCE_DEEPSTACK_VISUAL_INDEXES)
    manifest["target_deepstack_visual_indexes"] = list(TARGET_DEEPSTACK_VISUAL_INDEXES)
    manifest["deepstack_alignment_preserved"] = deepstack_selection is not None
    if deepstack_selection is not None:
        manifest["recommended_vision_original_layer_ids_12"] = list(deepstack_selection)
    target.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "manifest",
        nargs="?",
        default="artifacts/calibration/gui-owl-compression.json",
    )
    parser.add_argument("--config", default="config/gui_owl_students.json")
    args = parser.parse_args(argv)
    plan = load_student_plan(args.config) if Path(args.config).is_file() else None
    print(json.dumps(normalize_manifest(args.manifest, plan=plan), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
