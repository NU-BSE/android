#!/usr/bin/env python3
"""Normalize GUI-Owl calibration selections for grouped-query attention.

Qwen3-VL text attention repeats each KV head across a fixed group of query
heads. Selecting query and KV heads independently would change those semantic
pairings after pruning. This post-processing step ranks whole GQA groups, then
rewrites the 0.5B query/KV selections so every retained KV head keeps all of its
corresponding query heads.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence


def _floats(value: object, *, name: str) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be an array")
    result = [float(item) for item in value]
    if not result:
        raise ValueError(f"{name} cannot be empty")
    return result


def grouped_query_kv_selection(
    q_scores: Sequence[object],
    kv_scores: Sequence[object],
    *,
    target_q_heads: int,
    target_kv_heads: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    q = [float(value) for value in q_scores]
    kv = [float(value) for value in kv_scores]
    if not q or not kv:
        raise ValueError("query/KV head scores cannot be empty")
    if len(q) % len(kv):
        raise ValueError("source query-head count must be divisible by KV-head count")
    if target_q_heads % target_kv_heads:
        raise ValueError("target query-head count must be divisible by target KV-head count")
    source_group = len(q) // len(kv)
    target_group = target_q_heads // target_kv_heads
    if source_group != target_group:
        raise ValueError(
            "structured GQA pruning currently requires the source and target query/KV group size to match"
        )
    if not 1 <= target_kv_heads <= len(kv):
        raise ValueError("target KV-head count must fit inside source KV heads")

    group_scores: list[tuple[float, int]] = []
    for kv_index, kv_score in enumerate(kv):
        q_group = q[kv_index * source_group : (kv_index + 1) * source_group]
        score = float(kv_score) + sum(q_group) / len(q_group)
        group_scores.append((score, kv_index))
    selected_kv = tuple(
        sorted(index for _, index in sorted(group_scores, key=lambda item: (-item[0], item[1]))[:target_kv_heads])
    )
    selected_q = tuple(
        head
        for kv_index in selected_kv
        for head in range(kv_index * source_group, (kv_index + 1) * source_group)
    )
    if len(selected_q) != target_q_heads:
        raise RuntimeError("grouped-query selection produced the wrong query-head count")
    return selected_q, selected_kv


def normalize_manifest(path: str | Path) -> dict[str, object]:
    target = Path(path)
    value = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("calibration manifest must be an object")
    manifest = dict(value)
    text = manifest.get("text")
    recommended = manifest.get("recommended")
    if not isinstance(text, Mapping) or not isinstance(recommended, Mapping):
        raise ValueError("calibration manifest lacks text/recommended sections")

    source_q_heads = int(text.get("attention_heads") or 0)
    source_kv_heads = int(text.get("key_value_heads") or 0)
    q_scores_raw = text.get("q_head_scores")
    kv_scores_raw = text.get("kv_head_scores")
    if not isinstance(q_scores_raw, Mapping) or not isinstance(kv_scores_raw, Mapping):
        raise ValueError("calibration manifest lacks per-layer query/KV head scores")

    # The v2 compact architecture currently targets 8 query / 4 KV heads.
    target_q_heads = 8
    target_kv_heads = 4
    if source_q_heads < target_q_heads or source_kv_heads < target_kv_heads:
        raise ValueError("source calibration has too few attention heads for the 0.5B target")

    q_output: dict[str, list[int]] = {}
    kv_output: dict[str, list[int]] = {}
    layer_ids = sorted({str(key) for key in q_scores_raw} & {str(key) for key in kv_scores_raw}, key=int)
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
    manifest["recommended"] = normalized_recommended
    manifest["gqa_group_consistent"] = True
    manifest["gqa_source_group_size"] = source_q_heads // source_kv_heads
    manifest["gqa_target_group_size"] = target_q_heads // target_kv_heads
    manifest["gqa_selection_method"] = "rank_whole_kv_groups_by_kv_plus_mean_query_activation"
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "manifest",
        nargs="?",
        default="artifacts/calibration/gui-owl-compression.json",
    )
    args = parser.parse_args(argv)
    print(json.dumps(normalize_manifest(args.manifest), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
