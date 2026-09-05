#!/usr/bin/env python3
"""Evaluate free-running GUI-Owl syntax, action identity, and screenshot grounding."""
from __future__ import annotations

import argparse
import collections
import json
import random
import statistics
import sys
from pathlib import Path
from typing import Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from android_drlqn.gui_owl_metrics import (  # noqa: E402
    assistant_content_text,
    compare_gui_outputs,
)
from project_gui_owl_train import parse_gui_owl_output  # noqa: E402


def _to_device(batch: Mapping[str, object], device: str) -> dict[str, object]:
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in batch.items()
    }


def _trim(ids: Sequence[int], eos_ids: set[int], pad_id: int | None) -> list[int]:
    result: list[int] = []
    for raw in ids:
        token = int(raw)
        if token in eos_ids or (pad_id is not None and token == pad_id):
            break
        result.append(token)
    return result


def _load_gui_records(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, Mapping):
            raise ValueError(f"{path}:{line_number}: record must be an object")
        messages = value.get("messages")
        if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
            raise ValueError(f"{path}:{line_number}: messages must be an array")
        if len(messages) < 2 or not isinstance(messages[-1], Mapping):
            raise ValueError(f"{path}:{line_number}: assistant target is missing")
        if str(messages[-1].get("role") or "") != "assistant":
            raise ValueError(f"{path}:{line_number}: final message must be assistant")
        records.append(dict(value))
    if not records:
        raise ValueError(f"GUI policy corpus is empty: {path}")
    return records


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    try:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as exc:
        raise RuntimeError("install the training extra: python -m pip install -e '.[train]'") from exc

    # Resolve by architecture rather than hardcoding Qwen3-VL. For a qwen3_vl
    # checkpoint this is the same class; it also admits other multimodal
    # backbones (e.g. qwen3_5) so a candidate can be scored on this corpus
    # under the identical prompt and metric.
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_dir():
        # A Hugging Face repo id is not a local path but is a valid source.
        if "/" not in str(args.checkpoint) or str(args.checkpoint).startswith("."):
            raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    corpus = Path(args.corpus)
    records = _load_gui_records(corpus)

    rng = random.Random(args.seed)
    rng.shuffle(records)
    holdout_count = max(1, int(round(len(records) * args.holdout_fraction)))
    holdout_count = min(holdout_count, len(records))
    selected = records[:holdout_count][: args.max_records]
    if not selected:
        raise ValueError("no GUI policy holdout records selected")

    device = args.device
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    processor = AutoProcessor.from_pretrained(str(checkpoint))
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            str(checkpoint), dtype=dtype, attn_implementation="sdpa", low_cpu_mem_usage=True
        )
    except TypeError:
        model = AutoModelForImageTextToText.from_pretrained(
            str(checkpoint), torch_dtype=dtype, attn_implementation="sdpa", low_cpu_mem_usage=True
        )
    model.to(device)
    model.eval()

    eos = processor.tokenizer.eos_token_id
    eos_ids = {int(eos)} if isinstance(eos, int) else {int(value) for value in (eos or ())}
    pad_id = processor.tokenizer.pad_token_id

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    first_eos = empty = valid = 0
    action_records = action_matches = 0
    lengths: list[int] = []
    coordinate_errors: list[float] = []
    coordinate_within_50 = 0
    emitted_actions: set[str] = set()
    generated_x: list[float] = []
    # gui_action_name_match_rate has a majority-class floor: 65.3% of the
    # episode holdout is mobile.click, so a model with one action scores 0.653
    # against a 0.65 gate. Per-class recall is what separates coverage from
    # that floor, and the floor itself is reported so the pass can be read.
    action_class_total: collections.Counter[str] = collections.Counter()
    action_class_hit: collections.Counter[str] = collections.Counter()
    # The median annotated box is 224x64 on the 0..1000 scale, so a 50px radius
    # fails clicks that would land on the element. Point-in-box is the metric
    # the annotation actually supports.
    box_records = point_in_box = 0

    with output.open("w", encoding="utf-8") as handle, torch.no_grad():
        for index, record in enumerate(selected):
            messages = list(record["messages"])
            prompt = processor.apply_chat_template(
                messages[:-1],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            prompt = _to_device(prompt, device)
            prompt.pop("labels", None)
            sequence = model.generate(
                **prompt,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
            )
            prompt_tokens = int(prompt["input_ids"].shape[1])
            generated = [int(value) for value in sequence[0, prompt_tokens:].tolist()]
            is_first_eos = bool(generated and generated[0] in eos_ids)
            if is_first_eos:
                first_eos += 1
            trimmed = _trim(generated, eos_ids, pad_id)
            lengths.append(len(trimmed))
            text = processor.tokenizer.decode(
                trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()
            if not text:
                empty += 1
            parsed = bool(text and parse_gui_owl_output(text) is not None)
            if parsed:
                valid += 1

            target_content = messages[-1].get("content") if isinstance(messages[-1], Mapping) else None
            target_text = assistant_content_text(target_content)
            correctness = compare_gui_outputs(text, target_text)
            if correctness.generated_action:
                emitted_actions.add(correctness.generated_action)
            if correctness.target_action:
                action_records += 1
                action_class_total[correctness.target_action] += 1
                if correctness.action_match:
                    action_matches += 1
                    action_class_hit[correctness.target_action] += 1
            bbox = record.get("target_bbox")
            in_box: bool | None = None
            if (
                isinstance(bbox, Sequence)
                and not isinstance(bbox, (str, bytes))
                and len(bbox) == 4
            ):
                box_records += 1
                point = correctness.generated_coordinate
                if point is not None:
                    x0, y0, x1, y1 = (float(value) for value in bbox)
                    in_box = x0 <= float(point[0]) <= x1 and y0 <= float(point[1]) <= y1
                else:
                    in_box = False
                if in_box:
                    point_in_box += 1
            if correctness.coordinate_error is not None:
                coordinate_errors.append(float(correctness.coordinate_error))
                if correctness.coordinate_error <= 50.0:
                    coordinate_within_50 += 1
            if correctness.generated_coordinate is not None:
                generated_x.append(float(correctness.generated_coordinate[0]))

            row = {
                "index": index,
                "id": record.get("id"),
                "episode_id": record.get("episode_id"),
                "step_index": record.get("step_index"),
                "generated_tokens": len(trimmed),
                "first_token_eos": is_first_eos,
                "empty": not bool(text),
                "gui_parse_valid": parsed,
                "target_action": correctness.target_action,
                "generated_action": correctness.generated_action,
                "action_name_match": correctness.action_match,
                "target_coordinate": correctness.target_coordinate,
                "generated_coordinate": correctness.generated_coordinate,
                "coordinate_error": correctness.coordinate_error,
                "target_bbox": list(bbox) if in_box is not None else None,
                "point_in_box": in_box,
                "has_action_prefix": text.startswith("Action:"),
                "has_tool_call_open": "<tool_call>" in text,
                "has_tool_call_close": "</tool_call>" in text,
                "generation": text,
                "target": target_text,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    total = len(selected)
    coordinate_records = len(coordinate_errors)
    summary = {
        "checkpoint": str(checkpoint),
        "corpus": str(corpus),
        "records": total,
        "first_token_eos_rate": first_eos / total,
        "empty_generation_rate": empty / total,
        "median_generated_tokens": float(statistics.median(lengths)),
        "gui_parse_rate": valid / total,
        "gui_action_records": action_records,
        "gui_action_name_match_rate": action_matches / action_records if action_records else 0.0,
        "gui_action_macro_recall": (
            sum(action_class_hit[name] / count for name, count in action_class_total.items())
            / len(action_class_total)
            if action_class_total
            else 0.0
        ),
        "gui_action_class_recall": {
            name: action_class_hit[name] / count
            for name, count in sorted(action_class_total.items())
        },
        "gui_action_majority_class_rate": (
            max(action_class_total.values()) / action_records if action_records else 0.0
        ),
        "gui_action_target_types": len(action_class_total),
        "gui_distinct_action_types_emitted": len(emitted_actions),
        "gui_emitted_action_types": sorted(emitted_actions),
        "gui_coordinate_records": coordinate_records,
        "gui_median_coordinate_error": (
            float(statistics.median(coordinate_errors)) if coordinate_errors else 0.0
        ),
        "gui_coordinate_within_50_rate": (
            coordinate_within_50 / coordinate_records if coordinate_records else 0.0
        ),
        "gui_box_records": box_records,
        "gui_point_in_box_rate": point_in_box / box_records if box_records else 0.0,
        "gui_generated_x_min": min(generated_x) if generated_x else None,
        "gui_generated_x_max": max(generated_x) if generated_x else None,
        "output": str(output),
    }
    summary_path = output.with_suffix(output.suffix + ".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--corpus", default="artifacts/training/gui-policy-unigui-openmobile.jsonl"
    )
    parser.add_argument("--output", default="artifacts/diagnostics/gui-policy-generations.jsonl")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-records", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--holdout-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-action-name-match-rate", type=float)
    parser.add_argument("--max-median-coordinate-error", type=float)
    args = parser.parse_args(argv)
    if args.max_records < 1 or args.max_new_tokens < 1:
        parser.error("record/token limits must be positive")
    if not 0 < args.holdout_fraction <= 1:
        parser.error("--holdout-fraction must be in (0, 1]")
    if args.min_action_name_match_rate is not None and not 0 <= args.min_action_name_match_rate <= 1:
        parser.error("--min-action-name-match-rate must be in [0, 1]")
    if args.max_median_coordinate_error is not None and args.max_median_coordinate_error < 0:
        parser.error("--max-median-coordinate-error must be non-negative")

    summary = evaluate(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    failures: list[str] = []
    if (
        args.min_action_name_match_rate is not None
        and float(summary["gui_action_name_match_rate"]) < args.min_action_name_match_rate
    ):
        failures.append(
            f"action-name match {summary['gui_action_name_match_rate']:.4f} < "
            f"{args.min_action_name_match_rate:.4f}"
        )
    if args.max_median_coordinate_error is not None:
        if int(summary["gui_coordinate_records"]) <= 0:
            failures.append("coordinate error unmeasured: no coordinate-bearing targets")
        elif float(summary["gui_median_coordinate_error"]) > args.max_median_coordinate_error:
            failures.append(
                f"median coordinate error {summary['gui_median_coordinate_error']:.2f} > "
                f"{args.max_median_coordinate_error:.2f}"
            )
    if failures:
        raise SystemExit("GUI correctness gate failed: " + "; ".join(failures))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
