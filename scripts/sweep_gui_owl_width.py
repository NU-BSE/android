#!/usr/bin/env python3
"""Measure what structured width pruning does to GUI grounding, without training.

`ablate_gui_owl_text_layers.py` showed that removing text layers deletes visual
grounding before any recovery runs. This is its width counterpart: every
variant is built from the same source checkpoint with the production builder
(`build_gui_owl_student.py`), then evaluated free-running under the shipping
contract prompt with the same metric the acceptance gates use.

Variants name the axes they narrow, e.g.

    --variant ffn3072=text_intermediate_size:3072
    --variant heads12=text_attention_heads:12,text_key_value_heads:6
    --variant mixed=text_intermediate_size:4096,text_attention_heads:12,text_key_value_heads:6

Unnamed fields keep the source architecture, so a variant that lists no depth
keeps every text layer. `--selection random` replaces the calibration scores
with random ones before deriving the selections, which is the control that
tells whether the saliency metric is doing anything for grounding.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import random
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from android_drlqn.gui_owl_student import (  # noqa: E402
    StudentSpec,
    load_calibration_manifest,
    load_student_plan,
    spec_from_plan,
)

_INT_FIELDS = {
    "text_layers",
    "text_hidden_size",
    "text_intermediate_size",
    "text_attention_heads",
    "text_key_value_heads",
    "vision_depth",
    "vision_hidden_size",
    "vision_intermediate_size",
    "vision_heads",
}


def parse_variant(raw: str) -> tuple[str, dict[str, int]]:
    name, separator, spec = raw.partition("=")
    if not separator:
        raise ValueError(f"variant {raw!r} must look like name=field:value,...")
    overrides: dict[str, int] = {}
    for item in spec.split(","):
        if not item.strip():
            continue
        field, colon, value = item.partition(":")
        field = field.strip()
        if not colon or field not in _INT_FIELDS:
            raise ValueError(f"variant {name}: unknown field {field!r}")
        overrides[field] = int(value)
    if not overrides:
        raise ValueError(f"variant {name} changes nothing")
    return name.strip(), overrides


def source_architecture(source_config: Mapping[str, object]) -> dict[str, int]:
    text = source_config["text_config"]
    vision = source_config["vision_config"]
    return {
        "text_layers": int(text["num_hidden_layers"]),
        "text_hidden_size": int(text["hidden_size"]),
        "text_intermediate_size": int(text["intermediate_size"]),
        "text_attention_heads": int(text["num_attention_heads"]),
        "text_key_value_heads": int(text["num_key_value_heads"]),
        "vision_depth": int(vision["depth"]),
        "vision_hidden_size": int(vision["hidden_size"]),
        "vision_intermediate_size": int(vision["intermediate_size"]),
        "vision_heads": int(vision["num_heads"]),
    }


def variant_spec(
    base: StudentSpec, source: Mapping[str, int], overrides: Mapping[str, int]
) -> StudentSpec:
    """A spec that keeps the source architecture except for the named axes."""

    merged = {**asdict(base), **dict(source), **dict(overrides)}
    merged["deepstack_visual_indexes"] = tuple(int(v) for v in merged["deepstack_visual_indexes"])
    return StudentSpec(**merged)


def randomized_scores(calibration: Mapping[str, object], seed: int) -> dict[str, object]:
    """Replace every width score vector with random values; layer scores stay."""

    rng = random.Random(seed)
    result = copy.deepcopy(dict(calibration))
    for section_name in ("text", "vision"):
        section = result.get(section_name)
        if not isinstance(section, dict):
            continue
        for field, value in list(section.items()):
            if field == "layer_scores" or not field.endswith("_scores"):
                continue
            if isinstance(value, list):
                section[field] = [rng.random() for _ in value]
            elif isinstance(value, dict):
                section[field] = {
                    key: [rng.random() for _ in item] for key, item in value.items()
                }
    recommended = result.get("recommended")
    if isinstance(recommended, dict):
        # Recommended entries are cached rankings of the real scores; drop them so
        # the builder derives from the randomized ones.
        result["recommended"] = {
            key: item
            for key, item in recommended.items()
            if key.endswith("_by_depth")
        }
    return result


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return (0.0, 0.0)
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _summarize(name: str, summary: Mapping[str, object], parameters: int | None) -> dict[str, object]:
    coordinate_records = int(summary.get("gui_coordinate_records") or 0)
    within = float(summary.get("gui_coordinate_within_50_rate") or 0.0)
    hits = int(round(within * coordinate_records))
    low, high = wilson_interval(hits, coordinate_records)
    return {
        "variant": name,
        "parameters": parameters,
        "records": int(summary.get("records") or 0),
        "gui_parse_rate": float(summary.get("gui_parse_rate") or 0.0),
        "gui_action_name_match_rate": float(summary.get("gui_action_name_match_rate") or 0.0),
        "gui_median_coordinate_error": float(summary.get("gui_median_coordinate_error") or 0.0),
        "gui_coordinate_within_50_rate": within,
        "gui_coordinate_within_50_ci95": [low, high],
        "gui_coordinate_records": coordinate_records,
        "first_token_eos_rate": float(summary.get("first_token_eos_rate") or 0.0),
        "empty_generation_rate": float(summary.get("empty_generation_rate") or 0.0),
    }


def _evaluate(checkpoint: Path, args: argparse.Namespace, output: Path) -> dict[str, object]:
    from scripts import evaluate_gui_owl_policy_production as production

    namespace = argparse.Namespace(
        checkpoint=str(checkpoint),
        corpus=args.corpus,
        output=str(output),
        device=args.device,
        max_records=args.max_records,
        max_new_tokens=args.max_new_tokens,
        holdout_fraction=args.holdout_fraction,
        seed=args.seed,
    )
    return production.base.evaluate(namespace)


def _release_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # pragma: no cover - best effort
        pass


def sweep(args: argparse.Namespace) -> dict[str, object]:
    from transformers import AutoConfig

    from scripts import build_gui_owl_student as builder

    plan = load_student_plan(args.config)
    base_spec = spec_from_plan(plan, args.student)
    calibration_path = Path(args.calibration_manifest)
    calibration = load_calibration_manifest(calibration_path)
    source_config = AutoConfig.from_pretrained(args.source_model).to_dict()
    source = source_architecture(source_config)
    variants = [parse_variant(raw) for raw in args.variant]

    work = Path(args.work_root)
    work.mkdir(parents=True, exist_ok=True)
    diagnostics = Path(args.output).parent
    diagnostics.mkdir(parents=True, exist_ok=True)

    effective_calibration = calibration_path
    if args.selection == "random":
        effective_calibration = work / f"calibration-random-{args.selection_seed}.json"
        _write_json(effective_calibration, randomized_scores(calibration, args.selection_seed))

    rows: list[dict[str, object]] = []
    if args.baseline:
        started = time.time()
        summary = _evaluate(
            Path(args.source_model), args, diagnostics / f"{args.tag}-baseline.jsonl"
        )
        row = _summarize("baseline", summary, None)
        row["seconds"] = time.time() - started
        row["architecture"] = dict(source)
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        _release_memory()

    for name, overrides in variants:
        started = time.time()
        spec = variant_spec(base_spec, source, overrides)
        variant_plan = copy.deepcopy(plan)
        students = variant_plan.setdefault("students", {})
        entry = dict(students.get(args.student) or {})
        entry.update({key: value for key, value in asdict(spec).items() if key != "name"})
        entry["deepstack_visual_indexes"] = list(spec.deepstack_visual_indexes)
        students[args.student] = entry
        plan_path = work / f"plan-{name}.json"
        _write_json(plan_path, variant_plan)

        checkpoint = work / name
        build_args = argparse.Namespace(
            student=args.student,
            config=str(plan_path),
            source_model=args.source_model,
            target_text_layers=spec.text_layers,
            calibration_manifest=str(effective_calibration),
            output=str(checkpoint),
            dtype=args.build_dtype,
            overwrite=True,
            allow_parameter_miss=True,
        )
        manifest = builder.build(build_args)
        _release_memory()
        parameters = int(manifest["actual_parameters"])
        print(
            f"built {name}: {parameters:,} parameters, axes {manifest['width_pruned_axes']}",
            flush=True,
        )

        summary = _evaluate(checkpoint, args, diagnostics / f"{args.tag}-{name}.jsonl")
        row = _summarize(name, summary, parameters)
        row["seconds"] = time.time() - started
        row["architecture"] = {**source, **overrides}
        row["relative_parameter_error"] = float(manifest["relative_parameter_error"])
        row["width_pruned_axes"] = manifest["width_pruned_axes"]
        row["width_selection_sources"] = manifest["width_selection_sources"]
        row["text_mlp_score_source"] = manifest.get("text_mlp_score_source")
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        _release_memory()
        if not args.keep_checkpoints:
            shutil.rmtree(checkpoint, ignore_errors=True)

        result = {
            "source_model": args.source_model,
            "calibration_manifest": str(effective_calibration),
            "selection": args.selection,
            "selection_seed": args.selection_seed if args.selection == "random" else None,
            "corpus": args.corpus,
            "records": args.max_records,
            "holdout_fraction": args.holdout_fraction,
            "seed": args.seed,
            "max_new_tokens": args.max_new_tokens,
            "rows": rows,
        }
        _write_json(Path(args.output), result)
    return {"rows": rows, "output": args.output}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", required=True)
    parser.add_argument("--variant", action="append", default=[])
    parser.add_argument("--student", default="1.5b")
    parser.add_argument("--config", default="config/gui_owl_students.json")
    parser.add_argument(
        "--calibration-manifest", default="artifacts/calibration/gui-owl-compression.json"
    )
    parser.add_argument("--selection", choices=("saliency", "random"), default="saliency")
    parser.add_argument("--selection-seed", type=int, default=0)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--corpus", default="artifacts/training/gui-policy-unigui-openmobile.jsonl")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-records", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--build-dtype", default="bfloat16", choices=("float32", "float16", "bfloat16"))
    parser.add_argument("--work-root", default="artifacts/students/width-sweep")
    parser.add_argument("--tag", default="width")
    parser.add_argument("--output", default="artifacts/diagnostics/width/sweep.json")
    parser.add_argument("--keep-checkpoints", action="store_true")
    args = parser.parse_args(argv)
    if not args.variant and not args.baseline:
        parser.error("pass at least one --variant or --baseline")
    if args.max_records < 1 or args.max_new_tokens < 1:
        parser.error("record/token limits must be positive")
    if not 0 < args.holdout_fraction <= 1:
        parser.error("--holdout-fraction must be in (0, 1]")
    print(json.dumps(sweep(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
