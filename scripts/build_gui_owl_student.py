#!/usr/bin/env python3
"""Build an initialized 0.5B, 1B, or 1.5B GUI-Owl Qwen3-VL student."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Mapping

from android_drlqn.gui_owl_student import (
    build_student_config_dict,
    evenly_spaced_layer_map,
    load_student_plan,
    prefix_copy_shape,
    relative_parameter_error,
    spec_from_plan,
    teacher_key_for_student,
)


def _student_entry(plan: Mapping[str, object], name: str) -> dict[str, object]:
    students = plan.get("students")
    if not isinstance(students, Mapping):
        raise ValueError("student plan must define students")
    raw = students.get(name)
    if not isinstance(raw, Mapping):
        raise ValueError(f"student plan does not define {name}")
    return dict(raw)


def _torch_dtype(torch_module, name: str):
    values = {
        "float32": torch_module.float32,
        "float16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
    }
    try:
        return values[name]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype: {name}") from exc


def _copy_initial_weights(student, teacher, layer_map, torch_module) -> dict[str, int]:
    """Copy exact tensors or safe prefixes from mapped teacher tensors."""

    teacher_state = teacher.state_dict()
    copied_tensors = copied_elements = total_elements = 0
    with torch_module.no_grad():
        for key, destination in student.state_dict().items():
            total_elements += destination.numel()
            teacher_key = teacher_key_for_student(key, layer_map)
            source = teacher_state.get(teacher_key)
            if source is None:
                continue
            if tuple(source.shape) == tuple(destination.shape):
                destination.copy_(source.to(device=destination.device, dtype=destination.dtype))
                copied_tensors += 1
                copied_elements += destination.numel()
                continue
            slices = prefix_copy_shape(destination.shape, source.shape)
            if slices is None:
                continue
            destination.copy_(source[slices].to(device=destination.device, dtype=destination.dtype))
            copied_tensors += 1
            copied_elements += destination.numel()
    return {
        "copied_tensors": copied_tensors,
        "copied_elements": copied_elements,
        "total_elements": total_elements,
    }


def build(args: argparse.Namespace) -> dict[str, object]:
    try:
        import torch
        from transformers import AutoConfig, AutoProcessor, Qwen3VLConfig, Qwen3VLForConditionalGeneration
    except ImportError as exc:
        raise RuntimeError("install the training extra: python -m pip install -e '.[train]'") from exc

    plan = load_student_plan(args.config)
    spec = spec_from_plan(plan, args.student)
    entry = _student_entry(plan, args.student)
    teacher_model = args.teacher_model or str(plan.get("initialization_teacher") or "")
    if not teacher_model:
        raise ValueError("initialization teacher is missing")
    output = Path(args.output or str(entry.get("initialized") or f"artifacts/students/gui-owl-{args.student}-init"))
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"student output is not empty: {output}; pass --overwrite to replace")

    dtype = _torch_dtype(torch, args.dtype)
    teacher_config = AutoConfig.from_pretrained(teacher_model)
    teacher_config_dict = teacher_config.to_dict()
    teacher_layers = int(teacher_config_dict["text_config"]["num_hidden_layers"])
    layer_map = evenly_spaced_layer_map(teacher_layers, spec.text_layers)
    student_config_dict = build_student_config_dict(teacher_config_dict, spec)
    student_config = Qwen3VLConfig.from_dict(student_config_dict)

    previous_default = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        student = Qwen3VLForConditionalGeneration(student_config)
    finally:
        torch.set_default_dtype(previous_default)

    try:
        teacher = Qwen3VLForConditionalGeneration.from_pretrained(
            teacher_model,
            dtype=dtype,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
    except TypeError:
        teacher = Qwen3VLForConditionalGeneration.from_pretrained(
            teacher_model,
            torch_dtype=dtype,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
    teacher.eval()

    copy_stats = _copy_initial_weights(student, teacher, layer_map, torch)
    del teacher

    actual_parameters = sum(parameter.numel() for parameter in student.parameters())
    parameter_error = relative_parameter_error(actual_parameters, spec.target_parameters)
    if parameter_error > spec.max_relative_parameter_error and not args.allow_parameter_miss:
        raise RuntimeError(
            f"{args.student} architecture has {actual_parameters:,} parameters, "
            f"{parameter_error:.1%} from target {spec.target_parameters:,}; "
            "adjust config or pass --allow-parameter-miss for an exploratory build"
        )

    output.mkdir(parents=True, exist_ok=True)
    student.config.use_cache = False
    student.save_pretrained(output, safe_serialization=True)
    processor = AutoProcessor.from_pretrained(teacher_model)
    processor.save_pretrained(output)

    manifest = {
        "student": args.student,
        "initialization_teacher": teacher_model,
        "target_parameters": spec.target_parameters,
        "actual_parameters": actual_parameters,
        "relative_parameter_error": parameter_error,
        "architecture": asdict(spec),
        "teacher_text_layers": teacher_layers,
        "teacher_layer_map": list(layer_map),
        "prefix_initialization": copy_stats,
        "dtype": args.dtype,
        "full_student_checkpoint": True,
        "credentials_written": False,
    }
    (output / "student-build-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", required=True, choices=("0.5b", "1b", "1.5b"))
    parser.add_argument("--config", default="config/gui_owl_students.json")
    parser.add_argument("--teacher-model")
    parser.add_argument("--output")
    parser.add_argument("--dtype", default="bfloat16", choices=("float32", "float16", "bfloat16"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-parameter-miss", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(build(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
