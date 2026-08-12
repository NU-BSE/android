#!/usr/bin/env python3
"""Distill trained GUI-Owl teacher behavior into one compact full student model."""
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from android_drlqn.gui_owl_student import (
    load_student_plan,
    resolve_peft_adapter_dir,
    spec_from_plan,
)


@dataclass(frozen=True)
class TrainingRecord:
    mode: str
    messages: list[object]
    loss_weight: float


def _load_jsonl_messages(path: Path, mode: str, limit: int | None) -> list[list[object]]:
    output: list[list[object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        messages = value.get("messages") if isinstance(value, Mapping) else None
        if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
            raise ValueError(f"{path}:{line_number}: messages must be an array")
        if len(messages) < 2 or not isinstance(messages[-1], Mapping):
            raise ValueError(f"{path}:{line_number}: assistant completion is missing")
        if str(messages[-1].get("role") or "") != "assistant":
            raise ValueError(f"{path}:{line_number}: final message must be assistant")
        output.append(list(messages))
        if limit is not None and len(output) >= limit:
            break
    if not output:
        raise ValueError(f"distillation corpus is empty: {path}")
    return output


def _corpus_records(plan: Mapping[str, object], max_records: int | None) -> list[TrainingRecord]:
    raw_corpora = plan.get("corpora")
    if not isinstance(raw_corpora, Mapping):
        raise ValueError("student plan must define corpora")
    loaded: dict[str, tuple[list[list[object]], float]] = {}
    for mode in ("gui_policy", "chat"):
        raw = raw_corpora.get(mode)
        if not isinstance(raw, Mapping):
            raise ValueError(f"student plan is missing {mode} corpus")
        path = Path(str(raw.get("path") or ""))
        if not path.is_file():
            raise FileNotFoundError(f"distillation corpus not found: {path}")
        weight = float(raw.get("weight", 1.0))
        if weight <= 0:
            raise ValueError("corpus weights must be positive")
        loaded[mode] = (_load_jsonl_messages(path, mode, max_records), weight)

    total = sum(len(items) for items, _ in loaded.values())
    modes = len(loaded)
    result: list[TrainingRecord] = []
    for mode, (items, configured_weight) in loaded.items():
        scale = configured_weight * total / (modes * len(items))
        result.extend(TrainingRecord(mode, messages, scale) for messages in items)
    return result


def _to_device(batch: Mapping[str, object], device: str) -> dict[str, object]:
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in batch.items()
    }


def _encode(processor, messages: Sequence[object], device: str) -> tuple[dict[str, object], int, int]:
    prompt = processor.apply_chat_template(
        list(messages[:-1]),
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    full = processor.apply_chat_template(
        list(messages),
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_tensors="pt",
    )
    prompt = _to_device(prompt, device)
    full = _to_device(full, device)
    prompt_length = min(int(prompt["input_ids"].shape[1]), int(full["input_ids"].shape[1]))
    completion_length = int(full["input_ids"].shape[1]) - prompt_length
    if completion_length < 1:
        raise ValueError("assistant completion produced no training tokens")
    full.pop("labels", None)
    return full, prompt_length, completion_length


def _torch_dtype(torch_module, device: str):
    return torch_module.bfloat16 if device.startswith("cuda") else torch_module.float32


def _load_build_manifest(student_dir: Path) -> dict[str, object]:
    path = student_dir / "student-build-manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"student build manifest not found: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("student build manifest must be an object")
    return dict(value)


def _teacher_adapters(plan: Mapping[str, object]) -> dict[str, Path]:
    value = plan.get("teacher_adapters")
    if not isinstance(value, Mapping):
        raise ValueError("student plan must define teacher_adapters")
    result: dict[str, Path] = {}
    for mode in ("gui_policy", "chat"):
        configured = Path(str(value.get(mode) or ""))
        result[mode] = resolve_peft_adapter_dir(configured, mode)
    return result


def _load_teacher(
    *,
    model_path: str,
    plan: Mapping[str, object],
    unified: bool,
    dtype,
    device: str,
    model_class,
    peft_model_class,
):
    try:
        base = model_class.from_pretrained(
            model_path,
            dtype=dtype,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
    except TypeError:
        base = model_class.from_pretrained(
            model_path,
            torch_dtype=dtype,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
    if unified:
        teacher = base
    else:
        adapters = _teacher_adapters(plan)
        teacher = peft_model_class.from_pretrained(
            base,
            str(adapters["gui_policy"]),
            adapter_name="gui_policy",
            is_trainable=False,
        )
        teacher.load_adapter(
            str(adapters["chat"]),
            adapter_name="chat",
            is_trainable=False,
        )
    teacher.to(device)
    teacher.eval()
    teacher.config.use_cache = False
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher


def _loss_terms(
    *,
    torch_module,
    functional,
    student,
    teacher,
    model_inputs: Mapping[str, object],
    prompt_length: int,
    completion_length: int,
    layer_map: Sequence[int],
    temperature: float,
    hidden_weight: float,
) -> tuple[object, object, object]:
    keep = completion_length + 1
    use_hidden = hidden_weight > 0

    with torch_module.no_grad():
        teacher_output = teacher(
            **model_inputs,
            use_cache=False,
            output_hidden_states=use_hidden,
            logits_to_keep=keep,
        )
        teacher_logits = teacher_output.logits[:, :-1, :].detach().float()
        teacher_hidden = None
        if use_hidden:
            teacher_hidden = tuple(
                teacher_output.hidden_states[int(index) + 1][:, prompt_length:, :].detach()
                for index in layer_map
            )
        del teacher_output

    student_output = student(
        **model_inputs,
        use_cache=False,
        output_hidden_states=use_hidden,
        logits_to_keep=keep,
    )
    student_logits = student_output.logits[:, :-1, :].float()
    targets = model_inputs["input_ids"][:, prompt_length:]
    if student_logits.shape[1] != targets.shape[1]:
        raise RuntimeError("completion logit window does not align with target tokens")
    if teacher_logits.shape != student_logits.shape:
        raise RuntimeError("teacher and student logits must share vocabulary and completion shape")

    supervised = functional.cross_entropy(
        student_logits.reshape(-1, student_logits.shape[-1]),
        targets.reshape(-1),
    )
    student_log_probs = functional.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = functional.softmax(teacher_logits / temperature, dim=-1)
    logit_kl = functional.kl_div(student_log_probs, teacher_probs, reduction="sum")
    logit_kl = logit_kl * (temperature**2) / targets.numel()

    hidden = student_logits.new_zeros(())
    if use_hidden:
        if teacher_hidden is None or student_output.hidden_states is None:
            raise RuntimeError("hidden-state distillation requested but hidden states are missing")
        if len(layer_map) != len(student_output.hidden_states) - 1:
            raise RuntimeError("student layer map does not match student hidden-state depth")
        hidden_terms = []
        for student_index, expected in enumerate(teacher_hidden):
            actual = student_output.hidden_states[student_index + 1][:, prompt_length:, :]
            if actual.shape != expected.shape:
                raise RuntimeError(
                    "hidden-state KD requires equal teacher/student widths; "
                    "set hidden_state_weight=0 for narrowed students"
                )
            hidden_terms.append(functional.mse_loss(actual.float(), expected.float()))
        hidden = torch_module.stack(hidden_terms).mean()
    return supervised, logit_kl, hidden


def train(args: argparse.Namespace) -> dict[str, object]:
    try:
        import torch
        import torch.nn.functional as F
        from peft import PeftModel
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, set_seed
    except ImportError as exc:
        raise RuntimeError("install the training extra: python -m pip install -e '.[train]'") from exc

    plan = load_student_plan(args.config)
    spec = spec_from_plan(plan, args.student)
    students = plan.get("students")
    if not isinstance(students, Mapping) or not isinstance(students.get(args.student), Mapping):
        raise ValueError(f"student plan does not define {args.student}")
    student_entry = dict(students[args.student])
    student_dir = Path(args.student_model or str(student_entry.get("initialized") or ""))
    if not student_dir.is_dir():
        raise FileNotFoundError(f"initialized student checkpoint not found: {student_dir}")
    build_manifest = _load_build_manifest(student_dir)
    layer_map = tuple(int(value) for value in build_manifest.get("teacher_layer_map") or ())
    if len(layer_map) != spec.text_layers:
        raise ValueError("student build manifest has an invalid teacher layer map")

    objective = plan.get("objective")
    if not isinstance(objective, Mapping):
        raise ValueError("student plan must define objective")
    temperature = float(args.temperature or objective.get("temperature", 2.0))
    supervised_weight = float(objective.get("supervised_weight", 0.5))
    kl_weight = float(objective.get("logit_kl_weight", 1.0))
    default_hidden_weight = float(objective.get("hidden_state_weight", 0.1))
    hidden_weight = args.hidden_state_weight if args.hidden_state_weight is not None else default_hidden_weight
    if spec.text_hidden_size != 2048 and args.hidden_state_weight is None:
        hidden_weight = 0.0
    learning_rate = float(args.learning_rate or objective.get("learning_rate", 1e-5))
    epochs = int(args.epochs or objective.get("epochs", 1))
    accumulation = int(args.gradient_accumulation or objective.get("gradient_accumulation", 8))
    grad_clip = float(args.grad_clip or objective.get("grad_clip", 1.0))
    if min(temperature, supervised_weight, kl_weight, learning_rate, grad_clip) <= 0:
        raise ValueError("distillation objective values must be positive")
    if hidden_weight < 0 or epochs < 1 or accumulation < 1:
        raise ValueError("hidden weight must be non-negative and training counts positive")

    records = _corpus_records(plan, args.max_records_per_corpus)
    set_seed(args.seed)
    rng = random.Random(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = _torch_dtype(torch, device)

    teacher_model = args.teacher_model or str(plan.get("initialization_teacher") or "")
    if not teacher_model:
        raise ValueError("teacher model is missing")
    teacher = _load_teacher(
        model_path=teacher_model,
        plan=plan,
        unified=args.teacher_unified,
        dtype=dtype,
        device=device,
        model_class=Qwen3VLForConditionalGeneration,
        peft_model_class=PeftModel,
    )
    try:
        student = Qwen3VLForConditionalGeneration.from_pretrained(
            str(student_dir),
            dtype=dtype,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
    except TypeError:
        student = Qwen3VLForConditionalGeneration.from_pretrained(
            str(student_dir),
            torch_dtype=dtype,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
    student.to(device)
    student.config.use_cache = False
    student.gradient_checkpointing_enable()
    if spec.freeze_vision_during_kd:
        for parameter in student.model.visual.parameters():
            parameter.requires_grad_(False)
    student.train()
    trainable = [parameter for parameter in student.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("student has no trainable parameters")
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.0)
    processor = AutoProcessor.from_pretrained(str(student_dir))

    updates = micro_steps = 0
    sums = {"loss": 0.0, "supervised": 0.0, "logit_kl": 0.0, "hidden": 0.0}
    optimizer.zero_grad(set_to_none=True)
    for _ in range(epochs):
        epoch_records = list(records)
        rng.shuffle(epoch_records)
        for record in epoch_records:
            if not args.teacher_unified:
                teacher.set_adapter(record.mode)
            inputs, prompt_length, completion_length = _encode(processor, record.messages, device)
            supervised, logit_kl, hidden = _loss_terms(
                torch_module=torch,
                functional=F,
                student=student,
                teacher=teacher,
                model_inputs=inputs,
                prompt_length=prompt_length,
                completion_length=completion_length,
                layer_map=layer_map,
                temperature=temperature,
                hidden_weight=hidden_weight,
            )
            loss = record.loss_weight * (
                supervised_weight * supervised + kl_weight * logit_kl + hidden_weight * hidden
            )
            (loss / accumulation).backward()
            micro_steps += 1
            sums["loss"] += float(loss.detach().cpu())
            sums["supervised"] += float(supervised.detach().cpu())
            sums["logit_kl"] += float(logit_kl.detach().cpu())
            sums["hidden"] += float(hidden.detach().cpu())
            if micro_steps % accumulation == 0:
                torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                updates += 1
    if micro_steps % accumulation:
        torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        updates += 1

    output = Path(
        args.output
        or str(
            student_entry.get("distilled")
            or f"artifacts/students/gui-owl-{args.student}-distilled"
        )
    )
    output.mkdir(parents=True, exist_ok=True)
    student.save_pretrained(output, safe_serialization=True)
    processor.save_pretrained(output)
    count = max(1, micro_steps)
    manifest = {
        "student": args.student,
        "student_initialization": str(student_dir),
        "teacher_model": teacher_model,
        "teacher_unified": bool(args.teacher_unified),
        "teacher_adapters": None
        if args.teacher_unified
        else {key: str(value) for key, value in _teacher_adapters(plan).items()},
        "records_per_epoch": len(records),
        "epochs": epochs,
        "micro_steps": micro_steps,
        "updates": updates,
        "planned_updates": math.ceil(len(records) * epochs / accumulation),
        "temperature": temperature,
        "supervised_weight": supervised_weight,
        "logit_kl_weight": kl_weight,
        "hidden_state_weight": hidden_weight,
        "freeze_vision": spec.freeze_vision_during_kd,
        "mean_loss": sums["loss"] / count,
        "mean_supervised_loss": sums["supervised"] / count,
        "mean_logit_kl": sums["logit_kl"] / count,
        "mean_hidden_loss": sums["hidden"] / count,
        "device": device,
        "dtype": str(dtype).removeprefix("torch."),
        "full_student_checkpoint": True,
        "credentials_written": False,
    }
    (output / "distillation-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", required=True, choices=("0.5b", "1b", "1.5b"))
    parser.add_argument("--config", default="config/gui_owl_students.json")
    parser.add_argument("--student-model")
    parser.add_argument("--teacher-model")
    parser.add_argument(
        "--teacher-unified",
        action="store_true",
        help="teacher is a full distilled checkpoint; do not load/switch PEFT adapters",
    )
    parser.add_argument("--output")
    parser.add_argument("--device")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--hidden-state-weight", type=float)
    parser.add_argument("--gradient-accumulation", type=int)
    parser.add_argument("--grad-clip", type=float)
    parser.add_argument("--max-records-per-corpus", type=int)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    print(json.dumps(train(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
