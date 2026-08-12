#!/usr/bin/env python3
"""Recover and distill a structurally compressed GUI-Owl student.

The v1 one-epoch, teacher-forced recipe could reach a plausible completion CE
while free-running generation collapsed to EOS.  This trainer therefore uses
three explicit stages:

* recovery: frozen embedding/LM-head CE + forward KL + hidden-state alignment;
* on_policy: student-sampled autoregressive K3 reverse KL plus a small anchor;
* specialize: low-LR GUI/chat specialization after generative recovery.

Every stage evaluates greedy generation, records token/update counts, preserves
the strongest periodic checkpoint as a sibling directory, and refuses to mark
the final checkpoint accepted when the configured generation gate fails.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import random
import shutil
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, Sequence

from android_drlqn.gui_owl_student import (
    gate_from_mapping,
    generation_gate_failures,
    load_student_plan,
    resolve_peft_adapter_dir,
    spec_from_plan,
)


@dataclass(frozen=True)
class TrainingRecord:
    corpus: str
    mode: str
    messages: list[object]
    loss_weight: float


@dataclass(frozen=True)
class CorpusSource:
    name: str
    path: Path
    mode: str
    weight: float
    # Text-only corpora carry no images, so they exercise the language stack
    # but never the vision tower.
    text_only: bool = False


def _load_jsonl_messages(
    path: Path,
    default_mode: str,
    limit: int | None,
) -> list[tuple[str, list[object]]]:
    output: list[tuple[str, list[object]]] = []
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
        mode = (
            str(value.get("teacher_mode") or default_mode)
            if isinstance(value, Mapping)
            else default_mode
        )
        if mode not in {"base", "gui_policy", "chat"}:
            raise ValueError(f"{path}:{line_number}: unsupported teacher_mode {mode!r}")
        output.append((mode, list(messages)))
        if limit is not None and len(output) >= limit:
            break
    if not output:
        raise ValueError(f"distillation corpus is empty: {path}")
    return output


def _specialized_sources(plan: Mapping[str, object]) -> list[CorpusSource]:
    raw_corpora = plan.get("specialized_corpora")
    if not isinstance(raw_corpora, Mapping):
        raw_corpora = plan.get("corpora")
    if not isinstance(raw_corpora, Mapping):
        raise ValueError("student plan must define specialized_corpora")
    result: list[CorpusSource] = []
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
        result.append(CorpusSource(mode, path, mode, weight))
    return result


def _recovery_sources(plan: Mapping[str, object]) -> list[CorpusSource]:
    raw = plan.get("recovery_corpora")
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError(
            "recovery stage requires recovery_corpora; the small GUI/chat specialization set is not enough"
        )
    result: list[CorpusSource] = []
    for name, value in raw.items():
        if not isinstance(value, Mapping):
            raise ValueError(f"recovery corpus {name} must be an object")
        path = Path(str(value.get("path") or ""))
        required = bool(value.get("required", True))
        if not path.is_file():
            if required:
                raise FileNotFoundError(
                    f"required broad recovery corpus is missing: {path}; "
                    "prepare diverse instruction/multimodal teacher data before paid recovery training"
                )
            continue
        mode = str(value.get("teacher_mode") or "base")
        if mode not in {"base", "gui_policy", "chat"}:
            raise ValueError(f"unsupported teacher_mode for recovery corpus {name}: {mode}")
        weight = float(value.get("weight", 1.0))
        if weight <= 0:
            raise ValueError("recovery corpus weights must be positive")
        text_only = bool(value.get("text_only", False))
        result.append(CorpusSource(str(name), path, mode, weight, text_only))
    if not result:
        raise ValueError("no usable recovery corpora are configured")
    return result


def _student_entry(plan: Mapping[str, object], name: str) -> Mapping[str, object]:
    students = plan.get("students")
    if isinstance(students, Mapping) and isinstance(students.get(name), Mapping):
        return students[name]
    return {}


def _stage_sources(
    plan: Mapping[str, object],
    stage: str,
    stage_config: Mapping[str, object],
    *,
    student_entry: Mapping[str, object] | None = None,
    vision_trainable: bool = False,
) -> list[CorpusSource]:
    student_entry = student_entry or {}
    specialized = _specialized_sources(plan)
    if stage == "specialize":
        return specialized
    recovery = _recovery_sources(plan)
    # Stage A recovers general representations, so it defaults to the broad
    # corpus alone. Mixing the GUI/chat sets in here is what the v1 recipe did:
    # the student fitted a narrow distribution before it could form language at
    # all, reaching a plausible teacher-forced CE while free-running generation
    # collapsed. Specialization belongs in stage C, after generative recovery.
    include_specialized = bool(
        student_entry.get("recovery_include_specialized")
        if student_entry.get("recovery_include_specialized") is not None
        else stage_config.get("include_specialized", False)
    )
    if include_specialized:
        recovery.extend(specialized)
    elif vision_trainable and all(source.text_only for source in recovery):
        # A width-pruned student trains its vision tower, and the broad recovery
        # corpus is text-only: with no image in any record the tower is never
        # executed, so its pruned weights would receive no gradient for the whole
        # of stage A. Refuse rather than silently skip recovering it.
        raise ValueError(
            "vision is trainable for this student but every recovery corpus is "
            "text_only; set include_specialized=true for this stage or add a "
            "multimodal recovery corpus"
        )
    return recovery


def _corpus_records(
    sources: Sequence[CorpusSource], max_records: int | None
) -> list[TrainingRecord]:
    loaded: list[tuple[CorpusSource, list[tuple[str, list[object]]]]] = []
    for source in sources:
        loaded.append((source, _load_jsonl_messages(source.path, source.mode, max_records)))
    total = sum(len(items) for _, items in loaded)
    corpora = len(loaded)
    result: list[TrainingRecord] = []
    for source, items in loaded:
        scale = source.weight * total / (corpora * len(items))
        result.extend(
            TrainingRecord(source.name, mode, messages, scale)
            for mode, messages in items
        )
    return result


def _split_records(
    records: Sequence[TrainingRecord], *, holdout_fraction: float, seed: int
) -> tuple[list[TrainingRecord], list[TrainingRecord]]:
    if not 0 < holdout_fraction < 0.5:
        raise ValueError("holdout_fraction must be in (0, 0.5)")
    groups: dict[str, list[TrainingRecord]] = {}
    for record in records:
        groups.setdefault(record.corpus, []).append(record)
    rng = random.Random(seed)
    train: list[TrainingRecord] = []
    holdout: list[TrainingRecord] = []
    for items in groups.values():
        values = list(items)
        rng.shuffle(values)
        if len(values) == 1:
            train.extend(values)
            continue
        count = max(1, int(round(len(values) * holdout_fraction)))
        count = min(count, len(values) - 1)
        holdout.extend(values[:count])
        train.extend(values[count:])
    if not train or not holdout:
        raise ValueError("distillation needs enough records for both train and holdout splits")
    rng.shuffle(train)
    rng.shuffle(holdout)
    return train, holdout


def _to_device(batch: Mapping[str, object], device: str) -> dict[str, object]:
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in batch.items()
    }


def _encode_full(
    processor, messages: Sequence[object], device: str
) -> tuple[dict[str, object], int, int]:
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
    prompt_length = min(
        int(prompt["input_ids"].shape[1]), int(full["input_ids"].shape[1])
    )
    completion_length = int(full["input_ids"].shape[1]) - prompt_length
    if completion_length < 1:
        raise ValueError("assistant completion produced no training tokens")
    full.pop("labels", None)
    return full, prompt_length, completion_length


def _encode_prompt(processor, messages: Sequence[object], device: str) -> dict[str, object]:
    prompt = processor.apply_chat_template(
        list(messages[:-1]),
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    prompt = _to_device(prompt, device)
    prompt.pop("labels", None)
    return prompt


def _append_generated(
    prompt_inputs: Mapping[str, object], response_ids, torch_module
) -> dict[str, object]:
    prompt_ids = prompt_inputs["input_ids"]
    prompt_length = int(prompt_ids.shape[1])
    result: dict[str, object] = {}
    for key, value in prompt_inputs.items():
        if key in {"input_ids", "attention_mask", "position_ids", "cache_position"}:
            continue
        if (
            key in {"token_type_ids", "mm_token_type_ids"}
            and hasattr(value, "ndim")
            and value.ndim == 2
            and int(value.shape[1]) == prompt_length
        ):
            extension = torch_module.zeros(
                (int(value.shape[0]), int(response_ids.shape[1])),
                dtype=value.dtype,
                device=value.device,
            )
            result[key] = torch_module.cat((value, extension), dim=1)
        else:
            result[key] = value
    result["input_ids"] = torch_module.cat((prompt_ids, response_ids), dim=1)
    attention = prompt_inputs.get("attention_mask")
    if attention is not None:
        extension = torch_module.ones(
            (int(attention.shape[0]), int(response_ids.shape[1])),
            dtype=attention.dtype,
            device=attention.device,
        )
        result["attention_mask"] = torch_module.cat((attention, extension), dim=1)
    return result


def _torch_dtype(torch_module, device: str):
    return torch_module.bfloat16 if device.startswith("cuda") else torch_module.float32


def _load_build_manifest(student_dir: Path) -> dict[str, object]:
    for name in ("student-build-manifest.json", "compression-lineage.json"):
        path = student_dir / name
        if not path.is_file():
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, Mapping):
            return dict(value)
    raise FileNotFoundError(f"student compression lineage not found under: {student_dir}")


def _teacher_lineage(teacher_model: str, teacher_depth: int) -> tuple[int, ...]:
    path = Path(teacher_model)
    for name in ("compression-lineage.json", "student-build-manifest.json"):
        candidate = path / name
        if not candidate.is_file():
            continue
        value = json.loads(candidate.read_text(encoding="utf-8"))
        if isinstance(value, Mapping):
            ids = tuple(int(item) for item in value.get("text_original_layer_ids") or ())
            if len(ids) == teacher_depth:
                return ids
    return tuple(range(teacher_depth))


def _teacher_hidden_positions(
    teacher_model: str,
    teacher_depth: int,
    student_original_ids: Sequence[int],
) -> tuple[int, ...]:
    lineage = _teacher_lineage(teacher_model, teacher_depth)
    positions = {original: index for index, original in enumerate(lineage)}
    missing = [int(value) for value in student_original_ids if int(value) not in positions]
    if missing:
        raise ValueError(f"teacher checkpoint lacks student lineage layers: {missing}")
    return tuple(positions[int(value)] for value in student_original_ids)


def _teacher_adapters(plan: Mapping[str, object]) -> dict[str, Path]:
    value = plan.get("teacher_adapters")
    if not isinstance(value, Mapping):
        raise ValueError("student plan must define teacher_adapters")
    return {
        mode: resolve_peft_adapter_dir(Path(str(value.get(mode) or "")), mode)
        for mode in ("gui_policy", "chat")
    }


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


@contextlib.contextmanager
def _teacher_mode(teacher, mode: str, unified: bool) -> Iterator[None]:
    if unified:
        yield
        return
    if mode == "base":
        disable = getattr(teacher, "disable_adapter", None)
        if not callable(disable):
            raise RuntimeError("PEFT teacher cannot disable adapters for base-mode recovery")
        with disable():
            yield
        return
    teacher.set_adapter(mode)
    yield


def _teacher_forced_terms(
    *,
    torch_module,
    functional,
    student,
    teacher,
    model_inputs: Mapping[str, object],
    prompt_length: int,
    completion_length: int,
    teacher_hidden_positions: Sequence[int],
    temperature: float,
    hidden_weight: float,
    hidden_projector,
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
            if teacher_output.hidden_states is None:
                raise RuntimeError("teacher hidden states are missing")
            teacher_hidden = tuple(
                teacher_output.hidden_states[int(index) + 1][:, prompt_length:, :].detach()
                for index in teacher_hidden_positions
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
        raise RuntimeError("teacher and student logits must share vocabulary/completion shape")

    supervised = functional.cross_entropy(
        student_logits.reshape(-1, student_logits.shape[-1]),
        targets.reshape(-1),
    )
    student_log_probs = functional.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = functional.softmax(teacher_logits / temperature, dim=-1)
    forward_kl = functional.kl_div(student_log_probs, teacher_probs, reduction="sum")
    forward_kl = forward_kl * (temperature**2) / targets.numel()

    hidden = student_logits.new_zeros(())
    if use_hidden:
        if teacher_hidden is None or student_output.hidden_states is None:
            raise RuntimeError("hidden-state distillation requested but hidden states are missing")
        if len(teacher_hidden_positions) != len(student_output.hidden_states) - 1:
            raise RuntimeError("student lineage does not match hidden-state depth")
        terms = []
        for student_index, expected in enumerate(teacher_hidden):
            actual = student_output.hidden_states[student_index + 1][:, prompt_length:, :]
            if hidden_projector is not None:
                actual = hidden_projector(actual.to(dtype=hidden_projector.weight.dtype))
            if actual.shape != expected.shape:
                raise RuntimeError("hidden-state projector did not align teacher/student widths")
            terms.append(functional.mse_loss(actual.float(), expected.float()))
        hidden = torch_module.stack(terms).mean()
    return supervised, forward_kl, hidden


def _sampled_k3_loss(
    *,
    torch_module,
    functional,
    student,
    teacher,
    processor,
    messages: Sequence[object],
    device: str,
    min_new_tokens: int,
    max_new_tokens: int,
    sampling_temperature: float,
    top_p: float,
    delta_clip: float,
) -> tuple[object, int]:
    prompt_inputs = _encode_prompt(processor, messages, device)
    was_training = student.training
    student.eval()
    with torch_module.no_grad():
        sequences = student.generate(
            **prompt_inputs,
            do_sample=True,
            temperature=sampling_temperature,
            top_p=top_p,
            min_new_tokens=min_new_tokens,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )
    if was_training:
        student.train()
    prompt_length = int(prompt_inputs["input_ids"].shape[1])
    response_ids = sequences[:, prompt_length:]
    if int(response_ids.shape[1]) < 1:
        raise RuntimeError("student on-policy generation produced no sampled tokens")
    full_inputs = _append_generated(prompt_inputs, response_ids, torch_module)
    keep = int(response_ids.shape[1]) + 1

    student_output = student(**full_inputs, use_cache=False, logits_to_keep=keep)
    student_logits = student_output.logits[:, :-1, :].float()
    student_log_probs = functional.log_softmax(student_logits, dim=-1).gather(
        -1, response_ids.unsqueeze(-1)
    ).squeeze(-1)
    with torch_module.no_grad():
        teacher_output = teacher(**full_inputs, use_cache=False, logits_to_keep=keep)
        teacher_logits = teacher_output.logits[:, :-1, :].float()
        teacher_log_probs = functional.log_softmax(teacher_logits, dim=-1).gather(
            -1, response_ids.unsqueeze(-1)
        ).squeeze(-1)
    delta = torch_module.clamp(
        teacher_log_probs - student_log_probs,
        min=-float(delta_clip),
        max=float(delta_clip),
    )
    k3 = torch_module.exp(delta) - delta - 1.0
    return k3.mean(), int(response_ids.numel())


def _freeze_for_stage(
    student,
    *,
    freeze_embeddings: bool,
    freeze_lm_head: bool,
    freeze_vision: bool,
) -> None:
    for parameter in student.parameters():
        parameter.requires_grad_(True)
    if freeze_embeddings:
        for parameter in student.model.language_model.embed_tokens.parameters():
            parameter.requires_grad_(False)
    if freeze_lm_head:
        for parameter in student.lm_head.parameters():
            parameter.requires_grad_(False)
    if freeze_vision:
        for parameter in student.model.visual.parameters():
            parameter.requires_grad_(False)


def _make_hidden_projector(
    torch_module,
    student_hidden: int,
    teacher_hidden: int,
    build_manifest: Mapping[str, object],
    device: str,
    dtype,
):
    if student_hidden == teacher_hidden:
        return None
    projector = torch_module.nn.Linear(
        student_hidden, teacher_hidden, bias=False, device=device, dtype=dtype
    )
    indices: object = None
    calibration_path = Path(str(build_manifest.get("calibration_manifest") or ""))
    if calibration_path.is_file():
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        recommended = calibration.get("recommended") if isinstance(calibration, Mapping) else None
        if isinstance(recommended, Mapping):
            indices = recommended.get(f"text_hidden_indices_{student_hidden}")
    with torch_module.no_grad():
        projector.weight.zero_()
        if (
            isinstance(indices, Sequence)
            and not isinstance(indices, (str, bytes))
            and len(indices) == student_hidden
        ):
            for student_index, teacher_index in enumerate(indices):
                teacher_index = int(teacher_index)
                if 0 <= teacher_index < teacher_hidden:
                    projector.weight[teacher_index, student_index] = 1.0
        else:
            torch_module.nn.init.orthogonal_(projector.weight)
            projector.weight.mul_(0.25)
    return projector


def _trim_generated_tokens(
    ids: Sequence[int], *, eos_ids: set[int], pad_id: int | None
) -> list[int]:
    result: list[int] = []
    for value in ids:
        token = int(value)
        if token in eos_ids or (pad_id is not None and token == pad_id):
            break
        result.append(token)
    return result


def _evaluate_generation(
    *,
    torch_module,
    functional,
    student,
    teacher,
    teacher_unified: bool,
    processor,
    records: Sequence[TrainingRecord],
    device: str,
    temperature: float,
    max_records: int,
    max_new_tokens: int,
) -> dict[str, float]:
    from project_gui_owl_train import parse_gui_owl_output

    selected = list(records[:max_records])
    if not selected:
        raise ValueError("generation evaluation has no records")
    eos = processor.tokenizer.eos_token_id
    eos_ids = {int(eos)} if isinstance(eos, int) else {int(value) for value in (eos or ())}
    pad_id = processor.tokenizer.pad_token_id
    first_eos = empty = 0
    lengths: list[int] = []
    ce_values: list[float] = []
    kl_values: list[float] = []
    gui_total = gui_valid = 0
    was_training = student.training
    student.eval()
    with torch_module.no_grad():
        for record in selected:
            with _teacher_mode(teacher, record.mode, teacher_unified):
                full, prompt_length, completion_length = _encode_full(
                    processor, record.messages, device
                )
                supervised, forward_kl, _ = _teacher_forced_terms(
                    torch_module=torch_module,
                    functional=functional,
                    student=student,
                    teacher=teacher,
                    model_inputs=full,
                    prompt_length=prompt_length,
                    completion_length=completion_length,
                    teacher_hidden_positions=(),
                    temperature=temperature,
                    hidden_weight=0.0,
                    hidden_projector=None,
                )
                ce_values.append(float(supervised.cpu()))
                kl_values.append(float(forward_kl.cpu()))

                prompt = _encode_prompt(processor, record.messages, device)
                sequence = student.generate(
                    **prompt,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                )
                prompt_tokens = int(prompt["input_ids"].shape[1])
                generated = [int(value) for value in sequence[0, prompt_tokens:].tolist()]
                if generated and generated[0] in eos_ids:
                    first_eos += 1
                trimmed = _trim_generated_tokens(
                    generated, eos_ids=eos_ids, pad_id=pad_id
                )
                lengths.append(len(trimmed))
                text = processor.tokenizer.decode(
                    trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                ).strip()
                if not text:
                    empty += 1
                if record.mode == "gui_policy":
                    gui_total += 1
                    if text and parse_gui_owl_output(text) is not None:
                        gui_valid += 1
    if was_training:
        student.train()
    total = len(selected)
    return {
        "records": float(total),
        "first_token_eos_rate": first_eos / total,
        "empty_generation_rate": empty / total,
        "median_generated_tokens": float(statistics.median(lengths)),
        "mean_supervised_ce": sum(ce_values) / len(ce_values),
        "mean_teacher_kl": sum(kl_values) / len(kl_values),
        "gui_parse_rate": gui_valid / gui_total if gui_total else 1.0,
    }


def _stage_config(plan: Mapping[str, object], stage: str) -> dict[str, object]:
    stages = plan.get("stages")
    if not isinstance(stages, Mapping) or not isinstance(stages.get(stage), Mapping):
        raise ValueError(f"student plan lacks stage configuration: {stage}")
    return dict(stages[stage])


def _replace_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _copy_lineage(source_dir: Path, destination: Path) -> None:
    for name in ("student-build-manifest.json", "compression-lineage.json"):
        source = source_dir / name
        if source.is_file():
            shutil.copy2(source, destination / name)


def _save_checkpoint(
    student,
    processor,
    path: Path,
    source_dir: Path,
    metrics: Mapping[str, object] | None = None,
) -> None:
    _replace_dir(path)
    student.save_pretrained(path, safe_serialization=True)
    processor.save_pretrained(path)
    _copy_lineage(source_dir, path)
    if metrics is not None:
        (path / "checkpoint-metrics.json").write_text(
            json.dumps(dict(metrics), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


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
    stage_config = _stage_config(plan, args.stage)
    student_dir = Path(args.student_model or str(student_entry.get("initialized") or ""))
    if not student_dir.is_dir():
        raise FileNotFoundError(f"student checkpoint not found: {student_dir}")
    build_manifest = _load_build_manifest(student_dir)
    student_original_ids = tuple(
        int(value) for value in build_manifest.get("text_original_layer_ids") or ()
    )
    if not student_original_ids:
        raise ValueError("student compression lineage has no text_original_layer_ids")

    freeze_embeddings = bool(stage_config.get("freeze_embeddings", True))
    freeze_lm_head = bool(stage_config.get("freeze_lm_head", True))
    freeze_vision = bool(
        stage_config.get("freeze_vision", spec.freeze_vision_during_kd)
    )

    sources = _stage_sources(
        plan,
        args.stage,
        stage_config,
        student_entry=_student_entry(plan, args.student),
        vision_trainable=not freeze_vision,
    )
    records = _corpus_records(sources, args.max_records_per_corpus)
    train_records, holdout_records = _split_records(
        records,
        holdout_fraction=float(stage_config.get("holdout_fraction", 0.05)),
        seed=args.seed,
    )

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
    teacher_hidden_positions = _teacher_hidden_positions(
        teacher_model,
        int(teacher.config.text_config.num_hidden_layers),
        student_original_ids,
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

    _freeze_for_stage(
        student,
        freeze_embeddings=freeze_embeddings,
        freeze_lm_head=freeze_lm_head,
        freeze_vision=freeze_vision,
    )
    student.train()

    temperature = float(args.temperature or stage_config.get("temperature", 2.0))
    supervised_weight = float(stage_config.get("supervised_weight", 0.25))
    forward_kl_weight = float(stage_config.get("forward_kl_weight", 1.0))
    hidden_weight = float(stage_config.get("hidden_state_weight", 0.0))
    k3_weight = float(stage_config.get("k3_weight", 0.0))
    learning_rate = float(args.learning_rate or stage_config.get("learning_rate", 5e-6))
    accumulation = int(
        args.gradient_accumulation or stage_config.get("gradient_accumulation", 8)
    )
    grad_clip = float(args.grad_clip or stage_config.get("grad_clip", 1.0))
    max_updates = int(args.max_updates or stage_config.get("max_updates", 1000))
    eval_interval = int(stage_config.get("eval_interval_updates", 250))
    eval_records = int(stage_config.get("eval_records", 32))
    eval_max_new_tokens = int(stage_config.get("eval_max_new_tokens", 64))
    min_new_tokens = int(stage_config.get("on_policy_min_new_tokens", 4))
    max_new_tokens = int(stage_config.get("on_policy_max_new_tokens", 64))
    sampling_temperature = float(stage_config.get("sampling_temperature", 0.8))
    top_p = float(stage_config.get("top_p", 0.95))
    # K3 is exp(delta) - delta - 1, so the clip bounds the loss exponentially:
    # a clip of 20 admits terms near exp(20) ~ 4.9e8, where one token on which
    # the teacher strongly disagrees dictates the update direction. grad_clip
    # bounds the step size but not which direction it points.
    delta_clip = float(stage_config.get("k3_delta_clip", 5.0))
    if min(temperature, learning_rate, grad_clip, accumulation, max_updates) <= 0:
        raise ValueError("invalid positive training hyperparameter")
    if min(supervised_weight, forward_kl_weight, hidden_weight, k3_weight) < 0:
        raise ValueError("loss weights must be non-negative")
    if args.stage == "on_policy" and k3_weight <= 0:
        raise ValueError("on_policy stage requires a positive k3_weight")

    hidden_projector = None
    student_hidden = int(student.config.text_config.hidden_size)
    teacher_hidden = int(teacher.config.text_config.hidden_size)
    if hidden_weight > 0 and student_hidden != teacher_hidden:
        hidden_projector = _make_hidden_projector(
            torch,
            student_hidden,
            teacher_hidden,
            build_manifest,
            device,
            dtype,
        )

    trainable = [parameter for parameter in student.parameters() if parameter.requires_grad]
    if hidden_projector is not None:
        trainable.extend(
            parameter for parameter in hidden_projector.parameters() if parameter.requires_grad
        )
    if not trainable:
        raise RuntimeError("student stage has no trainable parameters")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=float(stage_config.get("weight_decay", 0.0)),
    )
    processor = AutoProcessor.from_pretrained(str(student_dir))
    raw_gate = stage_config.get("generation_gate")
    gate = gate_from_mapping(raw_gate if isinstance(raw_gate, Mapping) else None)

    output = Path(
        args.output
        or str(
            student_entry.get("distilled")
            or f"artifacts/students/v2/gui-owl-{args.student}-distilled"
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    best_output = output.parent / f"{output.name}-best"

    initial_eval = _evaluate_generation(
        torch_module=torch,
        functional=F,
        student=student,
        teacher=teacher,
        teacher_unified=args.teacher_unified,
        processor=processor,
        records=holdout_records,
        device=device,
        temperature=temperature,
        max_records=min(eval_records, len(holdout_records)),
        max_new_tokens=eval_max_new_tokens,
    )
    initial_failures = generation_gate_failures(initial_eval, gate)

    updates = micro_steps = dataset_passes = 0
    completion_tokens = sampled_tokens = 0
    sums = {
        "loss": 0.0,
        "supervised": 0.0,
        "forward_kl": 0.0,
        "hidden": 0.0,
        "k3": 0.0,
    }
    evaluations: list[dict[str, object]] = [
        {"update": 0, "metrics": initial_eval, "gate_failures": list(initial_failures)}
    ]
    best_score: tuple[int, float] | None = None
    best_update: int | None = None
    optimizer.zero_grad(set_to_none=True)

    while updates < max_updates:
        epoch_records = list(train_records)
        rng.shuffle(epoch_records)
        dataset_passes += 1
        for record in epoch_records:
            with _teacher_mode(teacher, record.mode, args.teacher_unified):
                full, prompt_length, completion_length = _encode_full(
                    processor, record.messages, device
                )
                completion_tokens += completion_length
                supervised, forward_kl, hidden = _teacher_forced_terms(
                    torch_module=torch,
                    functional=F,
                    student=student,
                    teacher=teacher,
                    model_inputs=full,
                    prompt_length=prompt_length,
                    completion_length=completion_length,
                    teacher_hidden_positions=teacher_hidden_positions,
                    temperature=temperature,
                    hidden_weight=hidden_weight,
                    hidden_projector=hidden_projector,
                )
                k3 = supervised.new_zeros(())
                if k3_weight > 0:
                    k3, sample_count = _sampled_k3_loss(
                        torch_module=torch,
                        functional=F,
                        student=student,
                        teacher=teacher,
                        processor=processor,
                        messages=record.messages,
                        device=device,
                        min_new_tokens=min_new_tokens,
                        max_new_tokens=max_new_tokens,
                        sampling_temperature=sampling_temperature,
                        top_p=top_p,
                        delta_clip=delta_clip,
                    )
                    sampled_tokens += sample_count
                loss = record.loss_weight * (
                    supervised_weight * supervised
                    + forward_kl_weight * forward_kl
                    + hidden_weight * hidden
                    + k3_weight * k3
                )
            (loss / accumulation).backward()
            micro_steps += 1
            sums["loss"] += float(loss.detach().cpu())
            sums["supervised"] += float(supervised.detach().cpu())
            sums["forward_kl"] += float(forward_kl.detach().cpu())
            sums["hidden"] += float(hidden.detach().cpu())
            sums["k3"] += float(k3.detach().cpu())

            if micro_steps % accumulation != 0:
                continue
            torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            updates += 1

            if eval_interval > 0 and (updates % eval_interval == 0 or updates == max_updates):
                metrics = _evaluate_generation(
                    torch_module=torch,
                    functional=F,
                    student=student,
                    teacher=teacher,
                    teacher_unified=args.teacher_unified,
                    processor=processor,
                    records=holdout_records,
                    device=device,
                    temperature=temperature,
                    max_records=min(eval_records, len(holdout_records)),
                    max_new_tokens=eval_max_new_tokens,
                )
                failures = generation_gate_failures(metrics, gate)
                evaluation = {
                    "update": updates,
                    "metrics": metrics,
                    "gate_failures": list(failures),
                }
                evaluations.append(evaluation)
                score = (
                    len(failures),
                    float(metrics["mean_teacher_kl"])
                    + 0.25 * float(metrics["mean_supervised_ce"]),
                )
                if best_score is None or score < best_score:
                    best_score = score
                    best_update = updates
                    _save_checkpoint(
                        student, processor, best_output, student_dir, evaluation
                    )
            if updates >= max_updates:
                break

    final_eval = _evaluate_generation(
        torch_module=torch,
        functional=F,
        student=student,
        teacher=teacher,
        teacher_unified=args.teacher_unified,
        processor=processor,
        records=holdout_records,
        device=device,
        temperature=temperature,
        max_records=min(eval_records, len(holdout_records)),
        max_new_tokens=eval_max_new_tokens,
    )
    final_failures = generation_gate_failures(final_eval, gate)
    if not evaluations or int(evaluations[-1]["update"]) != updates:
        evaluations.append(
            {"update": updates, "metrics": final_eval, "gate_failures": list(final_failures)}
        )

    # The best checkpoint lives next to output, not under it, so replacing the
    # final directory cannot accidentally erase a stronger intermediate state.
    _save_checkpoint(student, processor, output, student_dir)
    count = max(1, micro_steps)
    manifest = {
        "schema_version": 2,
        "student": args.student,
        "stage": args.stage,
        "student_initialization": str(student_dir),
        "teacher_model": teacher_model,
        "teacher_unified": bool(args.teacher_unified),
        "teacher_adapters": None
        if args.teacher_unified
        else {key: str(value) for key, value in _teacher_adapters(plan).items()},
        "train_records": len(train_records),
        "holdout_records": len(holdout_records),
        "optimizer_updates": updates,
        "micro_steps": micro_steps,
        "dataset_passes": dataset_passes,
        "completion_tokens": completion_tokens,
        "student_sampled_tokens": sampled_tokens,
        "max_updates": max_updates,
        "temperature": temperature,
        "supervised_weight": supervised_weight,
        "forward_kl_weight": forward_kl_weight,
        "hidden_state_weight": hidden_weight,
        "k3_reverse_kl_weight": k3_weight,
        "learning_rate": learning_rate,
        "gradient_accumulation": accumulation,
        "freeze_embeddings": freeze_embeddings,
        "freeze_lm_head": freeze_lm_head,
        "freeze_vision": freeze_vision,
        "mean_loss": sums["loss"] / count,
        "mean_supervised_loss": sums["supervised"] / count,
        "mean_forward_logit_kl": sums["forward_kl"] / count,
        "mean_hidden_loss": sums["hidden"] / count,
        "mean_k3_reverse_kl": sums["k3"] / count,
        "initial_evaluation": initial_eval,
        "initial_gate_failures": list(initial_failures),
        "evaluations": evaluations,
        "best_update": best_update,
        "best_checkpoint": str(best_output) if best_update is not None else None,
        "final_evaluation": final_eval,
        "final_gate_failures": list(final_failures),
        "accepted": not final_failures,
        "device": device,
        "dtype": str(dtype).removeprefix("torch."),
        "full_student_checkpoint": True,
        "credentials_written": False,
    }
    (output / "distillation-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if final_failures and not args.allow_gate_failure:
        raise RuntimeError(
            "distillation finished but free-running generation gate failed: "
            + "; ".join(final_failures)
            + f". Final debug checkpoint: {output}; best intermediate: {best_output}."
        )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", required=True, choices=("0.5b", "1b", "1.5b"))
    parser.add_argument(
        "--stage", default="recovery", choices=("recovery", "on_policy", "specialize")
    )
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
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--gradient-accumulation", type=int)
    parser.add_argument("--grad-clip", type=float)
    parser.add_argument("--max-records-per-corpus", type=int)
    parser.add_argument("--allow-gate-failure", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    print(json.dumps(train(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
