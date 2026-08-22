"""Shared free-running GUI correctness evaluation for distillation entrypoints."""
from __future__ import annotations

import statistics
from typing import Sequence

from android_drlqn.gui_owl_metrics import assistant_content_text, compare_gui_outputs


def evaluate_generation(base, **kwargs) -> dict[str, object]:
    """Evaluate syntax, action identity, and coordinate grounding.

    ``base`` is scripts.distill_gui_owl_student. Passing the module explicitly
    keeps this helper dependency-light and lets both the recovery wrapper and
    the policy-aligned wrapper reuse the base trainer's encoding/KD helpers.
    """

    torch_module = kwargs["torch_module"]
    functional = kwargs["functional"]
    student = kwargs["student"]
    teacher = kwargs["teacher"]
    teacher_unified = bool(kwargs["teacher_unified"])
    processor = kwargs["processor"]
    records: Sequence[object] = kwargs["records"]
    device = str(kwargs["device"])
    temperature = float(kwargs["temperature"])
    max_records = int(kwargs["max_records"])
    max_new_tokens = int(kwargs["max_new_tokens"])

    from project_gui_owl_train import parse_gui_owl_output

    selected = base._select_evaluation_records(records, max_records)
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
    gui_action_records = gui_action_matches = 0
    coordinate_errors: list[float] = []
    coordinate_within_50 = 0
    emitted_actions: set[str] = set()
    generated_x: list[float] = []

    was_training = student.training
    student.eval()
    with torch_module.no_grad():
        for record in selected:
            with base._teacher_mode(teacher, record.mode, teacher_unified):
                full, prompt_length, completion_length = base._encode_full(
                    processor, record.messages, device
                )
                supervised, forward_kl, _ = base._teacher_forced_terms(
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

                prompt = base._encode_prompt(processor, record.messages, device)
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
                trimmed = base._trim_generated_tokens(
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

                if record.mode != "gui_policy":
                    continue
                gui_total += 1
                if text and parse_gui_owl_output(text) is not None:
                    gui_valid += 1

                target_content = record.messages[-1].get("content") if record.messages else None
                target_text = assistant_content_text(target_content)
                correctness = compare_gui_outputs(text, target_text)
                if correctness.generated_action:
                    emitted_actions.add(correctness.generated_action)
                if correctness.target_action:
                    gui_action_records += 1
                    if correctness.action_match:
                        gui_action_matches += 1
                if correctness.coordinate_error is not None:
                    coordinate_errors.append(float(correctness.coordinate_error))
                    if correctness.coordinate_error <= 50.0:
                        coordinate_within_50 += 1
                if correctness.generated_coordinate is not None:
                    generated_x.append(float(correctness.generated_coordinate[0]))

    if was_training:
        student.train()

    total = len(selected)
    coordinate_records = len(coordinate_errors)
    return {
        "records": float(total),
        "first_token_eos_rate": first_eos / total,
        "empty_generation_rate": empty / total,
        "median_generated_tokens": float(statistics.median(lengths)),
        "mean_supervised_ce": sum(ce_values) / len(ce_values),
        "mean_teacher_kl": sum(kl_values) / len(kl_values),
        "gui_records": float(gui_total),
        "gui_parse_rate": gui_valid / gui_total if gui_total else 0.0,
        "gui_action_records": float(gui_action_records),
        "gui_action_name_match_rate": (
            gui_action_matches / gui_action_records if gui_action_records else 0.0
        ),
        "gui_distinct_action_types_emitted": float(len(emitted_actions)),
        "gui_emitted_action_types": sorted(emitted_actions),
        "gui_coordinate_records": float(coordinate_records),
        "gui_median_coordinate_error": (
            float(statistics.median(coordinate_errors)) if coordinate_errors else 0.0
        ),
        "gui_coordinate_within_50_rate": (
            coordinate_within_50 / coordinate_records if coordinate_records else 0.0
        ),
        "gui_generated_x_min": min(generated_x) if generated_x else None,
        "gui_generated_x_max": max(generated_x) if generated_x else None,
    }


__all__ = ["evaluate_generation"]
