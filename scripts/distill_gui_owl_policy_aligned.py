#!/usr/bin/env python3
"""Run GUI-Owl distillation with policy-alignment safeguards.

This entrypoint keeps the base trainer/checkpoint format but fixes details that
matter when a compressed student has collapsed to first-token EOS or generic
assistant mode on GUI prompts:

* generation evaluation is stratified so GUI policy records receive a material
  share of the holdout budget;
* K3 clips its delta value with a straight-through clamp so badly mismatched
  sampled tokens still receive a bounded, non-zero gradient instead of falling
  outside ``torch.clamp`` and getting zero gradient;
* teacher-forced CE/KL can upweight the first completion token and an optional
  early completion prefix so recovery can progress from choosing ``Action`` to
  continuing the ``Action: ... <tool_call>`` grammar.

Set ``GUI_OWL_ONLY_GUI=1`` for a short syntax bridge to train only on the
GUI-policy corpus. ``GUI_OWL_FIRST_TOKEN_WEIGHT`` controls token zero;
``GUI_OWL_PREFIX_TOKENS`` and ``GUI_OWL_PREFIX_WEIGHT`` optionally weight an
early teacher-forced prefix, with the first-token weight taking precedence.
Later on-policy/final stages can use the normal corpus mix and default weights.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import distill_gui_owl_student as base  # noqa: E402

_BASE_EVALUATE = base._evaluate_generation
_BASE_STAGE_SOURCES = base._stage_sources
MIN_GUI_EVAL_RECORDS = 8


def _argv_value(name: str, default: str) -> str:
    try:
        index = sys.argv.index(name)
    except ValueError:
        return default
    if index + 1 >= len(sys.argv):
        return default
    return sys.argv[index + 1]


_STAGE = _argv_value("--stage", "recovery")
_DEFAULT_FIRST_TOKEN_WEIGHT = 8.0 if _STAGE == "specialize" else 4.0
FIRST_TOKEN_WEIGHT = float(
    os.environ.get("GUI_OWL_FIRST_TOKEN_WEIGHT", str(_DEFAULT_FIRST_TOKEN_WEIGHT))
)
PREFIX_TOKENS = int(os.environ.get("GUI_OWL_PREFIX_TOKENS", "0"))
PREFIX_WEIGHT = float(os.environ.get("GUI_OWL_PREFIX_WEIGHT", "1.0"))
ONLY_GUI = os.environ.get("GUI_OWL_ONLY_GUI", "0") == "1"

if FIRST_TOKEN_WEIGHT <= 0:
    raise ValueError("GUI_OWL_FIRST_TOKEN_WEIGHT must be positive")
if PREFIX_TOKENS < 0:
    raise ValueError("GUI_OWL_PREFIX_TOKENS must be non-negative")
if PREFIX_WEIGHT <= 0:
    raise ValueError("GUI_OWL_PREFIX_WEIGHT must be positive")


def _stratified_evaluate_generation(**kwargs):
    records: Sequence[base.TrainingRecord] = kwargs["records"]
    max_records = int(kwargs["max_records"])
    if max_records < 1:
        return _BASE_EVALUATE(**kwargs)

    gui = [record for record in records if record.mode == "gui_policy"]
    other = [record for record in records if record.mode != "gui_policy"]
    if not gui:
        return _BASE_EVALUATE(**kwargs)

    # Reserve at least eight GUI examples where available and otherwise use up
    # to half the evaluation budget. This keeps EOS/CE/KL metrics mixed while
    # making the GUI parse-rate gate statistically meaningful.
    gui_quota = min(
        len(gui),
        max_records,
        max(MIN_GUI_EVAL_RECORDS, max_records // 2),
    )
    other_quota = min(len(other), max_records - gui_quota)
    selected = gui[:gui_quota] + other[:other_quota]
    if len(selected) < max_records:
        used_gui = gui_quota
        used_other = other_quota
        selected.extend(gui[used_gui : used_gui + max_records - len(selected)])
        if len(selected) < max_records:
            selected.extend(other[used_other : used_other + max_records - len(selected)])

    forwarded = dict(kwargs)
    forwarded["records"] = selected
    forwarded["max_records"] = len(selected)
    return _BASE_EVALUATE(**forwarded)


def _policy_stage_sources(
    plan,
    stage,
    stage_config,
    *,
    student_entry=None,
    vision_trainable=False,
):
    sources = _BASE_STAGE_SOURCES(
        plan,
        stage,
        stage_config,
        student_entry=student_entry,
        vision_trainable=vision_trainable,
    )
    if not ONLY_GUI:
        return sources
    gui = [source for source in sources if source.mode == "gui_policy"]
    if not gui:
        raise ValueError("GUI_OWL_ONLY_GUI=1 but this stage has no gui_policy corpus")
    return gui


def _weighted_teacher_forced_terms(
    *,
    torch_module,
    functional,
    student,
    teacher,
    model_inputs,
    prompt_length,
    completion_length,
    teacher_hidden_positions,
    temperature,
    hidden_weight,
    hidden_projector,
):
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

    token_weights = student_logits.new_ones(targets.shape, dtype=torch_module.float32)
    prefix_tokens = min(PREFIX_TOKENS, int(targets.shape[1]))
    if prefix_tokens > 0 and PREFIX_WEIGHT != 1.0:
        token_weights[:, :prefix_tokens] = PREFIX_WEIGHT
    if targets.shape[1] > 0 and FIRST_TOKEN_WEIGHT != 1.0:
        # Token zero remains independently controllable so mode selection can
        # stay strongly anchored while the rest of the early GUI grammar is
        # being recovered.
        token_weights[:, 0] = FIRST_TOKEN_WEIGHT
    denominator = token_weights.sum().clamp_min(1.0)

    ce = functional.cross_entropy(
        student_logits.reshape(-1, student_logits.shape[-1]),
        targets.reshape(-1),
        reduction="none",
    ).view_as(token_weights)
    supervised = (ce * token_weights).sum() / denominator

    student_log_probs = functional.log_softmax(student_logits / temperature, dim=-1)
    teacher_log_probs = functional.log_softmax(teacher_logits / temperature, dim=-1)
    teacher_probs = teacher_log_probs.exp()
    token_kl = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1)
    forward_kl = (token_kl * token_weights).sum() / denominator
    forward_kl = forward_kl * (temperature**2)

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


def _sampled_k3_with_live_clipped_gradient(
    *,
    torch_module,
    functional,
    student,
    teacher,
    processor,
    messages,
    device,
    min_new_tokens,
    max_new_tokens,
    sampling_temperature,
    top_p,
    delta_clip,
):
    prompt_inputs = base._encode_prompt(processor, messages, device)
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
    full_inputs = base._append_generated(prompt_inputs, response_ids, torch_module)
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

    raw_delta = teacher_log_probs - student_log_probs
    clipped_delta = raw_delta.clamp(min=-float(delta_clip), max=float(delta_clip))
    # Plain torch.clamp has derivative 0 outside the interval. In the observed
    # EOS collapse, teacher log p(EOS) - student log p(EOS) can be far below -5,
    # so the old implementation assigned exactly zero K3 gradient to the token
    # that most needed correction. Keep the bounded forward value but route the
    # derivative through raw_delta. The resulting K3 derivative is itself
    # bounded by the clipped exponential.
    delta = raw_delta + (clipped_delta - raw_delta).detach()
    k3 = torch_module.exp(delta) - delta - 1.0
    return k3.mean(), int(response_ids.numel())


base._evaluate_generation = _stratified_evaluate_generation
base._stage_sources = _policy_stage_sources
base._teacher_forced_terms = _weighted_teacher_forced_terms
base._sampled_k3_loss = _sampled_k3_with_live_clipped_gradient


if __name__ == "__main__":
    raise SystemExit(base.main())
