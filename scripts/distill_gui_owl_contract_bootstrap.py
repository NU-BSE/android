#!/usr/bin/env python3
"""GUI-only contract bootstrap for a structurally pruned GUI-Owl checkpoint.

The depth-24 diagnostics showed a specific teacher-forcing/free-generation gap:
the student learns the natural-language ``Action:`` sentence while terminating
before ``<tool_call>``.  This entrypoint therefore concentrates CE/KL mass on
the tool-call suffix and temporarily unfreezes the LM head.  It is intentionally
used only as a short bootstrap before normal grounded recovery.
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
from scripts.gui_owl_grounded_evaluation import evaluate_generation  # noqa: E402
from scripts.gui_owl_grounding_revision import balanced_corpus_records  # noqa: E402


def _argv_value(name: str) -> str | None:
    try:
        index = sys.argv.index(name)
    except ValueError:
        return None
    if index + 1 >= len(sys.argv):
        return None
    return sys.argv[index + 1]


STUDENT_MODEL = _argv_value("--student-model")
if not STUDENT_MODEL:
    raise SystemExit("distill_gui_owl_contract_bootstrap.py requires --student-model")

try:
    from transformers import AutoProcessor
except ImportError as exc:  # pragma: no cover - paid training environment check
    raise RuntimeError("install the training extra: python -m pip install -e '.[train]'") from exc

_BOOT_PROCESSOR = AutoProcessor.from_pretrained(STUDENT_MODEL)
_TOKENIZER = _BOOT_PROCESSOR.tokenizer

PREFIX_WEIGHT = float(os.environ.get("GUI_OWL_CONTRACT_PREFIX_WEIGHT", "0.25"))
SUFFIX_WEIGHT = float(os.environ.get("GUI_OWL_CONTRACT_SUFFIX_WEIGHT", "8.0"))
FOCUS_WEIGHT = float(os.environ.get("GUI_OWL_CONTRACT_FOCUS_WEIGHT", "16.0"))
if min(PREFIX_WEIGHT, SUFFIX_WEIGHT, FOCUS_WEIGHT) <= 0:
    raise ValueError("contract token weights must be positive")
if not PREFIX_WEIGHT <= SUFFIX_WEIGHT <= FOCUS_WEIGHT:
    raise ValueError("expected PREFIX_WEIGHT <= SUFFIX_WEIGHT <= FOCUS_WEIGHT")


def _token_ids(text: str) -> tuple[int, ...]:
    return tuple(int(value) for value in _TOKENIZER.encode(text, add_special_tokens=False))


_OPEN_PATTERNS = tuple(
    pattern
    for pattern in (_token_ids("\n<tool_call>"), _token_ids("<tool_call>"))
    if pattern
)
_CLOSE_PATTERNS = tuple(
    pattern
    for pattern in (_token_ids("</tool_call>"), _token_ids("</tool_call>\n"))
    if pattern
)
_FOCUS_PATTERNS = tuple(
    pattern
    for pattern in (
        _token_ids("<tool_call>"),
        _token_ids('"name":"mobile.'),
        _token_ids('"arguments":'),
        _token_ids('"coordinate":['),
        _token_ids('"coordinate2":['),
        _token_ids("</tool_call>"),
    )
    if pattern
)


def _find(ids: Sequence[int], pattern: Sequence[int], start: int = 0) -> int | None:
    if not pattern:
        return None
    stop = len(ids) - len(pattern) + 1
    for index in range(max(0, start), max(0, stop)):
        if tuple(ids[index : index + len(pattern)]) == tuple(pattern):
            return index
    return None


def _find_any(ids: Sequence[int], patterns: Sequence[Sequence[int]], start: int = 0):
    matches = []
    for pattern in patterns:
        index = _find(ids, pattern, start)
        if index is not None:
            matches.append((index, tuple(pattern)))
    return min(matches, key=lambda value: value[0]) if matches else None


def _contract_token_weights(targets, torch_module):
    weights = targets.new_full(targets.shape, PREFIX_WEIGHT, dtype=torch_module.float32)
    for row in range(int(targets.shape[0])):
        ids = [int(value) for value in targets[row].tolist()]
        opening = _find_any(ids, _OPEN_PATTERNS)
        if opening is None:
            decoded = _TOKENIZER.decode(ids, skip_special_tokens=False)
            raise RuntimeError(
                "GUI contract bootstrap target lacks <tool_call> marker; refusing silent fallback: "
                + decoded[-300:]
            )
        open_index, open_pattern = opening
        closing = _find_any(ids, _CLOSE_PATTERNS, open_index + len(open_pattern))
        suffix_end = len(ids)
        if closing is not None:
            close_index, close_pattern = closing
            suffix_end = min(len(ids), close_index + len(close_pattern))
        weights[row, open_index:suffix_end] = SUFFIX_WEIGHT

        for pattern in _FOCUS_PATTERNS:
            search = open_index
            while search < suffix_end:
                found = _find(ids, pattern, search)
                if found is None or found >= suffix_end:
                    break
                end = min(suffix_end, found + len(pattern))
                weights[row, found:end] = FOCUS_WEIGHT
                search = max(found + 1, end)

        # The first tool-marker token is the exact autoregressive transition
        # that the failed checkpoints never entered, so give it focused mass.
        weights[row, open_index : min(suffix_end, open_index + len(open_pattern))] = FOCUS_WEIGHT
    return weights


_ORIGINAL_STAGE_SOURCES = base._stage_sources
_ORIGINAL_CORPUS_RECORDS = base._corpus_records
_ORIGINAL_FREEZE = base._freeze_for_stage


def _gui_only_sources(plan, stage, stage_config, *, student_entry=None, vision_trainable=False):
    sources = _ORIGINAL_STAGE_SOURCES(
        plan,
        stage,
        stage_config,
        student_entry=student_entry,
        vision_trainable=vision_trainable,
    )
    gui = [source for source in sources if source.mode == "gui_policy"]
    if not gui:
        raise ValueError("contract bootstrap requires a gui_policy corpus")
    return gui


def _balanced_records(sources, max_records):
    return balanced_corpus_records(_ORIGINAL_CORPUS_RECORDS, sources, max_records)


def _bootstrap_freeze(student, *, freeze_embeddings, freeze_lm_head, freeze_vision):
    # LM-head freedom is deliberate and temporary: the pruned hidden stack can
    # first recover the discrete output contract instead of reconstructing the
    # teacher's exact output-space geometry before emitting a tool marker.
    _ORIGINAL_FREEZE(
        student,
        freeze_embeddings=freeze_embeddings,
        freeze_lm_head=False,
        freeze_vision=freeze_vision,
    )


def _suffix_weighted_teacher_forced_terms(
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

    token_weights = _contract_token_weights(targets, torch_module)
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


def _grounded_evaluate(**kwargs):
    return evaluate_generation(base, **kwargs)


base._stage_sources = _gui_only_sources
base._corpus_records = _balanced_records
base._freeze_for_stage = _bootstrap_freeze
base._teacher_forced_terms = _suffix_weighted_teacher_forced_terms
base._evaluate_generation = _grounded_evaluate

print(
    "GUI contract bootstrap token weights: "
    f"prefix={PREFIX_WEIGHT:g}, suffix={SUFFIX_WEIGHT:g}, focus={FOCUS_WEIGHT:g}; LM head trainable",
    flush=True,
)


if __name__ == "__main__":
    raise SystemExit(base.main())
