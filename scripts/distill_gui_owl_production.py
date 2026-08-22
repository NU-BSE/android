#!/usr/bin/env python3
"""Production GUI-Owl distillation with explicit contract and grounding recovery.

This entrypoint composes the safeguards that the failed 1.5B runs showed were
needed together rather than as independent experiments:

* GUI records always receive the exact Action + <tool_call> system contract;
* GUI action classes are damped inverse-frequency balanced;
* CE/KL can concentrate gradient mass on the complete tool-call suffix;
* the LM head can be temporarily unfrozen after a structural depth cut;
* policy stages retain grounded evaluation and live-gradient clipped K3.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from android_drlqn.gui_owl_contract import normalize_gui_messages  # noqa: E402
from scripts import distill_gui_owl_policy_aligned as policy  # noqa: E402
from scripts.gui_owl_contract_loss import ContractTokenWeighter  # noqa: E402
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
    raise SystemExit("distill_gui_owl_production.py requires --student-model")

try:
    from transformers import AutoProcessor
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("install the training extra: python -m pip install -e '.[train]'") from exc

_PROCESSOR = AutoProcessor.from_pretrained(STUDENT_MODEL)
_WEIGHER = ContractTokenWeighter(
    _PROCESSOR.tokenizer,
    prefix_weight=float(os.environ.get("GUI_OWL_CONTRACT_PREFIX_WEIGHT", "1.0")),
    suffix_weight=float(os.environ.get("GUI_OWL_CONTRACT_SUFFIX_WEIGHT", "3.0")),
    focus_weight=float(os.environ.get("GUI_OWL_CONTRACT_FOCUS_WEIGHT", "6.0")),
    first_token_weight=float(os.environ.get("GUI_OWL_FIRST_TOKEN_WEIGHT", "2.0")),
    require_marker=os.environ.get("GUI_OWL_REQUIRE_TOOL_MARKER", "0") == "1",
)
UNFREEZE_LM_HEAD = os.environ.get("GUI_OWL_UNFREEZE_LM_HEAD", "0") == "1"

_ORIGINAL_LOAD = policy.base._load_jsonl_messages
_ORIGINAL_CORPUS_RECORDS = policy.base._corpus_records
_ORIGINAL_FREEZE = policy.base._freeze_for_stage


def _normalized_load(path, default_mode, limit):
    values = _ORIGINAL_LOAD(path, default_mode, limit)
    output = []
    for mode, messages in values:
        if mode == "gui_policy":
            messages = normalize_gui_messages(messages)
        output.append((mode, messages))
    return output


def _balanced_records(sources, max_records):
    return balanced_corpus_records(_ORIGINAL_CORPUS_RECORDS, sources, max_records)


def _production_freeze(student, *, freeze_embeddings, freeze_lm_head, freeze_vision):
    _ORIGINAL_FREEZE(
        student,
        freeze_embeddings=freeze_embeddings,
        freeze_lm_head=False if UNFREEZE_LM_HEAD else freeze_lm_head,
        freeze_vision=freeze_vision,
    )


def _contract_teacher_forced_terms(
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

    token_weights = _WEIGHER.weights(targets, torch_module)
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


policy.base._load_jsonl_messages = _normalized_load
policy.base._corpus_records = _balanced_records
policy.base._freeze_for_stage = _production_freeze
policy.base._teacher_forced_terms = _contract_teacher_forced_terms

print(
    "GUI-Owl production trainer: "
    f"prefix={_WEIGHER.prefix_weight:g} suffix={_WEIGHER.suffix_weight:g} "
    f"focus={_WEIGHER.focus_weight:g} first={_WEIGHER.first_token_weight:g} "
    f"unfreeze_lm_head={UNFREEZE_LM_HEAD}",
    flush=True,
)


if __name__ == "__main__":
    raise SystemExit(policy.base.main())
