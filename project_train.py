"""Project student plugin: Qwen2.5-1.5B-Instruct with lora-speedrun distillation.

``create_student`` is the ``--student-plugin`` entry point consumed by
``scripts/train_on_policy.py`` and ``scripts/train_on_policy_ollama.py``::

    python scripts/train_on_policy_ollama.py \
      --documents artifacts/normalized-documents.json \
      --student-plugin project_train:create_student \
      --environment-plugin project_lab:create_environment

The student is a text-only Qwen2.5-1.5B-Instruct policy. Screenshots stay with
the multimodal teacher; the student reasons over the accessibility labels and
the normalized OS-settings evidence in the task payload and emits one closed
JSON action per step.

Training follows the current lora-speedrun Track-1 record
(``submissions/003-pmigdal``, 1m 44s to 60.3% GSM8K on Qwen2.5-1.5B, L40S):

- LoRA r=16, alpha=32, dropout=0.0 on all seven linear projections
  (q/k/v/o/gate/up/down), bias none -- 18.5M adapter parameters;
- AdamW (fused on CUDA), betas (0.9, 0.95), no weight decay, LR 4e-4 with an
  8-step warmup and cosine decay to 5%;
- samples packed greedily into full fixed-length blocks that are moved to the
  device once (no dataloader, no collation, no attention mask);
- completion-only cross-entropy through a chunked custom autograd function so
  the full ``[tokens, vocab]`` fp32 logits tensor is never materialized.

The on-policy loss keeps the repository semantics (supervised action CE,
advantage-weighted policy term, box supervision, language CE, reference-policy
KL) mapped onto the text-only student: the action distribution is the softmax
over the six proposal-kind first-tokens at the forced ``"kind"`` position, box
supervision is the cross-entropy over the teacher ``bbox_norm`` token span,
and the reference policy is the frozen base model with LoRA adapters disabled.

Optional ``student_config`` keys (from ``config/distillation.json``)::

    model_id        HF id or local path (default Qwen/Qwen2.5-1.5B-Instruct)
    device          "cuda" | "cpu" | ... (default: CUDA when available)
    dtype           fp16/bf16/fp32 alias (Windows CUDA default: float16)
    max_new_tokens  generation budget per proposal (default 128)
    seed            master seed (default 0)
    lora            {rank, alpha, dropout, target_modules}
    speedrun        {lr, warmup_steps, min_lr_frac, total_steps, pack_len,
                     ce_chunk, adam_beta2, grad_clip}
    resume          load the latest valid adapter checkpoint (default true)
    resume_checkpoint explicit step-* directory to load instead of latest

The module imports without torch so dry-runs and plugin discovery stay cheap;
heavy dependencies load lazily inside ``create_student``.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Sequence

from android_drlqn.distillation import (
    DistillationSample,
    OnPolicyConfig,
    ProposalKind,
    RolloutObservation,
    StudentProposal,
    TeacherTarget,
)
from android_drlqn.document_curriculum import CurriculumTask
from android_drlqn.emulator_lab import EmulatorProfile

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
ENV_STUDENT_DEVICE = "ANDROID_DRLQN_STUDENT_DEVICE"
ENV_STUDENT_DTYPE = "ANDROID_DRLQN_STUDENT_DTYPE"

# lora-speedrun Track-1 record hyperparameters (submissions/003-pmigdal).
SPEEDRUN_LORA = {
    "rank": 16,
    "alpha": 32,
    "dropout": 0.0,
    "target_modules": (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ),
}
SPEEDRUN_TRAIN = {
    "lr": 4e-4,
    "warmup_steps": 8,
    "min_lr_frac": 0.05,
    "total_steps": 512,
    "pack_len": 1024,
    "ce_chunk": 2048,
    "adam_beta2": 0.95,
    "grad_clip": 1.0,
}

ACTION_PREFIX = '{"kind": "'
_MAX_SETTINGS_LINES = 12
_MAX_VISIBLE_LABELS = 24
_MAX_SUMMARY_CHARS = 400
_MAX_MESSAGE_WORDS = 40

_SYSTEM_PROMPT = (
    "You are a student policy for Android settings navigation. Reply with "
    "exactly one JSON object and nothing else:\n"
    '{"kind": "tap|scroll|back|wait|ask_confirmation|explain", '
    '"target_label": string|null, "bbox_norm": [x0,y0,x1,y1]|null, '
    '"message": string}\n'
    "Rules: choose exactly one next action toward the requested setting; use "
    "tap with target_label copied from the visible labels when the target is "
    "on screen; use scroll when it may be below the fold; use explain only for "
    "informational questions; never invent labels; normalized coordinates stay "
    "within [0,1]; never claim an action succeeded; never output adb, shell, "
    f"root or accessibility commands; keep message under {_MAX_MESSAGE_WORDS} words."
)

_KINDS = tuple(ProposalKind)


# ---------------------------------------------------------------------------
# Pure helpers (importable and testable without torch)
# ---------------------------------------------------------------------------


def resolve_student_runtime(
    config: Mapping[str, object],
    *,
    cuda_available: bool,
    windows_host: bool,
    environ: Mapping[str, str] | None = None,
) -> tuple[str, str | None]:
    """Resolve device/dtype without importing torch at module import time.

    Explicit student config wins over environment variables. Windows CUDA
    defaults to float16 because it is the stable low-memory choice for RTX
    consumer GPUs; other hosts keep the model class' existing automatic dtype.
    """
    environment = os.environ if environ is None else environ
    device = str(
        config.get("device")
        or environment.get(ENV_STUDENT_DEVICE)
        or ("cuda" if cuda_available else "cpu")
    ).strip()
    if not device:
        raise ValueError("student device must not be empty")
    uses_cuda = device.casefold().startswith("cuda")
    if uses_cuda and not cuda_available:
        raise RuntimeError(
            f"student device {device!r} requires a CUDA-enabled PyTorch build "
            "and a visible NVIDIA GPU"
        )

    raw_dtype = config.get("dtype") or environment.get(ENV_STUDENT_DTYPE)
    dtype = str(raw_dtype).strip() if raw_dtype is not None else None
    if dtype == "":
        dtype = None
    if dtype is None and windows_host and uses_cuda:
        dtype = "float16"
    return device, dtype


def canonical_action_json(
    kind: ProposalKind,
    target_label: str | None,
    bbox_norm: Sequence[float] | None,
    message: str,
) -> str:
    """Serialize one action with a fixed key order for train/infer parity."""
    return json.dumps(
        {
            "kind": kind.value,
            "target_label": target_label or None,
            "bbox_norm": [float(value) for value in bbox_norm] if bbox_norm else None,
            "message": message,
        },
        ensure_ascii=False,
    )


def teacher_completion_text(target: TeacherTarget) -> str:
    """Render the teacher target as the completion the student must imitate.

    The returned text continues the forced ``ACTION_PREFIX``: it starts with
    the kind value and keeps the canonical key order.
    """
    return canonical_action_json(
        target.preferred_kind,
        target.target_label,
        target.bbox_norm,
        target.response,
    )[len(ACTION_PREFIX) :]


def sanitize_bbox(value: object) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        box = (float(value[0]), float(value[1]), float(value[2]), float(value[3]))
    except (TypeError, ValueError):
        return None
    x0, y0, x1, y1 = box
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        return None
    return box


def parse_action_json(text: str) -> dict[str, object] | None:
    """Parse the first complete JSON object in ``text``; None on failure."""
    start = text.find("{")
    if start < 0:
        return None
    try:
        value, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def build_messages(
    task: CurriculumTask,
    profile: EmulatorProfile,
    observation: RolloutObservation,
) -> list[dict[str, str]]:
    """Chat prompt from the task, the emulator scope and the current screen."""
    labels = [str(label) for label in observation.visible_labels[:_MAX_VISIBLE_LABELS]]
    summary = json.dumps(dict(observation.accessibility_summary), ensure_ascii=False)[
        :_MAX_SUMMARY_CHARS
    ]
    lines = [
        f"Task: {task.prompt}",
        f"Task kind: {task.kind.value}; risk: {task.risk_level}",
        (
            f"OS: {profile.os_family} {profile.os_release} "
            f"(Android {profile.android_version}, API {profile.api_level})"
        ),
        (
            f"Screen: {observation.package}/{observation.activity} "
            f"step {observation.step_index}"
        ),
        f"Visible labels: {', '.join(labels) if labels else '(none)'}",
        f"Accessibility: {summary or '{}'}"[:_MAX_SUMMARY_CHARS + 15],
    ]
    settings = _settings_context_lines(task)
    if settings:
        lines.append("Known settings for this OS:")
        lines.extend(settings)
    if task.expected_output:
        lines.append(
            "Expected: " + json.dumps(dict(task.expected_output), ensure_ascii=False)
        )
    if task.kind.value in ("explain_setting", "informational_qa"):
        lines.append('This is an informational task: answer with kind "explain".')
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(lines)},
    ]


def _settings_context_lines(task: CurriculumTask) -> list[str]:
    evidence = task.evidence if isinstance(task.evidence, Mapping) else {}
    context = evidence.get("os_settings_context")
    if not isinstance(context, Mapping):
        return []
    records = context.get("settings")
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        return []
    lines = []
    for record in list(records)[:_MAX_SETTINGS_LINES]:
        if not isinstance(record, Mapping):
            continue
        name = str(record.get("display_name") or record.get("setting_id") or "")
        path = " > ".join(str(item) for item in record.get("ui_path") or ())
        if name:
            lines.append(f"- {name}: {path}" if path else f"- {name}")
    return lines


def speedrun_lr(step: int, config: Mapping[str, float]) -> float:
    """Warmup + cosine-to-floor schedule from the speedrun record."""
    base = float(config["lr"])
    warmup = int(config["warmup_steps"])
    total = int(config["total_steps"])
    floor = float(config["min_lr_frac"])
    if step < 0:
        raise ValueError("step must be non-negative")
    if base <= 0 or warmup < 0 or total < 1 or not 0 <= floor <= 1:
        raise ValueError("invalid speedrun learning-rate configuration")
    total = max(warmup + 1, total)
    if warmup and step < warmup:
        return base * (step + 1) / warmup
    fraction = min(1.0, (step - warmup) / max(1, total - warmup))
    return base * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * fraction)))


@dataclass(frozen=True)
class PackedExample:
    """One tokenized teacher completion with its action/grounding anchors.

    ``kind_token``/``student_kind_token`` are vocabulary ids of the first
    subword of the teacher/student proposal kind. ``box_span`` delimits the
    ``bbox_norm`` value tokens within ``completion_ids``.
    """

    prompt_ids: tuple[int, ...]
    completion_ids: tuple[int, ...]
    kind_token: int
    student_kind_token: int
    advantage: float
    box_span: tuple[int, int] | None


def pack_examples(
    examples: Sequence[PackedExample],
    pack_len: int,
    pad_id: int,
    rng: random.Random,
) -> list[dict[str, object]]:
    """Seed-shuffled greedy packing into full fixed-length blocks.

    Same scheme as the record: full blocks only, so no attention mask is ever
    needed. The final partial block is right-padded with ``pad_id`` and
    ``-100`` labels instead of being dropped -- on-policy batches are small
    and every teacher correction matters. If one example exceeds ``pack_len``
    the prompt head (system prompt side) is truncated; the completion is never
    cut, and examples whose completion alone fills a block are skipped.

    Block layout: ``input_ids``/``labels`` cover the full block;
    ``kind_positions`` holds ``(shifted_pos, teacher_kind, student_kind,
    advantage)`` where ``shifted_pos`` indexes the hidden state that predicts
    the kind token; ``box_spans`` holds shifted ``(start, end)`` ranges that
    supervise the bbox value tokens.
    """
    if pack_len < 8:
        raise ValueError("pack_len must be at least 8")
    order = list(range(len(examples)))
    rng.shuffle(order)
    blocks: list[dict[str, object]] = []
    buf_ids: list[int] = []
    buf_labels: list[int] = []
    buf_kind: list[tuple[int, int, int, float]] = []
    buf_box: list[tuple[int, int]] = []

    def flush() -> None:
        if not buf_ids:
            return
        pad = pack_len - len(buf_ids)
        blocks.append(
            {
                "input_ids": tuple(buf_ids + [pad_id] * pad),
                "labels": tuple(buf_labels + [-100] * pad),
                "kind_positions": tuple(buf_kind),
                "box_spans": tuple(buf_box),
            }
        )

    for index in order:
        example = examples[index]
        if len(example.completion_ids) >= pack_len - 1:
            continue
        prompt_ids = list(example.prompt_ids)
        overflow = len(prompt_ids) + len(example.completion_ids) - pack_len
        if overflow > 0:
            prompt_ids = prompt_ids[overflow:]
        if not prompt_ids:
            continue
        prompt_len = len(prompt_ids)
        ids = prompt_ids + list(example.completion_ids)
        labels = [-100] * prompt_len + list(example.completion_ids)
        if len(buf_ids) + len(ids) > pack_len:
            flush()
            buf_ids, buf_labels, buf_kind, buf_box = [], [], [], []
        base = len(buf_ids)
        buf_ids += ids
        buf_labels += labels
        # shifted index base+prompt_len-1 predicts the first completion token.
        buf_kind.append(
            (
                base + prompt_len - 1,
                example.kind_token,
                example.student_kind_token,
                example.advantage,
            )
        )
        if example.box_span is not None:
            start, end = example.box_span
            buf_box.append(
                (base + prompt_len + start - 1, base + prompt_len + end - 1)
            )
    flush()
    return blocks


# ---------------------------------------------------------------------------
# Torch-dependent pieces, loaded lazily
# ---------------------------------------------------------------------------


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "project_train requires the training extra: pip install -e '.[train]'"
        ) from exc
    return torch


@contextmanager
def _adapters_disabled(peft_model):
    """Temporarily disable LoRA adapters across supported peft versions."""
    if hasattr(peft_model, "disable_adapter"):
        with peft_model.disable_adapter():
            yield
        return
    disable = getattr(peft_model, "disable_adapter_layers", None)
    enable = getattr(peft_model, "enable_adapter_layers", None)
    if not callable(disable) or not callable(enable):
        raise RuntimeError("installed peft version cannot disable adapters safely")
    disable()
    try:
        yield
    finally:
        enable()


@lru_cache(maxsize=1)
def _chunked_ce_class():
    """Chunked completion-only cross-entropy from the speedrun record.

    Computes ``sum(lse - gold) / n`` over the labeled positions and forms
    ``d(loss)/d(hidden) = (softmax - 1) @ W`` analytically inside forward, so
    the full ``[tokens, vocab]`` fp32 logits tensor is never materialized.
    """
    torch = _require_torch()

    class ChunkedCompletionCE(torch.autograd.Function):
        @staticmethod
        def forward(ctx, hidden, weight, labels, chunk):
            n = labels.numel()
            grad_hidden = torch.empty_like(hidden)
            total = hidden.new_zeros((), dtype=torch.float32)
            for i in range(0, n, chunk):
                h_c = hidden[i : i + chunk]
                y_c = labels[i : i + chunk]
                logits = (h_c @ weight.T).float()
                lse = torch.logsumexp(logits, dim=-1)
                gold = logits.gather(1, y_c[:, None]).squeeze(1)
                total += (lse - gold).sum()
                logits.sub_(lse[:, None]).exp_()  # in-place softmax
                logits[torch.arange(y_c.numel(), device=labels.device), y_c] -= 1.0
                grad_hidden[i : i + chunk] = logits.to(hidden.dtype) @ weight
            ctx.save_for_backward(grad_hidden)
            ctx.n = n
            return total / n

        @staticmethod
        def backward(ctx, grad_out):
            (grad_hidden,) = ctx.saved_tensors
            return grad_hidden * (grad_out / ctx.n), None, None, None

    return ChunkedCompletionCE


class QwenSpeedrunStudent:
    """Text-only StudentPolicy backed by Qwen2.5-1.5B-Instruct + LoRA."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        *,
        device: str | None = None,
        dtype: str | None = None,
        max_new_tokens: int = 128,
        seed: int = 0,
        lora: Mapping[str, object] | None = None,
    ) -> None:
        torch = _require_torch()
        try:
            from peft import LoraConfig, get_peft_model
            from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
        except ImportError as exc:
            raise RuntimeError(
                "project_train requires transformers and peft: pip install -e '.[train]'"
            ) from exc

        set_seed(seed)
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = int(max_new_tokens)
        self._policy_version = 0

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        dtype_name = str(dtype).lower() if dtype else None
        dtype_map = {
            "float32": torch.float32,
            "fp32": torch.float32,
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        if dtype_name:
            if dtype_name not in dtype_map:
                raise ValueError(f"unsupported dtype: {dtype}")
            torch_dtype = dtype_map[dtype_name]
        else:
            torch_dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch_dtype, attn_implementation="sdpa"
            )
        except TypeError:
            # transformers 5 renamed torch_dtype to dtype.
            model = AutoModelForCausalLM.from_pretrained(
                model_id, dtype=torch_dtype, attn_implementation="sdpa"
            )

        lora_cfg = dict(SPEEDRUN_LORA)
        if lora:
            lora_cfg.update(lora)
        self.lora_config = dict(lora_cfg)
        self.model = get_peft_model(
            model.to(self.device),
            LoraConfig(
                r=int(lora_cfg["rank"]),
                lora_alpha=int(lora_cfg["alpha"]),
                lora_dropout=float(lora_cfg["dropout"]),
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=list(lora_cfg["target_modules"]),
            ),
        )
        self.model.config.use_cache = False
        self.model.train()

        eos_id = self.tokenizer.eos_token_id
        if eos_id is None:
            eos_id = getattr(self.model.config, "eos_token_id", None)
        if isinstance(eos_id, (list, tuple)):
            eos_id = eos_id[0] if eos_id else None
        if eos_id is None:
            raise RuntimeError("student tokenizer/model does not define an EOS token")
        self.eos_id = int(eos_id)
        self.kind_first_tokens = self._resolve_kind_tokens()

    # -- StudentPolicy protocol -------------------------------------------

    @property
    def policy_version(self) -> int:
        return self._policy_version

    def propose(
        self,
        task: CurriculumTask,
        profile: EmulatorProfile,
        observation: RolloutObservation,
    ) -> StudentProposal:
        torch = _require_torch()
        prompt_ids = self.encode_prompt(task, profile, observation)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.inference_mode():
                hidden = self.hidden_states(input_ids)
                logits = (hidden[0, -1] @ self.lm_head_weight().T).float()
                kind_logits = logits[self._kind_token_ids]
                probabilities = torch.softmax(kind_logits, dim=-1)
                generated = self.model.generate(
                    input_ids=input_ids,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    eos_token_id=self.eos_id,
                    pad_token_id=self.eos_id,
                )
        finally:
            if was_training:
                self.model.train()
        continuation = generated[0, input_ids.shape[1] :]
        text = ACTION_PREFIX + self.tokenizer.decode(
            continuation, skip_special_tokens=True
        )
        probabilities_cpu = probabilities.detach().float().cpu()
        distribution = tuple(float(value) for value in probabilities_cpu)
        parsed = parse_action_json(text)
        valid_kinds = {kind.value for kind in _KINDS}
        if parsed is None or str(parsed.get("kind")) not in valid_kinds:
            index = int(probabilities_cpu.argmax())
            return StudentProposal(
                kind=_KINDS[index],
                target_label=None,
                bbox_norm=None,
                message="(unparseable student output; fell back to argmax action)",
                confidence=min(float(probabilities_cpu[index]), 0.2),
                action_distribution=distribution,
                log_probability=math.log(max(float(probabilities_cpu[index]), 1e-9)),
                policy_version=self.policy_version,
            )
        kind = ProposalKind(str(parsed["kind"]))
        index = _KINDS.index(kind)
        label = parsed.get("target_label")
        message = str(parsed.get("message") or "")
        return StudentProposal(
            kind=kind,
            target_label=str(label) if label else None,
            bbox_norm=sanitize_bbox(parsed.get("bbox_norm")),
            message=message or (str(label) if label else task.prompt),
            confidence=float(probabilities_cpu[index]),
            action_distribution=distribution,
            log_probability=math.log(max(float(probabilities_cpu[index]), 1e-9)),
            policy_version=self.policy_version,
        )

    # -- helpers shared with the updater -----------------------------------

    def encode_prompt(
    self,
    task: CurriculumTask,
    profile: EmulatorProfile,
    observation: RolloutObservation,
) -> list[int]:
        """Encode a one-step Android action prompt with manual navigation rules."""

        messages = build_messages(task, profile, observation)

        expected = (
            dict(task.expected_output)
            if isinstance(task.expected_output, Mapping)
            else {}
        )
        evidence = (
            dict(task.evidence)
            if isinstance(task.evidence, Mapping)
            else {}
        )

        raw_ui_path = expected.get("ui_path") or evidence.get("ui_path") or ()

        if isinstance(raw_ui_path, Sequence) and not isinstance(
            raw_ui_path,
            (str, bytes),
        ):
            ui_path = [
                str(segment).strip()
                for segment in raw_ui_path
                if str(segment).strip()
            ]
        else:
            ui_path = []

        visible_labels = [
            str(label).strip()
            for label in observation.visible_labels
            if str(label).strip()
        ]
        visible_by_name = {
            label.casefold(): label
            for label in visible_labels
        }

        # Select the deepest route segment currently visible.
        #
        # Example:
        # ui_path:
        #   Settings > Network & internet > Internet
        #
        # On the Settings root:
        #   next_visible_segment = "Network & internet"
        #
        # On the Network & internet screen:
        #   next_visible_segment = "Internet"
        next_visible_segment = next(
            (
                visible_by_name[segment.casefold()]
                for segment in reversed(ui_path)
                if segment.casefold() in visible_by_name
                and segment.casefold() != "settings"
            ),
            None,
        )

        informational = task.kind.value in {
            "explain_setting",
            "informational_qa",
        }

        if ui_path and not informational:
            navigation_policy = {
                "mode": "manual_ui_path",
                "ui_path": ui_path,
                "next_visible_segment": next_visible_segment,
                "search_allowed": False,
                "rules": [
                    "Never tap a Settings search icon.",
                    "Never open a Settings search field.",
                    "Never type the destination setting name.",
                    "Navigate only through visible categories and rows.",
                    "Choose exactly one next action.",
                    (
                        "Tap next_visible_segment when it is not null and is a "
                        "visible actionable row."
                    ),
                    (
                        "Scroll when the next route segment is not visible but may "
                        "be below the fold."
                    ),
                    (
                        "Use back only when the current screen is outside the "
                        "required ui_path."
                    ),
                    "Do not jump directly to the final destination.",
                ],
            }

            # build_messages() returns a system message followed by a user message.
            # Strengthen the system instruction because it has higher priority.
            system_message = dict(messages[0])
            system_message["content"] = (
                system_message["content"]
                + "\nManual navigation is mandatory. Settings Search is forbidden. "
                "Do not tap a search icon, open a search field, or type the target "
                "setting. Follow the supplied ui_path through visible UI rows."
            )
            messages[0] = system_message

            user_message = dict(messages[-1])
            user_message["content"] = (
                user_message["content"]
                + "\nNavigation policy: "
                + json.dumps(
                    navigation_policy,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
            messages[-1] = user_message

        rendered = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )

        if isinstance(rendered, Mapping):
            # transformers >= 5 may return BatchEncoding.
            rendered = rendered["input_ids"]

        if rendered and isinstance(rendered[0], (list, tuple)):
            rendered = rendered[0]

        prompt_ids = [int(token) for token in rendered]

        # Generation starts inside the canonical action JSON.
        return prompt_ids + self._tokenize(ACTION_PREFIX)

    def encode_sample(self, sample: DistillationSample) -> PackedExample:
        prompt_ids = self.encode_prompt(sample.task, sample.profile, sample.observation)
        completion = teacher_completion_text(sample.judgement.target)
        kind = sample.judgement.target.preferred_kind
        if not completion.startswith(kind.value):
            raise ValueError("teacher completion does not start with the kind value")
        kind_ids = self._tokenize(kind.value)
        rest = completion[len(kind.value) :]
        rest_box_span, rest_ids = self._rest_with_box_span(
            rest, sample.judgement.target
        )
        box_span = None
        if rest_box_span is not None:
            start, end = rest_box_span
            box_span = (len(kind_ids) + start, len(kind_ids) + end)
        return PackedExample(
            prompt_ids=tuple(prompt_ids),
            completion_ids=tuple(kind_ids + rest_ids + [self.eos_id]),
            kind_token=kind_ids[0],
            student_kind_token=int(self.kind_first_tokens[sample.proposal.kind]),
            advantage=float(sample.advantage),
            box_span=box_span,
        )

    def hidden_states(self, input_ids):
        """Last hidden states through the LoRA-injected transformer."""
        return self.model.base_model.model.model(
            input_ids=input_ids, use_cache=False
        ).last_hidden_state

    def lm_head_weight(self):
        return self.model.base_model.model.lm_head.weight

    def trainable_parameters(self) -> list:
        return [param for param in self.model.parameters() if param.requires_grad]

    def save_checkpoint(self, directory: str | Path, step: int) -> str:
        directory = Path(directory) / f"step-{step:08d}"
        directory.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(directory)
        candidates = (
            directory / "adapter_model.safetensors",
            directory / "adapter_model.bin",
        )
        adapter = next((path for path in candidates if path.is_file()), None)
        if adapter is None:
            raise RuntimeError("peft did not write an adapter checkpoint")
        digest = hashlib.sha256(adapter.read_bytes()).hexdigest()[:24]
        (directory / "checkpoint.id").write_text(digest, encoding="utf-8")
        return digest

    def _tokenize(self, text: str) -> list[int]:
        return list(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def _rest_with_box_span(self, rest: str, target: TeacherTarget):
        box = sanitize_bbox(target.bbox_norm)
        if box is None:
            return None, self._tokenize(rest)
        marker = json.dumps([float(value) for value in box])
        start = rest.find(marker)
        if start < 0:
            return None, self._tokenize(rest)
        before = self._tokenize(rest[:start])
        value_ids = self._tokenize(marker)
        after = self._tokenize(rest[start + len(marker) :])
        return (len(before), len(before) + len(value_ids)), before + value_ids + after

    def _resolve_kind_tokens(self) -> dict[ProposalKind, int]:
        mapping: dict[ProposalKind, int] = {}
        for kind in _KINDS:
            ids = self._tokenize(kind.value)
            if not ids:
                raise RuntimeError(f"tokenizer produced no tokens for {kind.value!r}")
            mapping[kind] = ids[0]
        first_tokens = list(mapping.values())
        if len(set(first_tokens)) != len(first_tokens):
            raise RuntimeError(
                "proposal kinds share a first subword token; the closed action "
                "head requires distinct first tokens"
            )
        self._kind_token_ids = first_tokens
        return mapping


class SpeedrunLoraUpdater:
    """PolicyUpdater implementing the lora-speedrun record training loop.

    One ``update`` call packs the rollout batch into full blocks, accumulates
    the combined distillation loss over all blocks and performs a single AdamW
    step under the record's warmup/cosine schedule, then writes a new LoRA
    adapter checkpoint and bumps the student policy version.
    """

    def __init__(
        self,
        student: QwenSpeedrunStudent,
        checkpoint_dir: str | Path,
        *,
        speedrun: Mapping[str, float] | None = None,
        seed: int = 0,
    ) -> None:
        torch = _require_torch()
        self.student = student
        self.checkpoint_dir = Path(checkpoint_dir)
        self.config = dict(SPEEDRUN_TRAIN)
        if speedrun:
            self.config.update({key: float(value) for key, value in speedrun.items()})
        self.rng = random.Random(seed)
        self.step_count = 0
        if int(self.config["pack_len"]) < 8:
            raise ValueError("speedrun.pack_len must be at least 8")
        if int(self.config["ce_chunk"]) < 1:
            raise ValueError("speedrun.ce_chunk must be positive")
        if not 0 < float(self.config["adam_beta2"]) < 1:
            raise ValueError("speedrun.adam_beta2 must be in (0, 1)")
        parameters = student.trainable_parameters()
        if not parameters:
            raise RuntimeError("student has no trainable LoRA parameters")
        optimizer_kwargs = {
            "lr": float(self.config["lr"]),
            "betas": (0.9, float(self.config["adam_beta2"])),
            "weight_decay": 0.0,
        }
        if str(student.device).startswith("cuda"):
            optimizer_kwargs["fused"] = True
        try:
            self.optimizer = torch.optim.AdamW(parameters, **optimizer_kwargs)
        except TypeError:
            optimizer_kwargs.pop("fused", None)
            self.optimizer = torch.optim.AdamW(parameters, **optimizer_kwargs)

    def update(self, samples: Sequence[DistillationSample], config: OnPolicyConfig) -> str:
        torch = _require_torch()
        trainable = [sample for sample in samples if sample.judgement.should_update]
        if not trainable:
            # A fully rejected batch (e.g. teacher/critic disagreement) must not
            # move weights; still return the current checkpoint id.
            return self.student.save_checkpoint(self.checkpoint_dir, self.step_count)

        examples = [self.student.encode_sample(sample) for sample in trainable]
        pack_len = int(self.config["pack_len"])
        blocks = pack_examples(examples, pack_len, self.student.eos_id, self.rng)
        if not blocks:
            return self.student.save_checkpoint(self.checkpoint_dir, self.step_count)
        device = self.student.device
        chunked_ce = _chunked_ce_class()
        ce_chunk = int(self.config["ce_chunk"])
        lm_weight = self.student.lm_head_weight()
        kind_ids = torch.tensor(
            self.student._kind_token_ids, dtype=torch.long, device=device
        )

        for group in self.optimizer.param_groups:
            group["lr"] = speedrun_lr(self.step_count, self.config)
        self.optimizer.zero_grad(set_to_none=True)

        for block in blocks:
            input_ids = torch.tensor(
                [block["input_ids"]], dtype=torch.long, device=device
            )
            labels = torch.tensor([block["labels"]], dtype=torch.long, device=device)
            hidden = self.student.hidden_states(input_ids)
            shifted_h = hidden[:, :-1, :].reshape(-1, hidden.shape[-1])
            shifted_y = labels[:, 1:].reshape(-1)
            keep = (shifted_y != -100).nonzero(as_tuple=True)[0]

            loss = hidden.sum() * 0.0
            if keep.numel() and config.language_weight > 0:
                loss = loss + config.language_weight * chunked_ce.apply(
                    shifted_h[keep], lm_weight, shifted_y[keep], ce_chunk
                )

            kind_positions = block["kind_positions"]
            if kind_positions:
                positions = torch.tensor(
                    [item[0] for item in kind_positions], dtype=torch.long, device=device
                )
                logits = (shifted_h[positions] @ lm_weight.T).float()
                kind_logits = logits.index_select(1, kind_ids)
                log_probs = torch.log_softmax(kind_logits, dim=-1)
                teacher_index = torch.tensor(
                    [self.student._kind_token_ids.index(item[1]) for item in kind_positions],
                    dtype=torch.long,
                    device=device,
                )
                supervised = -log_probs.gather(1, teacher_index[:, None]).mean()
                student_index = torch.tensor(
                    [self.student._kind_token_ids.index(item[2]) for item in kind_positions],
                    dtype=torch.long,
                    device=device,
                )
                advantages = torch.tensor(
                    [item[3] for item in kind_positions],
                    dtype=torch.float32,
                    device=device,
                )
                policy = -(
                    log_probs.gather(1, student_index[:, None]).squeeze(1) * advantages
                ).mean()
                loss = loss + config.supervised_weight * supervised
                loss = loss + config.policy_weight * policy

                if config.kl_weight > 0:
                    with torch.no_grad():
                        with _adapters_disabled(self.student.model):
                            ref_hidden = self.student.hidden_states(input_ids)
                        ref_shifted = ref_hidden[:, :-1, :].reshape(
                            -1, ref_hidden.shape[-1]
                        )[positions]
                        ref_logits = (ref_shifted @ lm_weight.T).float()
                        ref_probs = torch.softmax(
                            ref_logits.index_select(1, kind_ids), dim=-1
                        )
                    kl = torch.nn.functional.kl_div(
                        log_probs, ref_probs, reduction="batchmean"
                    )
                    loss = loss + config.kl_weight * kl

            box_spans = block["box_spans"]
            if box_spans and config.grounding_weight > 0:
                gather = torch.cat(
                    [
                        torch.arange(start, end, dtype=torch.long, device=device)
                        for start, end in box_spans
                    ]
                )
                grounding = chunked_ce.apply(
                    shifted_h[gather], lm_weight, shifted_y[gather], ce_chunk
                )
                loss = loss + config.grounding_weight * grounding

            (loss / len(blocks)).backward()

        torch.nn.utils.clip_grad_norm_(
            self.student.trainable_parameters(), float(self.config["grad_clip"])
        )
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.step_count += 1
        self.student._policy_version += 1
        return self.student.save_checkpoint(self.checkpoint_dir, self.step_count)


# ---------------------------------------------------------------------------
# Checkpoint discovery / resume
# ---------------------------------------------------------------------------

_CHECKPOINT_DIRECTORY = re.compile(r"step-(\d{8,})$")


def _config_bool(value: object, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().casefold()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"boolean configuration expected, got {value!r}")


def _adapter_path(directory: Path) -> Path | None:
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        candidate = directory / name
        if candidate.is_file():
            return candidate
    return None


def _checkpoint_step(directory: Path) -> int:
    match = _CHECKPOINT_DIRECTORY.fullmatch(directory.name)
    if match is None:
        raise ValueError(
            f"resume checkpoint directory must be named step-XXXXXXXX: {directory}"
        )
    return int(match.group(1))


def _valid_checkpoint(directory: Path) -> bool:
    try:
        _checkpoint_step(directory)
    except ValueError:
        return False
    return _adapter_path(directory) is not None


def _latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    if not checkpoint_dir.is_dir():
        return None
    candidates = [
        path
        for path in checkpoint_dir.iterdir()
        if path.is_dir() and _valid_checkpoint(path)
    ]
    return max(candidates, key=_checkpoint_step) if candidates else None


def _resolve_resume_checkpoint(
    checkpoint_dir: Path,
    configured: object,
) -> Path | None:
    if configured is None or not str(configured).strip():
        return _latest_checkpoint(checkpoint_dir)
    directory = Path(str(configured)).expanduser()
    if not directory.is_dir():
        raise FileNotFoundError(f"resume checkpoint is not a directory: {directory}")
    if not _valid_checkpoint(directory):
        raise RuntimeError(
            f"resume checkpoint has no valid adapter_model file: {directory}"
        )
    return directory


def _load_adapter_state(adapter: Path) -> Mapping[str, object]:
    if adapter.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise RuntimeError(
                "safetensors is required to resume this adapter checkpoint"
            ) from exc
        return load_file(str(adapter), device="cpu")

    torch = _require_torch()
    try:
        return torch.load(adapter, map_location="cpu", weights_only=True)
    except TypeError:  # torch versions before weights_only
        return torch.load(adapter, map_location="cpu")


def _restore_checkpoint(
    student: QwenSpeedrunStudent,
    directory: Path,
) -> tuple[int, str]:
    adapter = _adapter_path(directory)
    if adapter is None:
        raise RuntimeError(f"adapter checkpoint is missing in {directory}")
    digest = hashlib.sha256(adapter.read_bytes()).hexdigest()[:24]
    identifier = directory / "checkpoint.id"
    expected = (
        identifier.read_text(encoding="utf-8").strip()
        if identifier.is_file()
        else ""
    )
    if expected and expected != digest:
        raise RuntimeError(
            f"checkpoint digest mismatch for {directory}: expected {expected}, got {digest}"
        )

    try:
        from peft import set_peft_model_state_dict
    except ImportError as exc:
        raise RuntimeError("peft is required to resume an adapter checkpoint") from exc
    state = _load_adapter_state(adapter)
    try:
        set_peft_model_state_dict(student.model, state, adapter_name="default")
    except TypeError:  # older peft versions
        set_peft_model_state_dict(student.model, state)

    step = _checkpoint_step(directory)
    student._policy_version = step
    student.model.train()
    return step, digest


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------


def create_student(
    student_config: Mapping[str, object],
    output_dir: str | Path,
) -> tuple[QwenSpeedrunStudent, SpeedrunLoraUpdater]:
    """``--student-plugin`` factory: ``(student, updater)`` for the distillers."""
    config = dict(student_config or {})
    model_id = str(config.get("model_id") or "")
    if not model_id:
        candidate = str(config.get("llm") or "")
        model_id = candidate if "/" in candidate else DEFAULT_MODEL_ID
    seed = int(config.get("seed", 0))
    lora = config.get("lora")
    torch = _require_torch()
    device, dtype = resolve_student_runtime(
        config,
        cuda_available=bool(torch.cuda.is_available()),
        windows_host=os.name == "nt",
    )
    accelerator = "cpu"
    if device.casefold().startswith("cuda"):
        try:
            accelerator = str(torch.cuda.get_device_name(0))
        except Exception:
            accelerator = "CUDA GPU"
    print(
        f"[STUDENT] device={device} dtype={dtype or 'auto'} "
        f"accelerator={accelerator}",
        file=sys.stderr,
        flush=True,
    )
    student = QwenSpeedrunStudent(
        model_id,
        device=device,
        dtype=dtype,
        max_new_tokens=int(config.get("max_new_tokens", 128)),
        seed=seed,
        lora=lora if isinstance(lora, Mapping) else None,
    )
    checkpoint_dir = Path(output_dir) / "checkpoints"
    resume_directory = None
    resume_step = 0
    resume_digest = ""
    if _config_bool(config.get("resume"), True):
        resume_directory = _resolve_resume_checkpoint(
            checkpoint_dir, config.get("resume_checkpoint")
        )
        if resume_directory is not None:
            resume_step, resume_digest = _restore_checkpoint(
                student, resume_directory
            )

    speedrun = config.get("speedrun")
    updater = SpeedrunLoraUpdater(
        student,
        checkpoint_dir,
        speedrun=speedrun if isinstance(speedrun, Mapping) else None,
        seed=seed,
    )
    updater.step_count = resume_step
    if resume_directory is not None:
        print(
            f"[RESUME] checkpoint={resume_directory} step={resume_step} "
            f"id={resume_digest} optimizer_state=fresh",
            file=sys.stderr,
            flush=True,
        )
    return student, updater


__all__ = [
    "ACTION_PREFIX",
    "DEFAULT_MODEL_ID",
    "ENV_STUDENT_DEVICE",
    "ENV_STUDENT_DTYPE",
    "SPEEDRUN_LORA",
    "SPEEDRUN_TRAIN",
    "PackedExample",
    "QwenSpeedrunStudent",
    "SpeedrunLoraUpdater",
    "build_messages",
    "canonical_action_json",
    "create_student",
    "pack_examples",
    "parse_action_json",
    "resolve_student_runtime",
    "sanitize_bbox",
    "speedrun_lr",
    "teacher_completion_text",
]
