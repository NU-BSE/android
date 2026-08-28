"""Default GUI-Owl-1.5-2B-Instruct student for Android settings training.

The student follows the native GUI-Owl interaction contract rather than the
legacy text-only action JSON contract:

* multimodal current/recent observations;
* compressed action conclusions for older history;
* one concise ``Action:`` conclusion followed by one ``<tool_call>``;
* mobile coordinates represented on the model's 0..1000 scale.

The 2B Instruct model is deliberately trained without hidden chain-of-thought.
Its concise action conclusion is user-visible and auditable. The updater keeps
teacher SFT/grounding losses, an advantage-weighted on-policy sequence loss,
and a frozen-base KL term. Exact inference token ids are retained in a local
sidecar so the policy loss evaluates the same sampled token event.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from contextlib import contextmanager
from dataclasses import dataclass
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

DEFAULT_MODEL_ID = "mPLUG/GUI-Owl-1.5-2B-Instruct"
COORDINATE_SCALE = 1000
DEFAULT_HISTORY_N = 4

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
_ACTION_RE = re.compile(r"(?:^|\n)Action:\s*(.*?)(?=\n<tool_call>|$)", re.S)

_SYSTEM_PROMPT = """You are GUI-Owl-1.5 operating Android Settings.
Return exactly one concise visible action conclusion and one tool call:
Action: <what to do next and why, without hidden chain-of-thought>
<tool_call>{\"name\": \"mobile.click|mobile.swipe|mobile.system_button|mobile.wait|mobile.answer|mobile.interact\", \"arguments\": {...}}</tool_call>
Use integer screen coordinates on a 0..1000 scale. Choose one next action only.
Never claim success before the environment verifies the transition. Never emit
ADB, shell, root, AccessibilityService, DevicePolicyManager, or Settings
Provider commands. Ask for confirmation before a sensitive setting change."""


@dataclass(frozen=True)
class RolloutTrace:
    messages: tuple[dict[str, object], ...]
    generated_token_ids: tuple[int, ...]


def _text_content(text: str) -> list[dict[str, str]]:
    return [{"type": "text", "text": text}]


def _image_content(path: str) -> dict[str, str]:
    return {"type": "image", "image": path}


def _settings_lines(task: CurriculumTask, limit: int = 12) -> list[str]:
    evidence = task.evidence if isinstance(task.evidence, Mapping) else {}
    context = evidence.get("os_settings_context")
    if not isinstance(context, Mapping):
        return []
    records = context.get("settings")
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        return []
    lines: list[str] = []
    for record in list(records)[:limit]:
        if not isinstance(record, Mapping):
            continue
        name = str(record.get("display_name") or record.get("setting_id") or "").strip()
        path = " > ".join(str(value) for value in record.get("ui_path") or ())
        if name:
            lines.append(f"- {name}: {path}" if path else f"- {name}")
    return lines


def _task_ui_path(task: CurriculumTask) -> tuple[str, ...]:
    expected = task.expected_output if isinstance(task.expected_output, Mapping) else {}
    evidence = task.evidence if isinstance(task.evidence, Mapping) else {}
    raw = expected.get("ui_path") or evidence.get("ui_path") or ()
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return ()
    return tuple(str(value).strip() for value in raw if str(value).strip())


def build_gui_owl_messages(
    task: CurriculumTask,
    profile: EmulatorProfile,
    observation: RolloutObservation,
    history: Sequence[Mapping[str, object]] = (),
    *,
    history_n: int = DEFAULT_HISTORY_N,
) -> list[dict[str, object]]:
    """Build GUI-Owl's system/history/current-observation interaction flow."""
    if history_n < 1:
        raise ValueError("history_n must be positive")
    older = list(history[:-history_n]) if len(history) > history_n else []
    recent = list(history[-history_n:])
    compressed = [
        f"Step {index + 1}: {str(turn.get('conclusion') or '').strip()}"
        for index, turn in enumerate(older)
        if str(turn.get("conclusion") or "").strip()
    ]

    route = _task_ui_path(task)
    lines = [
        f"Task instruction: {task.prompt}",
        f"Task kind: {task.kind.value}; risk: {task.risk_level}",
        (
            f"Device: {profile.os_family} {profile.os_release}; Android "
            f"{profile.android_version}; API {profile.api_level}; "
            f"profile {profile.profile_id}"
        ),
        f"Current app: {observation.package}/{observation.activity}",
        "Visible labels: " + ", ".join(observation.visible_labels[:40]),
        "Accessibility summary: "
        + json.dumps(dict(observation.accessibility_summary), ensure_ascii=False)[:800],
    ]
    if compressed:
        lines.append("Compressed earlier action conclusions:\n" + "\n".join(compressed))
    if route:
        lines.append("Documented settings route: " + " > ".join(route))
        lines.append(
            "Follow the documented route through visible rows. Settings Search is "
            "allowed only when task evidence explicitly sets search_allowed=true."
        )
    settings = _settings_lines(task)
    if settings:
        lines.append("Evidence-backed settings for this OS:\n" + "\n".join(settings))
    if task.expected_output:
        lines.append(
            "Expected target: "
            + json.dumps(dict(task.expected_output), ensure_ascii=False)[:1200]
        )

    messages: list[dict[str, object]] = [
        {"role": "system", "content": _text_content(_SYSTEM_PROMPT)}
    ]
    for turn in recent:
        screenshot = str(turn.get("screenshot_reference") or "")
        user_content: list[dict[str, str]] = []
        if screenshot:
            user_content.append(_image_content(screenshot))
        user_content.extend(_text_content("Continue from this observation."))
        messages.append({"role": "user", "content": user_content})
        messages.append(
            {
                "role": "assistant",
                "content": _text_content(str(turn.get("output") or "")),
            }
        )

    current_content: list[dict[str, str]] = [_image_content(observation.screenshot_reference)]
    current_content.extend(_text_content("\n".join(lines)))
    messages.append({"role": "user", "content": current_content})
    return messages


def _clamp_coordinate(value: object) -> int:
    return max(0, min(COORDINATE_SCALE, int(round(float(value)))))


def _point_box(coordinate: Sequence[object]) -> tuple[float, float, float, float] | None:
    if len(coordinate) != 2:
        return None
    try:
        x = _clamp_coordinate(coordinate[0]) / COORDINATE_SCALE
        y = _clamp_coordinate(coordinate[1]) / COORDINATE_SCALE
    except (TypeError, ValueError):
        return None
    radius = 0.006
    x0, x1 = max(0.0, x - radius), min(1.0, x + radius)
    y0, y1 = max(0.0, y - radius), min(1.0, y + radius)
    if x0 == x1:
        x0, x1 = max(0.0, x - 0.001), min(1.0, x + 0.001)
    if y0 == y1:
        y0, y1 = max(0.0, y - 0.001), min(1.0, y + 0.001)
    return (x0, y0, x1, y1)


def _distribution(selected: ProposalKind) -> tuple[float, ...]:
    kinds = tuple(ProposalKind)
    remainder = 0.15 / (len(kinds) - 1)
    return tuple(0.85 if kind is selected else remainder for kind in kinds)


def parse_gui_owl_output(text: str) -> StudentProposal | None:
    """Map GUI-Owl's action conclusion/tool call to the repository action API."""
    match = _TOOL_CALL_RE.search(text)
    if not match:
        return None
    try:
        call = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    if not isinstance(call, Mapping):
        return None
    name = str(call.get("name") or "").casefold().rsplit(".", 1)[-1]
    arguments = call.get("arguments") or {}
    if not isinstance(arguments, Mapping):
        arguments = {}
    action = _ACTION_RE.search(text)
    conclusion = (
        action.group(1).strip()
        if action
        else str(arguments.get("text") or "").strip()
    )
    target_label = str(arguments.get("target_label") or "").strip() or None
    bbox = None
    typed_text = None

    if name in {"click", "tap", "long_press"}:
        # long_press is a touch at a point like click; the proposal API has no
        # separate press duration, so it maps onto TAP with the same bbox.
        kind = ProposalKind.TAP
        coordinate = arguments.get("coordinate") or arguments.get("point") or ()
        if isinstance(coordinate, Sequence) and not isinstance(coordinate, (str, bytes)):
            bbox = _point_box(coordinate)
    elif name in {"swipe", "scroll"}:
        kind = ProposalKind.SCROLL
    elif name == "type":
        # The typed string is the payload, not the conclusion: keep it separate
        # so the environment can enter it verbatim.
        typed_text = str(arguments.get("text") or "").strip()
        if not typed_text:
            return None
        kind = ProposalKind.TYPE
    elif name in {"system_button", "key"}:
        button = str(arguments.get("button") or arguments.get("key") or "").casefold()
        kind = ProposalKind.BACK if "back" in button else ProposalKind.WAIT
    elif name == "wait":
        kind = ProposalKind.WAIT
    elif name in {"interact", "ask_confirmation", "ask_user"}:
        kind = ProposalKind.ASK_CONFIRMATION
    elif name in {"answer", "terminate"}:
        kind = ProposalKind.EXPLAIN
    else:
        return None

    distribution = _distribution(kind)
    index = tuple(ProposalKind).index(kind)
    confidence = distribution[index]
    return StudentProposal(
        kind=kind,
        target_label=target_label,
        bbox_norm=bbox,
        message=conclusion or target_label or kind.value,
        confidence=confidence,
        action_distribution=distribution,
        log_probability=math.log(confidence),
        policy_version=0,
        text=typed_text,
    )


def teacher_gui_owl_output(target: TeacherTarget) -> str:
    """Serialize a teacher correction in the GUI-Owl action/tool-call format."""
    arguments: dict[str, object]
    if target.preferred_kind is ProposalKind.TAP:
        if target.bbox_norm:
            x0, y0, x1, y1 = target.bbox_norm
            coordinate = [
                _clamp_coordinate((x0 + x1) * 0.5 * COORDINATE_SCALE),
                _clamp_coordinate((y0 + y1) * 0.5 * COORDINATE_SCALE),
            ]
        else:
            coordinate = [500, 500]
        arguments = {"coordinate": coordinate, "target_label": target.target_label}
        name = "mobile.click"
    elif target.preferred_kind is ProposalKind.SCROLL:
        name = "mobile.swipe"
        arguments = {"start": [500, 720], "end": [500, 280], "duration": 300}
    elif target.preferred_kind is ProposalKind.BACK:
        name, arguments = "mobile.system_button", {"button": "Back"}
    elif target.preferred_kind is ProposalKind.WAIT:
        name, arguments = "mobile.wait", {"seconds": 1}
    elif target.preferred_kind is ProposalKind.TYPE:
        name, arguments = "mobile.type", {"text": target.response}
    elif target.preferred_kind is ProposalKind.ASK_CONFIRMATION:
        name, arguments = "mobile.interact", {"text": target.response}
    else:
        name, arguments = "mobile.answer", {"text": target.response}
    conclusion = target.response or target.target_label or target.preferred_kind.value
    payload = json.dumps(
        {"name": name, "arguments": arguments},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return f"Action: {conclusion}\n<tool_call>{payload}</tool_call>"


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("GUI-Owl training requires the train extra") from exc
    return torch


@contextmanager
def _adapters_disabled(model):
    if hasattr(model, "disable_adapter"):
        with model.disable_adapter():
            yield
        return
    disable = getattr(model, "disable_adapter_layers", None)
    enable = getattr(model, "enable_adapter_layers", None)
    if not callable(disable) or not callable(enable):
        raise RuntimeError("installed peft version cannot disable adapters")
    disable()
    try:
        yield
    finally:
        enable()


class GUIOwlStudent:
    """Multimodal StudentPolicy backed by GUI-Owl-1.5-2B-Instruct."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        *,
        device: str | None = None,
        dtype: str | None = None,
        max_new_tokens: int = 192,
        history_n: int = DEFAULT_HISTORY_N,
        seed: int = 0,
        lora: Mapping[str, object] | None = None,
        adapter_checkpoint: str | Path | None = None,
    ) -> None:
        torch = _require_torch()
        try:
            from peft import LoraConfig, PeftModel, get_peft_model
            from transformers import (
                AutoProcessor,
                Qwen3VLForConditionalGeneration,
                set_seed,
            )
        except ImportError as exc:
            raise RuntimeError(
                "GUI-Owl requires transformers>=4.57.1 and peft"
            ) from exc

        set_seed(seed)
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = int(max_new_tokens)
        self.history_n = int(history_n)
        self._policy_version = 0
        self._history: dict[tuple[str, str], list[dict[str, object]]] = {}
        self._traces: dict[tuple[str, str, int, int], RolloutTrace] = {}
        self.processor = AutoProcessor.from_pretrained(model_id)

        dtype_map = {
            "float32": torch.float32,
            "fp32": torch.float32,
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        dtype_name = str(dtype).casefold() if dtype else ""
        if dtype_name and dtype_name not in dtype_map:
            raise ValueError(f"unsupported dtype: {dtype}")
        model_dtype = dtype_map.get(
            dtype_name,
            torch.bfloat16 if self.device.startswith("cuda") else torch.float32,
        )
        kwargs = {"attn_implementation": "sdpa"}
        try:
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_id, dtype=model_dtype, **kwargs
            )
        except TypeError:
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_id, torch_dtype=model_dtype, **kwargs
            )
        model = model.to(self.device)
        if adapter_checkpoint:
            self.model = PeftModel.from_pretrained(
                model, str(adapter_checkpoint), is_trainable=True
            )
            self._policy_version = _checkpoint_step(Path(adapter_checkpoint))
        else:
            cfg = {
                "rank": 16,
                "alpha": 32,
                "dropout": 0.0,
                "target_modules": (
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ),
            }
            if lora:
                cfg.update(lora)
            self.model = get_peft_model(
                model,
                LoraConfig(
                    r=int(cfg["rank"]),
                    lora_alpha=int(cfg["alpha"]),
                    lora_dropout=float(cfg["dropout"]),
                    target_modules=list(cfg["target_modules"]),
                    bias="none",
                    task_type="CAUSAL_LM",
                ),
            )
        self.model.config.use_cache = False
        self.model.train()

    @property
    def policy_version(self) -> int:
        return self._policy_version

    @staticmethod
    def _key(
        task: CurriculumTask,
        profile: EmulatorProfile,
        observation: RolloutObservation,
        policy_version: int,
    ) -> tuple[str, str, int, int]:
        return (
            task.task_id,
            profile.profile_id,
            int(observation.step_index),
            int(policy_version),
        )

    def _to_device(self, batch: Mapping[str, object]) -> dict[str, object]:
        return {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in batch.items()
        }

    def encode_messages(
        self, messages: Sequence[Mapping[str, object]], *, add_generation_prompt: bool
    ) -> dict[str, object]:
        batch = self.processor.apply_chat_template(
            list(messages),
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            return_dict=True,
            return_tensors="pt",
        )
        return self._to_device(batch)

    def propose(
        self,
        task: CurriculumTask,
        profile: EmulatorProfile,
        observation: RolloutObservation,
    ) -> StudentProposal:
        torch = _require_torch()
        history_key = (task.task_id, profile.profile_id)
        if int(observation.step_index) == 0:
            self._history[history_key] = []
        messages = build_gui_owl_messages(
            task,
            profile,
            observation,
            self._history.get(history_key, ()),
            history_n=self.history_n,
        )
        batch = self.encode_messages(messages, add_generation_prompt=True)
        prompt_length = int(batch["input_ids"].shape[1])
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.inference_mode():
                generated = self.model.generate(
                    **batch,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                )
        finally:
            if was_training:
                self.model.train()
        continuation = generated[0, prompt_length:]
        text = self.processor.decode(continuation, skip_special_tokens=True)
        parsed = parse_gui_owl_output(text)
        if parsed is None:
            kind = ProposalKind.WAIT
            distribution = _distribution(kind)
            parsed = StudentProposal(
                kind,
                None,
                None,
                "Unparseable GUI-Owl output; wait for a fresh observation.",
                distribution[tuple(ProposalKind).index(kind)],
                distribution,
                math.log(distribution[tuple(ProposalKind).index(kind)]),
                self.policy_version,
            )
        else:
            parsed = StudentProposal(
                parsed.kind,
                parsed.target_label,
                parsed.bbox_norm,
                parsed.message,
                parsed.confidence,
                parsed.action_distribution,
                parsed.log_probability,
                self.policy_version,
            )
        self._traces[
            self._key(task, profile, observation, self.policy_version)
        ] = RolloutTrace(
            tuple(copy.deepcopy(messages)),
            tuple(int(value) for value in continuation.detach().cpu().tolist()),
        )
        self._history.setdefault(history_key, []).append(
            {
                "screenshot_reference": observation.screenshot_reference,
                "conclusion": parsed.message,
                "output": text,
            }
        )
        self._history[history_key] = self._history[history_key][-32:]
        return parsed

    def trace_for(self, sample: DistillationSample) -> RolloutTrace | None:
        return self._traces.get(
            self._key(
                sample.task,
                sample.profile,
                sample.observation,
                sample.proposal.policy_version,
            )
        )

    def trainable_parameters(self) -> list[object]:
        return [parameter for parameter in self.model.parameters() if parameter.requires_grad]

    def save_checkpoint(self, directory: str | Path, step: int) -> str:
        directory = Path(directory) / f"step-{step:08d}"
        directory.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(directory)
        self.processor.save_pretrained(directory)
        adapter = next(
            (
                path
                for path in (
                    directory / "adapter_model.safetensors",
                    directory / "adapter_model.bin",
                )
                if path.is_file()
            ),
            None,
        )
        if adapter is None:
            raise RuntimeError("peft did not write a GUI-Owl adapter checkpoint")
        digest = hashlib.sha256(adapter.read_bytes()).hexdigest()[:24]
        (directory / "checkpoint.id").write_text(digest, encoding="utf-8")
        return digest


class GUIOwlLoraUpdater:
    """Multimodal SFT + exact-token on-policy update for GUI-Owl."""

    def __init__(
        self,
        student: GUIOwlStudent,
        checkpoint_dir: str | Path,
        *,
        learning_rate: float = 2e-5,
        grad_clip: float = 1.0,
    ) -> None:
        torch = _require_torch()
        self.student = student
        self.checkpoint_dir = Path(checkpoint_dir)
        self.grad_clip = float(grad_clip)
        self.step_count = int(student.policy_version)
        parameters = student.trainable_parameters()
        if not parameters:
            raise RuntimeError("GUI-Owl student has no trainable LoRA parameters")
        self.optimizer = torch.optim.AdamW(
            parameters,
            lr=float(learning_rate),
            betas=(0.9, 0.95),
            weight_decay=0.0,
        )

    def _teacher_batch(self, sample: DistillationSample):
        trace = self.student.trace_for(sample)
        if trace is None:
            messages = build_gui_owl_messages(
                sample.task, sample.profile, sample.observation
            )
        else:
            messages = list(trace.messages)
        prompt = self.student.encode_messages(messages, add_generation_prompt=True)
        target = teacher_gui_owl_output(sample.judgement.target)
        full_messages = list(messages) + [
            {"role": "assistant", "content": _text_content(target)}
        ]
        full = self.student.encode_messages(
            full_messages, add_generation_prompt=False
        )
        labels = full["input_ids"].clone()
        prompt_length = min(int(prompt["input_ids"].shape[1]), int(labels.shape[1]))
        labels[:, :prompt_length] = -100
        full["labels"] = labels
        return full

    def _sampled_batch(self, sample: DistillationSample):
        torch = _require_torch()
        trace = self.student.trace_for(sample)
        if trace is None or not trace.generated_token_ids:
            return None
        batch = self.student.encode_messages(
            list(trace.messages), add_generation_prompt=True
        )
        sampled = torch.tensor(
            [trace.generated_token_ids], dtype=torch.long, device=self.student.device
        )
        prompt_length = int(batch["input_ids"].shape[1])
        batch["input_ids"] = torch.cat([batch["input_ids"], sampled], dim=1)
        if "attention_mask" in batch:
            extension = torch.ones_like(sampled)
            batch["attention_mask"] = torch.cat(
                [batch["attention_mask"], extension], dim=1
            )
        labels = batch["input_ids"].clone()
        labels[:, :prompt_length] = -100
        batch["labels"] = labels
        return batch

    def update(
        self, samples: Sequence[DistillationSample], config: OnPolicyConfig
    ) -> str:
        torch = _require_torch()
        trainable = [sample for sample in samples if sample.judgement.should_update]
        if not trainable:
            return self.student.save_checkpoint(self.checkpoint_dir, self.step_count)
        self.optimizer.zero_grad(set_to_none=True)

        for sample in trainable:
            teacher_batch = self._teacher_batch(sample)
            outputs = self.student.model(**teacher_batch)
            sft_weight = config.language_weight + config.supervised_weight
            if sample.judgement.target.preferred_kind is ProposalKind.TAP:
                sft_weight += config.grounding_weight
            loss = sft_weight * outputs.loss

            sampled_batch = self._sampled_batch(sample)
            if sampled_batch is not None and sample.advantage != 0 and config.policy_weight > 0:
                sampled_outputs = self.student.model(**sampled_batch)
                loss = loss + (
                    config.policy_weight
                    * float(sample.advantage)
                    * sampled_outputs.loss
                )

            if config.kl_weight > 0:
                labels = teacher_batch["labels"]
                keep = labels[:, 1:] != -100
                current_logits = outputs.logits[:, :-1, :]
                with torch.no_grad():
                    with _adapters_disabled(self.student.model):
                        reference_logits = self.student.model(**teacher_batch).logits[:, :-1, :]
                if keep.any():
                    current_log_probs = torch.log_softmax(current_logits[keep].float(), dim=-1)
                    reference_probs = torch.softmax(reference_logits[keep].float(), dim=-1)
                    kl = torch.nn.functional.kl_div(
                        current_log_probs, reference_probs, reduction="batchmean"
                    )
                    loss = loss + config.kl_weight * kl

            (loss / len(trainable)).backward()

        torch.nn.utils.clip_grad_norm_(
            self.student.trainable_parameters(), self.grad_clip
        )
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.step_count += 1
        self.student._policy_version += 1
        return self.student.save_checkpoint(self.checkpoint_dir, self.step_count)


def _checkpoint_step(directory: Path) -> int:
    match = re.fullmatch(r"step-(\d{8,})", directory.name)
    return int(match.group(1)) if match else 0


def _latest_checkpoint(directory: Path) -> Path | None:
    if not directory.is_dir():
        return None
    candidates = [
        path
        for path in directory.iterdir()
        if path.is_dir()
        and _checkpoint_step(path) > 0
        and any(
            (path / name).is_file()
            for name in ("adapter_model.safetensors", "adapter_model.bin")
        )
    ]
    return max(candidates, key=_checkpoint_step) if candidates else None


def _config_bool(value: object, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"boolean configuration expected, got {value!r}")


def create_student(
    student_config: Mapping[str, object], output_dir: str | Path
) -> tuple[GUIOwlStudent, GUIOwlLoraUpdater]:
    """``--student-plugin`` factory for the default multimodal GUI-Owl student."""
    config = dict(student_config or {})
    model_id = str(config.get("model_id") or DEFAULT_MODEL_ID)
    checkpoint_dir = Path(output_dir) / "checkpoints"
    adapter = None
    if _config_bool(config.get("resume"), True):
        configured = config.get("resume_checkpoint")
        adapter = Path(str(configured)).expanduser() if configured else _latest_checkpoint(checkpoint_dir)
    lora = config.get("lora")
    student = GUIOwlStudent(
        model_id,
        device=str(config["device"]) if config.get("device") else None,
        dtype=str(config["dtype"]) if config.get("dtype") else None,
        max_new_tokens=int(config.get("max_new_tokens", 192)),
        history_n=int(config.get("history_n", DEFAULT_HISTORY_N)),
        seed=int(config.get("seed", 0)),
        lora=lora if isinstance(lora, Mapping) else None,
        adapter_checkpoint=adapter,
    )
    updater = GUIOwlLoraUpdater(
        student,
        checkpoint_dir,
        learning_rate=float(config.get("learning_rate", 2e-5)),
        grad_clip=float(config.get("grad_clip", 1.0)),
    )
    return student, updater


__all__ = [
    "COORDINATE_SCALE",
    "DEFAULT_HISTORY_N",
    "DEFAULT_MODEL_ID",
    "GUIOwlLoraUpdater",
    "GUIOwlStudent",
    "RolloutTrace",
    "build_gui_owl_messages",
    "create_student",
    "parse_gui_owl_output",
    "teacher_gui_owl_output",
]
