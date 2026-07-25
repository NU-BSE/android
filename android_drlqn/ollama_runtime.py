"""Loopback-only Ollama adapters for Android VLM/chat and distillation.

The runtime deliberately accepts only a local Ollama server on port 11434. It
never pulls models, follows remote image URLs, or exposes execution tools to a
model. Qwen3-VL supplies the structured multimodal judgement; UI-TARS acts as
an independent GUI-action critic.
"""
from __future__ import annotations

import base64
import io
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlparse
from urllib.request import ProxyHandler, Request, build_opener

import numpy as np

from .advisor import ChatMessage, GenerationOptions
from .distillation import (
    EnvironmentOutcome,
    ProposalKind,
    RolloutObservation,
    StudentProposal,
    TeacherEvaluator,
    TeacherJudgement,
    TeacherTarget,
)
from .document_curriculum import CurriculumTask
from .emulator_lab import EmulatorProfile

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
_FORBIDDEN_MODEL_TEXT = (
    "adb shell",
    "input tap",
    "input swipe",
    "dispatchgesture",
    "performaction",
    "uiautomator",
    "am start",
    "node_id",
    "coordinates:",
)
_TINY_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Wl6ZfoAAAAASUVORK5CYII="
)


class OllamaRuntimeError(RuntimeError):
    """Raised when the loopback server or a configured model is unusable."""


@dataclass(frozen=True)
class OllamaServerConfig:
    base_url: str = "http://127.0.0.1:11434"
    timeout_seconds: float = 300.0
    max_image_bytes: int = 20 * 1024 * 1024
    allowed_image_roots: tuple[str, ...] = ("data", "artifacts", "tests/fixtures")

    def __post_init__(self) -> None:
        parsed = urlparse(self.base_url)
        if parsed.scheme != "http":
            raise ValueError("local Ollama must use http")
        if parsed.hostname not in _LOOPBACK_HOSTS:
            raise ValueError("Ollama host must be localhost or a loopback address")
        if parsed.port != 11434:
            raise ValueError("Ollama must be served on port 11434")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Ollama base_url must not contain credentials, query or fragment")
        if parsed.path not in ("", "/"):
            raise ValueError("Ollama base_url must not contain an API path")
        if self.timeout_seconds <= 0 or self.max_image_bytes <= 0:
            raise ValueError("timeout and image limit must be positive")

    @property
    def normalized_base_url(self) -> str:
        return self.base_url.rstrip("/")


@dataclass(frozen=True)
class OllamaModelSpec:
    model: str
    role: str
    required_capabilities: tuple[str, ...] = ("completion", "vision")
    keep_alive: str = "30m"
    num_ctx: int = 32768
    temperature: float = 0.0
    top_p: float = 1.0
    coordinate_scale: int = 1000

    def __post_init__(self) -> None:
        if not self.model or not self.role:
            raise ValueError("model and role are required")
        if self.num_ctx < 1024:
            raise ValueError("num_ctx is unexpectedly small")
        if not 0 <= self.temperature <= 2 or not 0 < self.top_p <= 1:
            raise ValueError("invalid generation settings")
        if self.coordinate_scale < 1:
            raise ValueError("coordinate_scale must be positive")


@dataclass(frozen=True)
class OllamaEnsembleConfig:
    server: OllamaServerConfig
    qwen3_vl: OllamaModelSpec
    ui_tars: OllamaModelSpec
    ui_tars_weight: float = 0.25
    min_critic_agreement: float = 0.35

    def __post_init__(self) -> None:
        if not 0 <= self.ui_tars_weight <= 0.5:
            raise ValueError("ui_tars_weight must be in [0, 0.5]")
        if not 0 <= self.min_critic_agreement <= 1:
            raise ValueError("min_critic_agreement must be in [0, 1]")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "OllamaEnsembleConfig":
        server_raw = _mapping(payload.get("server"), "server")
        qwen_raw = _mapping(payload.get("qwen3_vl"), "qwen3_vl")
        ui_raw = _mapping(payload.get("ui_tars"), "ui_tars")
        ensemble_raw = _mapping(payload.get("ensemble") or {}, "ensemble")
        server = OllamaServerConfig(
            base_url=str(server_raw.get("base_url") or "http://127.0.0.1:11434"),
            timeout_seconds=float(server_raw.get("timeout_seconds", 300.0)),
            max_image_bytes=int(server_raw.get("max_image_bytes", 20 * 1024 * 1024)),
            allowed_image_roots=tuple(
                str(value)
                for value in server_raw.get("allowed_image_roots")
                or ("data", "artifacts", "tests/fixtures")
            ),
        )
        return cls(
            server=server,
            qwen3_vl=_model_spec(qwen_raw, "multimodal_teacher", 32768),
            ui_tars=_model_spec(ui_raw, "gui_critic", 16384),
            ui_tars_weight=float(ensemble_raw.get("ui_tars_weight", 0.25)),
            min_critic_agreement=float(ensemble_raw.get("min_critic_agreement", 0.35)),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "OllamaEnsembleConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("Ollama config must be a JSON object")
        return cls.from_mapping(payload)


@dataclass(frozen=True)
class OllamaModelStatus:
    model: str
    installed: bool
    capabilities: tuple[str, ...] = ()
    parameter_size: str | None = None
    quantization_level: str | None = None
    digest: str | None = None

    @property
    def vision_ready(self) -> bool:
        return self.installed and "vision" in self.capabilities


OllamaTransport = Callable[[str, str, Mapping[str, object] | None], Mapping[str, object]]


class OllamaClient:
    """Small dependency-free client pinned to the local Ollama API."""

    def __init__(
        self,
        config: OllamaServerConfig | None = None,
        transport: OllamaTransport | None = None,
    ):
        self.config = config or OllamaServerConfig()
        self.transport = transport or self._request

    def _request(
        self,
        method: str,
        path: str,
        payload: Mapping[str, object] | None,
    ) -> Mapping[str, object]:
        if path not in {"/api/tags", "/api/show", "/api/chat", "/api/ps"}:
            raise ValueError("unapproved Ollama API path")
        data = None if payload is None else json.dumps(payload, separators=(",", ":")).encode("utf-8")
        request = Request(
            self.config.normalized_base_url + path,
            data=data,
            method=method,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        opener = build_opener(ProxyHandler({}))
        try:
            with opener.open(request, timeout=self.config.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise OllamaRuntimeError(
                f"Ollama {path} returned HTTP {exc.code}: {body[:400]}"
            ) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise OllamaRuntimeError(
                f"cannot reach local Ollama at {self.config.normalized_base_url}: {exc}"
            ) from exc
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise OllamaRuntimeError("Ollama returned malformed JSON") from exc
        if not isinstance(value, Mapping):
            raise OllamaRuntimeError("Ollama response must be a JSON object")
        if value.get("error"):
            raise OllamaRuntimeError(str(value["error"]))
        return value

    def list_models(self) -> tuple[Mapping[str, object], ...]:
        value = self.transport("GET", "/api/tags", None)
        models = value.get("models") or ()
        if not isinstance(models, Sequence):
            raise OllamaRuntimeError("Ollama /api/tags returned invalid models")
        return tuple(item for item in models if isinstance(item, Mapping))

    def show_model(self, model: str) -> Mapping[str, object]:
        return self.transport("POST", "/api/show", {"model": model, "verbose": False})

    def model_status(self, spec: OllamaModelSpec) -> OllamaModelStatus:
        names: dict[str, Mapping[str, object]] = {}
        for item in self.list_models():
            for key in ("name", "model"):
                value = str(item.get(key) or "")
                if value:
                    names[value] = item
                    names[_without_latest(value)] = item
        listed = names.get(spec.model) or names.get(_without_latest(spec.model))
        if listed is None:
            return OllamaModelStatus(spec.model, False)
        shown = self.show_model(spec.model)
        capabilities = tuple(str(value) for value in shown.get("capabilities") or ())
        details = shown.get("details") if isinstance(shown.get("details"), Mapping) else {}
        return OllamaModelStatus(
            model=spec.model,
            installed=True,
            capabilities=capabilities,
            parameter_size=_optional_str(details.get("parameter_size")),
            quantization_level=_optional_str(details.get("quantization_level")),
            digest=_optional_str(listed.get("digest")),
        )

    def require_model(self, spec: OllamaModelSpec) -> OllamaModelStatus:
        status = self.model_status(spec)
        if not status.installed:
            raise OllamaRuntimeError(f"required Ollama model is not installed: {spec.model}")
        missing = set(spec.required_capabilities) - set(status.capabilities)
        if missing:
            raise OllamaRuntimeError(
                f"Ollama model {spec.model} lacks required capabilities: {sorted(missing)}; "
                f"reported={list(status.capabilities)}"
            )
        return status

    def chat(
        self,
        spec: OllamaModelSpec,
        messages: Sequence[Mapping[str, object]],
        *,
        images: Sequence[str] = (),
        response_format: str | Mapping[str, object] | None = None,
        max_new_tokens: int = 768,
        think: bool = False,
    ) -> Mapping[str, object]:
        prepared = [dict(message) for message in messages]
        if images:
            user_index = next(
                (
                    index
                    for index in range(len(prepared) - 1, -1, -1)
                    if prepared[index].get("role") == "user"
                ),
                None,
            )
            if user_index is None:
                raise ValueError("images require a user message")
            prepared[user_index]["images"] = list(images)
        payload: dict[str, object] = {
            "model": spec.model,
            "messages": prepared,
            "stream": False,
            "think": think,
            "keep_alive": spec.keep_alive,
            "options": {
                "temperature": spec.temperature,
                "top_p": spec.top_p,
                "num_ctx": spec.num_ctx,
                "num_predict": int(max_new_tokens),
            },
        }
        if response_format is not None:
            payload["format"] = response_format
        value = self.transport("POST", "/api/chat", payload)
        message = value.get("message")
        if not isinstance(message, Mapping) or not isinstance(message.get("content"), str):
            raise OllamaRuntimeError("Ollama chat response is missing message.content")
        return value

    @staticmethod
    def content(response: Mapping[str, object]) -> str:
        message = response.get("message")
        if not isinstance(message, Mapping):
            raise OllamaRuntimeError("missing message object")
        return str(message.get("content") or "")


class OllamaChatModel:
    """`LocalChatModel` adapter backed by Qwen3-VL through Ollama."""

    def __init__(self, client: OllamaClient, model: OllamaModelSpec):
        self.client = client
        self.model = model

    def generate(self, messages: Sequence[ChatMessage], options: GenerationOptions) -> str:
        response = self.client.chat(
            self.model,
            tuple({"role": message.role, "content": message.content} for message in messages),
            max_new_tokens=options.max_new_tokens,
            think=False,
        )
        return self.client.content(response)


GROUNDING_SCHEMA: Mapping[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["label", "role", "bbox_norm", "confidence", "explanation", "metadata"],
    "properties": {
        "label": {"type": "string"},
        "role": {"type": "string"},
        "bbox_norm": {
            "type": "array",
            "items": {"type": "number", "minimum": 0, "maximum": 1},
            "minItems": 4,
            "maxItems": 4,
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "explanation": {"type": "string"},
        "metadata": {"type": "object", "additionalProperties": True},
    },
}


class OllamaMultimodalGenerator:
    """`MultimodalGenerator` adapter for `JsonGroundingVLM`."""

    def __init__(self, client: OllamaClient, model: OllamaModelSpec):
        self.client = client
        self.model = model

    def generate(
        self,
        image: np.ndarray,
        messages: Sequence[Mapping[str, str]],
        *,
        max_new_tokens: int,
    ) -> str:
        encoded = _encode_numpy_image(image)
        response = self.client.chat(
            self.model,
            tuple(dict(message) for message in messages),
            images=(encoded,),
            response_format=GROUNDING_SCHEMA,
            max_new_tokens=max_new_tokens,
            think=False,
        )
        return self.client.content(response)


JUDGEMENT_SCHEMA: Mapping[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["confidence", "scores", "should_update", "target", "tags"],
    "properties": {
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "scores": {
            "type": "object",
            "additionalProperties": False,
            "required": ["total", "grounding", "planning", "language", "safety"],
            "properties": {
                name: {"type": "number", "minimum": 0, "maximum": 1}
                for name in ("total", "grounding", "planning", "language", "safety")
            },
        },
        "should_update": {"type": "boolean"},
        "target": {
            "type": "object",
            "additionalProperties": False,
            "required": ["kind", "target_label", "bbox_norm", "response"],
            "properties": {
                "kind": {"type": "string", "enum": [value.value for value in ProposalKind]},
                "target_label": {"type": ["string", "null"]},
                "bbox_norm": {
                    "anyOf": [
                        {"type": "null"},
                        {
                            "type": "array",
                            "items": {"type": "number", "minimum": 0, "maximum": 1},
                            "minItems": 4,
                            "maxItems": 4,
                        },
                    ]
                },
                "response": {"type": "string"},
            },
        },
        "tags": {"type": "array", "items": {"type": "string"}},
    },
}

_QWEN_TEACHER_SYSTEM = """You are the local multimodal evaluator for an Android settings agent.
Inspect the screenshot and the supplied immutable evidence summary. Score only the
student's single next action. Do not execute tools, reveal chain-of-thought, invent
Android settings, or emit shell/ADB/Accessibility commands. Return only the JSON
object required by the provided schema. Bounding boxes use normalized [0,1]
coordinates in [left, top, right, bottom] order.
"""


class OllamaQwenTeacher:
    """Qwen3-VL-30B structured teacher running on local Ollama."""

    def __init__(
        self,
        client: OllamaClient,
        model: OllamaModelSpec,
        server: OllamaServerConfig,
    ):
        self.client = client
        self.model = model
        self.server = server

    @property
    def teacher_id(self) -> str:
        return f"ollama:{self.model.model}"

    def evaluate(
        self,
        task: CurriculumTask,
        profile: EmulatorProfile,
        observation: RolloutObservation,
        proposal: StudentProposal,
        outcome: EnvironmentOutcome,
    ) -> TeacherJudgement:
        image = load_image_reference(observation.screenshot_reference, self.server)
        context = {
            "rubric": {
                "grounding": "visible target and normalized box correctness",
                "planning": "evidence-backed single-next-step correctness",
                "language": "concise and truthful user-facing explanation",
                "safety": "no unsupported execution, secrets, or permission bypass",
            },
            "task": {
                "task_id": task.task_id,
                "kind": task.kind.value,
                "prompt": task.prompt,
                "expected_output": dict(task.expected_output),
                "evidence": dict(task.evidence),
            },
            "environment": {
                "profile_id": profile.profile_id,
                "os_family": profile.os_family,
                "android_version": profile.android_version,
                "visible_labels": list(observation.visible_labels),
                "accessibility_summary": dict(observation.accessibility_summary),
                "package": observation.package,
                "activity": observation.activity,
                "step_index": observation.step_index,
            },
            "student": {
                "kind": proposal.kind.value,
                "target_label": proposal.target_label,
                "bbox_norm": proposal.bbox_norm,
                "message": proposal.message,
                "confidence": proposal.confidence,
            },
            "outcome": {
                "reward": outcome.reward,
                "verified": outcome.verified,
                "terminal": outcome.terminal,
                "reason": outcome.reason,
            },
        }
        response = self.client.chat(
            self.model,
            (
                {"role": "system", "content": _QWEN_TEACHER_SYSTEM},
                {
                    "role": "user",
                    "content": json.dumps(context, ensure_ascii=False, separators=(",", ":")),
                },
            ),
            images=(image,),
            response_format=JUDGEMENT_SCHEMA,
            max_new_tokens=768,
            think=False,
        )
        try:
            raw = json.loads(self.client.content(response))
        except json.JSONDecodeError as exc:
            raise OllamaRuntimeError("Qwen3-VL did not return valid JSON") from exc
        if not isinstance(raw, Mapping):
            raise OllamaRuntimeError("Qwen3-VL judgement must be a JSON object")
        return parse_teacher_judgement(raw, self.teacher_id)


@dataclass(frozen=True)
class UITarsAction:
    kind: ProposalKind | None
    point_norm: tuple[float, float] | None
    summary: str
    raw_action: str
    abstained: bool = False


_ACTION_LINE = re.compile(r"^\s*Action\s*:\s*(.+?)\s*$", re.I | re.M)
_SUMMARY_LINE = re.compile(r"^\s*Action_Summary\s*:\s*(.+?)\s*$", re.I | re.M)
_POINT = re.compile(r"\((-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\)")
_SCROLL = re.compile(r"scroll\s*\(\s*direction\s*=\s*['\"](up|down|left|right)['\"]", re.I)


class OllamaUITarsCritic:
    """Independent UI-TARS action critic using its native action grammar."""

    def __init__(
        self,
        client: OllamaClient,
        model: OllamaModelSpec,
        server: OllamaServerConfig,
    ):
        self.client = client
        self.model = model
        self.server = server

    @property
    def teacher_id(self) -> str:
        return f"ollama:{self.model.model}"

    def critique(
        self,
        task: CurriculumTask,
        profile: EmulatorProfile,
        observation: RolloutObservation,
        proposal: StudentProposal,
    ) -> UITarsAction:
        image = load_image_reference(observation.screenshot_reference, self.server)
        prompt = (
            "Act as an independent GUI critic. Inspect the Android screenshot and choose exactly one next GUI action. "
            "Do not execute it. Use UI-TARS native output format with Action_Summary and Action only.\n"
            f"Task: {task.prompt}\n"
            f"OS profile: {profile.profile_id}\n"
            f"Visible labels: {json.dumps(list(observation.visible_labels), ensure_ascii=False)}\n"
            f"Student proposal: {json.dumps({'kind': proposal.kind.value, 'target_label': proposal.target_label, 'bbox_norm': proposal.bbox_norm}, ensure_ascii=False)}"
        )
        response = self.client.chat(
            self.model,
            ({"role": "user", "content": prompt},),
            images=(image,),
            max_new_tokens=160,
            think=False,
        )
        return parse_ui_tars_action(
            self.client.content(response),
            self.model.coordinate_scale,
        )


class OllamaTeacherEnsemble(TeacherEvaluator):
    """Qwen3-VL judgement with an independently generated UI-TARS critique."""

    def __init__(
        self,
        config: OllamaEnsembleConfig,
        client: OllamaClient | None = None,
    ):
        self.config = config
        self.client = client or OllamaClient(config.server)
        self.qwen = OllamaQwenTeacher(self.client, config.qwen3_vl, config.server)
        self.ui_tars = OllamaUITarsCritic(self.client, config.ui_tars, config.server)
        self._ready = False

    @property
    def teacher_id(self) -> str:
        return f"{self.qwen.teacher_id}+{self.ui_tars.teacher_id}"

    def preflight(self) -> tuple[OllamaModelStatus, OllamaModelStatus]:
        qwen = self.client.require_model(self.config.qwen3_vl)
        ui_tars = self.client.require_model(self.config.ui_tars)
        self._ready = True
        return qwen, ui_tars

    def evaluate(
        self,
        task: CurriculumTask,
        profile: EmulatorProfile,
        observation: RolloutObservation,
        proposal: StudentProposal,
        outcome: EnvironmentOutcome,
    ) -> TeacherJudgement:
        if not self._ready:
            self.preflight()
        primary = self.qwen.evaluate(task, profile, observation, proposal, outcome)
        critic = self.ui_tars.critique(task, profile, observation, proposal)
        agreement = critic_agreement(proposal, primary.target, critic)
        weight = self.config.ui_tars_weight
        grounding = _clamp01(
            (1.0 - weight) * primary.grounding_score + weight * agreement
        )
        total = _clamp01(
            (1.0 - weight) * primary.total_score
            + weight * ((agreement + primary.safety_score) / 2.0)
        )
        tags = list(primary.tags)
        tags.append(
            "ui-tars-abstained"
            if critic.abstained
            else f"ui-tars-agreement:{agreement:.3f}"
        )
        should_update = primary.should_update and (
            critic.abstained or agreement >= self.config.min_critic_agreement
        )
        if not should_update and primary.should_update:
            tags.append("teacher-disagreement")
        confidence = (
            primary.confidence
            if critic.abstained
            else min(primary.confidence, 0.5 + 0.5 * agreement)
        )
        return TeacherJudgement(
            teacher_id=self.teacher_id,
            confidence=_clamp01(confidence),
            total_score=total,
            grounding_score=grounding,
            planning_score=primary.planning_score,
            language_score=primary.language_score,
            safety_score=primary.safety_score,
            should_update=should_update,
            target=primary.target,
            tags=tuple(tags),
        )


class OllamaRuntimeBundle:
    """Convenience bundle for coordinator chat, VLM grounding and distillation."""

    def __init__(
        self,
        config: OllamaEnsembleConfig,
        transport: OllamaTransport | None = None,
    ):
        self.config = config
        self.client = OllamaClient(config.server, transport=transport)
        self.chat_model = OllamaChatModel(self.client, config.qwen3_vl)
        self.multimodal_generator = OllamaMultimodalGenerator(
            self.client,
            config.qwen3_vl,
        )
        self.teacher = OllamaTeacherEnsemble(config, self.client)

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        transport: OllamaTransport | None = None,
    ) -> "OllamaRuntimeBundle":
        return cls(OllamaEnsembleConfig.from_json(path), transport=transport)

    def preflight(self) -> tuple[OllamaModelStatus, OllamaModelStatus]:
        return self.teacher.preflight()


def parse_teacher_judgement(
    raw: Mapping[str, object],
    teacher_id: str,
) -> TeacherJudgement:
    allowed = {"confidence", "scores", "should_update", "target", "tags"}
    if set(raw) != allowed:
        raise ValueError(f"teacher fields must exactly match schema: {sorted(set(raw))}")
    scores = raw.get("scores")
    target = raw.get("target")
    score_fields = {"total", "grounding", "planning", "language", "safety"}
    target_fields = {"kind", "target_label", "bbox_norm", "response"}
    if not isinstance(scores, Mapping) or set(scores) != score_fields:
        raise ValueError("teacher scores must match the closed schema")
    if not isinstance(target, Mapping) or set(target) != target_fields:
        raise ValueError("teacher target must match the closed schema")
    response = str(target.get("response") or "")
    if any(value in response.casefold() for value in _FORBIDDEN_MODEL_TEXT):
        raise ValueError("teacher response attempted executable automation")
    box = _parse_box(target.get("bbox_norm"))
    return TeacherJudgement(
        teacher_id=teacher_id,
        confidence=float(raw["confidence"]),
        total_score=float(scores["total"]),
        grounding_score=float(scores["grounding"]),
        planning_score=float(scores["planning"]),
        language_score=float(scores["language"]),
        safety_score=float(scores["safety"]),
        should_update=bool(raw["should_update"]),
        target=TeacherTarget(
            preferred_kind=ProposalKind(str(target["kind"])),
            target_label=_optional_str(target.get("target_label")),
            bbox_norm=box,
            response=response,
        ),
        tags=tuple(str(value) for value in raw.get("tags") or ()),
    )


def parse_ui_tars_action(
    text: str,
    coordinate_scale: int = 1000,
) -> UITarsAction:
    lowered = text.casefold()
    if any(value in lowered for value in _FORBIDDEN_MODEL_TEXT):
        raise ValueError("UI-TARS emitted forbidden executable text")
    action_match = _ACTION_LINE.search(text)
    summary_match = _SUMMARY_LINE.search(text)
    summary = " ".join((summary_match.group(1) if summary_match else "").split())
    if not action_match:
        return UITarsAction(None, None, summary, "", True)
    action = action_match.group(1).strip()
    action_lower = action.casefold()
    if action_lower.startswith(("click(", "long_press(")):
        point = _POINT.search(action)
        if not point:
            return UITarsAction(None, None, summary, action, True)
        x = _normalize_coordinate(float(point.group(1)), coordinate_scale)
        y = _normalize_coordinate(float(point.group(2)), coordinate_scale)
        return UITarsAction(ProposalKind.TAP, (x, y), summary, action)
    if _SCROLL.search(action):
        return UITarsAction(ProposalKind.SCROLL, None, summary, action)
    if action_lower.startswith("navigate_back"):
        return UITarsAction(ProposalKind.BACK, None, summary, action)
    if action_lower.startswith("wait"):
        return UITarsAction(ProposalKind.WAIT, None, summary, action)
    if action_lower.startswith("finished"):
        return UITarsAction(ProposalKind.EXPLAIN, None, summary, action)
    return UITarsAction(None, None, summary, action, True)


def critic_agreement(
    proposal: StudentProposal,
    target: TeacherTarget,
    critic: UITarsAction,
) -> float:
    if critic.abstained or critic.kind is None:
        return 0.5
    if critic.kind is not proposal.kind:
        return 0.0
    if critic.kind is not ProposalKind.TAP:
        return 1.0
    if critic.point_norm is None:
        return 0.5
    if proposal.bbox_norm is not None and _point_in_box(
        critic.point_norm,
        proposal.bbox_norm,
        margin=0.02,
    ):
        return 1.0
    if target.bbox_norm is not None and _point_in_box(
        critic.point_norm,
        target.bbox_norm,
        margin=0.03,
    ):
        return 0.85
    return 0.15


def load_image_reference(reference: str, config: OllamaServerConfig) -> str:
    value = (reference or "").strip()
    if value.startswith("data:image/"):
        marker = ";base64,"
        if marker not in value:
            raise ValueError("image data URI must be base64 encoded")
        encoded = value.split(marker, 1)[1]
        _validate_base64_size(encoded, config.max_image_bytes)
        return encoded
    if value.startswith("base64:"):
        encoded = value.split(":", 1)[1]
        _validate_base64_size(encoded, config.max_image_bytes)
        return encoded
    parsed = urlparse(value)
    if parsed.scheme not in ("", "file"):
        raise ValueError("screenshot_reference must be a local file or base64 data URI")
    path = Path(
        unquote(parsed.path) if parsed.scheme == "file" else value
    ).expanduser().resolve()
    allowed = tuple(Path(root).expanduser().resolve() for root in config.allowed_image_roots)
    if not any(path == root or root in path.parents for root in allowed):
        raise ValueError(f"screenshot path is outside allowed roots: {path}")
    if not path.is_file():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    if size <= 0 or size > config.max_image_bytes:
        raise ValueError("screenshot file size is invalid")
    return base64.b64encode(path.read_bytes()).decode("ascii")


def smoke_image_base64() -> str:
    return _TINY_PNG_BASE64


def _encode_numpy_image(image: np.ndarray) -> str:
    array = np.asarray(image)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    if array.ndim != 3 or array.shape[-1] not in (3, 4):
        raise ValueError("image must be HxW, HxWx3 or HxWx4")
    if np.issubdtype(array.dtype, np.floating):
        array = np.clip(array, 0, 1) * 255.0
    array = np.clip(array, 0, 255).astype(np.uint8)
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Pillow is required for the Ollama multimodal adapter; install android-drlqn[ollama]"
        ) from exc
    output = io.BytesIO()
    Image.fromarray(array).save(output, format="PNG", optimize=True)
    return base64.b64encode(output.getvalue()).decode("ascii")


def _mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a JSON object")
    return value


def _model_spec(
    raw: Mapping[str, object],
    default_role: str,
    default_num_ctx: int,
) -> OllamaModelSpec:
    return OllamaModelSpec(
        model=str(raw.get("model") or ""),
        role=str(raw.get("role") or default_role),
        required_capabilities=tuple(
            str(value)
            for value in raw.get("required_capabilities")
            or ("completion", "vision")
        ),
        keep_alive=str(raw.get("keep_alive") or "30m"),
        num_ctx=int(raw.get("num_ctx", default_num_ctx)),
        temperature=float(raw.get("temperature", 0.0)),
        top_p=float(raw.get("top_p", 1.0)),
        coordinate_scale=int(raw.get("coordinate_scale", 1000)),
    )


def _without_latest(value: str) -> str:
    return value[:-7] if value.endswith(":latest") else value


def _optional_str(value: object) -> str | None:
    return None if value is None else str(value)


def _parse_box(value: object) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("bbox_norm must contain four numbers or null")
    box = tuple(float(item) for item in value)
    x0, y0, x1, y1 = box
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise ValueError("bbox_norm is invalid")
    return box


def _validate_base64_size(encoded: str, max_bytes: int) -> None:
    try:
        decoded = base64.b64decode(encoded, validate=True)
    except Exception as exc:
        raise ValueError("invalid base64 screenshot") from exc
    if not decoded or len(decoded) > max_bytes:
        raise ValueError("base64 screenshot size is invalid")


def _normalize_coordinate(value: float, scale: int) -> float:
    if not math.isfinite(value):
        raise ValueError("UI-TARS coordinate must be finite")
    normalized = value if 0 <= value <= 1 else value / float(scale)
    if not 0 <= normalized <= 1:
        raise ValueError("UI-TARS coordinate is outside the configured scale")
    return normalized


def _point_in_box(
    point: tuple[float, float],
    box: tuple[float, float, float, float],
    margin: float = 0.0,
) -> bool:
    x, y = point
    left, top, right, bottom = box
    return left - margin <= x <= right + margin and top - margin <= y <= bottom + margin


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))
