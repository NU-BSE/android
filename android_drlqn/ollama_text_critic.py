"""Qwen-VL perception -> text-only UI-TARS critic bridge for local Ollama.

Only the perception model receives screenshot bytes. The critic receives a
bounded JSON description of the UI plus the existing task/proposal/outcome
state, so a text-only UI-TARS derivative needs only the ``completion``
capability.

The bridge implements the repository's ``TeacherEvaluator`` protocol and is
intentionally independent of the existing Ollama runtime internals. It uses the
native ``/api/show`` and ``/api/chat`` endpoints through a small injectable
transport, which keeps unit tests network-free.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import socket
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .distillation import (
    EnvironmentOutcome,
    ProposalKind,
    RolloutObservation,
    StudentProposal,
    TeacherJudgement,
    TeacherTarget,
)
from .document_curriculum import CurriculumTask
from .emulator_lab import EmulatorProfile

_JSON = Mapping[str, object]
Transport = Callable[[str, Mapping[str, object]], Mapping[str, object]]

ENV_ENABLE = "ANDROID_DRLQN_TEXT_CRITIC_BRIDGE"
ENV_BASE_URL = "ANDROID_DRLQN_OLLAMA_BASE_URL"
ENV_VISION_MODEL = "ANDROID_DRLQN_QWEN_VL_MODEL"
ENV_CRITIC_MODEL = "ANDROID_DRLQN_UI_TARS_CRITIC_MODEL"
ENV_ALLOWED_ROOTS = "ANDROID_DRLQN_ALLOWED_IMAGE_ROOTS"
ENV_TIMEOUT = "ANDROID_DRLQN_OLLAMA_TIMEOUT_S"
ENV_MAX_IMAGE_BYTES = "ANDROID_DRLQN_MAX_IMAGE_BYTES"

DEFAULT_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_VISION_MODEL = "qwen3-vl:4b"
DEFAULT_CRITIC_MODEL = "android-ui-tars-critic:latest"
DEFAULT_TIMEOUT_S = 120.0
DEFAULT_MAX_IMAGE_BYTES = 20 * 1024 * 1024
_MAX_ELEMENTS = 64
_MAX_CACHE_ENTRIES = 64
_MAX_TEXT = 2_000
_MAX_LABEL = 240
_MAX_TAGS = 24

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_JPEG_MAGIC = b"\xff\xd8\xff"


class TextCriticBridgeError(RuntimeError):
    """Raised when perception, capability checks, or strict parsing fails."""


@dataclass(frozen=True)
class BridgeConfig:
    base_url: str = DEFAULT_BASE_URL
    vision_model: str = DEFAULT_VISION_MODEL
    critic_model: str = DEFAULT_CRITIC_MODEL
    allowed_image_roots: tuple[Path, ...] = ()
    timeout_s: float = DEFAULT_TIMEOUT_S
    max_image_bytes: int = DEFAULT_MAX_IMAGE_BYTES

    def __post_init__(self) -> None:
        parsed = urlparse(self.base_url)
        host = (parsed.hostname or "").casefold()
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("Ollama base URL must use http or https")
        if parsed.username or parsed.password:
            raise ValueError("credentials in the Ollama base URL are forbidden")
        if parsed.scheme == "http" and host not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("plain HTTP is allowed only for loopback Ollama")
        if not self.vision_model.strip() or not self.critic_model.strip():
            raise ValueError("vision_model and critic_model are required")
        if self.timeout_s <= 0 or self.max_image_bytes < 1:
            raise ValueError("timeout and max image size must be positive")

    @classmethod
    def from_environment(cls, cwd: str | Path | None = None) -> "BridgeConfig":
        root = Path(cwd or Path.cwd()).resolve()
        configured_roots = [
            Path(value).expanduser().resolve()
            for value in os.environ.get(ENV_ALLOWED_ROOTS, "").split(os.pathsep)
            if value.strip()
        ]
        if not configured_roots:
            configured_roots = [
                (root / "artifacts" / "lab-screenshots").resolve(),
                (root / "artifacts").resolve(),
            ]
        return cls(
            base_url=os.environ.get(ENV_BASE_URL, DEFAULT_BASE_URL).rstrip("/"),
            vision_model=os.environ.get(ENV_VISION_MODEL, DEFAULT_VISION_MODEL),
            critic_model=os.environ.get(ENV_CRITIC_MODEL, DEFAULT_CRITIC_MODEL),
            allowed_image_roots=tuple(configured_roots),
            timeout_s=float(os.environ.get(ENV_TIMEOUT, DEFAULT_TIMEOUT_S)),
            max_image_bytes=int(
                os.environ.get(ENV_MAX_IMAGE_BYTES, DEFAULT_MAX_IMAGE_BYTES)
            ),
        )


class NativeOllamaTransport:
    """Minimal JSON transport for Ollama's native API."""

    def __init__(self, base_url: str, timeout_s: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = float(timeout_s)

    def __call__(
        self, endpoint: str, payload: Mapping[str, object]
    ) -> Mapping[str, object]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = Request(
            self.base_url + endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_s) as response:  # noqa: S310
                raw = response.read()
        except (HTTPError, URLError, TimeoutError, socket.timeout, OSError) as exc:
            detail = getattr(exc, "reason", None) or str(exc)
            raise TextCriticBridgeError(
                f"Ollama request failed for {endpoint}: {detail}"
            ) from exc
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TextCriticBridgeError(
                f"Ollama returned invalid JSON for {endpoint}"
            ) from exc
        if not isinstance(value, Mapping):
            raise TextCriticBridgeError(
                f"Ollama returned a non-object response for {endpoint}"
            )
        return value


PERCEPTION_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["screen_summary", "visible_elements", "uncertainties"],
    "properties": {
        "screen_summary": {"type": "string"},
        "visible_elements": {
            "type": "array",
            "maxItems": _MAX_ELEMENTS,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["label", "role", "bbox_norm", "state"],
                "properties": {
                    "label": {"type": "string"},
                    "role": {"type": "string"},
                    "bbox_norm": {
                        "anyOf": [
                            {"type": "null"},
                            {
                                "type": "array",
                                "minItems": 4,
                                "maxItems": 4,
                                "items": {"type": "number"},
                            },
                        ]
                    },
                    "state": {"type": "string"},
                },
            },
        },
        "uncertainties": {
            "type": "array",
            "maxItems": 16,
            "items": {"type": "string"},
        },
    },
}

JUDGEMENT_SCHEMA: dict[str, object] = {
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
                "kind": {
                    "type": "string",
                    "enum": [kind.value for kind in ProposalKind],
                },
                "target_label": {"type": ["string", "null"]},
                "bbox_norm": {
                    "anyOf": [
                        {"type": "null"},
                        {
                            "type": "array",
                            "minItems": 4,
                            "maxItems": 4,
                            "items": {"type": "number"},
                        },
                    ]
                },
                "response": {"type": "string"},
            },
        },
        "tags": {
            "type": "array",
            "maxItems": _MAX_TAGS,
            "items": {"type": "string"},
        },
    },
}

_PERCEPTION_SYSTEM = """You are the visual perception stage for an Android UI evaluator.
Inspect only the supplied screenshot. Return the requested JSON schema and no prose.
Report visible text and controls, approximate normalized bounding boxes when clear,
and explicit uncertainty when a detail is unreadable. Do not propose actions, do not
infer hidden UI, and do not reveal chain-of-thought."""

_CRITIC_SYSTEM = """You are a text-only Android UI-TARS critic. You never receive image
bytes. Judge the student proposal using the Qwen-VL perception JSON, accessibility
state, task evidence, and observed outcome. Return exactly the requested JSON schema.
Do not claim hidden visual facts, do not emit shell/adb/accessibility commands, and do
not reveal chain-of-thought. Set should_update=false when perception is insufficient or
when the evidence and proposal cannot be reconciled safely."""

_FORBIDDEN = (
    "adb shell",
    "input tap",
    "dispatchgesture",
    "performaction",
    "node_id",
    "coordinates:",
)


def _clip_text(value: object, limit: int = _MAX_TEXT) -> str:
    return str(value or "").strip()[:limit]


def _bounded_json(value: object, depth: int = 0) -> object:
    """Copy arbitrary task state into a small JSON-safe structure."""
    if depth >= 4:
        return _clip_text(value, 300)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value[:_MAX_TEXT]
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in list(value.items())[:48]:
            result[_clip_text(key, 120)] = _bounded_json(item, depth + 1)
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_bounded_json(item, depth + 1) for item in list(value)[:64]]
    return _clip_text(value, 500)


def _bbox(value: object) -> tuple[float, float, float, float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return None
    if len(value) != 4:
        return None
    try:
        x0, y0, x1, y1 = (float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        return None
    return (x0, y0, x1, y1)


def _parse_message_json(response: Mapping[str, object], stage: str) -> dict[str, object]:
    message = response.get("message")
    if not isinstance(message, Mapping):
        raise TextCriticBridgeError(f"{stage} response lacks message object")
    content = message.get("content")
    if isinstance(content, Mapping):
        value = dict(content)
    elif isinstance(content, str):
        try:
            decoded = json.loads(content)
        except json.JSONDecodeError as exc:
            raise TextCriticBridgeError(f"{stage} returned invalid JSON content") from exc
        if not isinstance(decoded, Mapping):
            raise TextCriticBridgeError(f"{stage} JSON content is not an object")
        value = dict(decoded)
    else:
        raise TextCriticBridgeError(f"{stage} response lacks textual JSON content")
    return value


def _capabilities(response: Mapping[str, object]) -> tuple[str, ...]:
    values = response.get("capabilities") or ()
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        return ()
    return tuple(str(value).casefold() for value in values)


def _normalise_perception(raw: Mapping[str, object]) -> dict[str, object]:
    allowed = {"screen_summary", "visible_elements", "uncertainties"}
    extra = set(raw) - allowed
    if extra:
        raise TextCriticBridgeError(
            f"Qwen-VL perception contains unexpected fields: {sorted(extra)}"
        )
    summary = _clip_text(raw.get("screen_summary"))
    elements_raw = raw.get("visible_elements") or ()
    uncertainties_raw = raw.get("uncertainties") or ()
    if not isinstance(elements_raw, Sequence) or isinstance(
        elements_raw, (str, bytes, bytearray)
    ):
        raise TextCriticBridgeError("Qwen-VL visible_elements must be an array")
    if not isinstance(uncertainties_raw, Sequence) or isinstance(
        uncertainties_raw, (str, bytes, bytearray)
    ):
        raise TextCriticBridgeError("Qwen-VL uncertainties must be an array")

    elements: list[dict[str, object]] = []
    for item in list(elements_raw)[:_MAX_ELEMENTS]:
        if not isinstance(item, Mapping):
            continue
        elements.append(
            {
                "label": _clip_text(item.get("label"), _MAX_LABEL),
                "role": _clip_text(item.get("role"), 80),
                "bbox_norm": list(box) if (box := _bbox(item.get("bbox_norm"))) else None,
                "state": _clip_text(item.get("state"), 160),
            }
        )
    uncertainties = [
        _clip_text(item, 240) for item in list(uncertainties_raw)[:16] if _clip_text(item, 240)
    ]
    if not summary and not elements:
        raise TextCriticBridgeError("Qwen-VL returned an empty UI perception")
    return {
        "screen_summary": summary,
        "visible_elements": elements,
        "uncertainties": uncertainties,
    }


def _score(value: object, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TextCriticBridgeError(f"critic score {name!r} is not numeric") from exc
    if not 0 <= number <= 1:
        raise TextCriticBridgeError(f"critic score {name!r} is outside [0, 1]")
    return number


class QwenVisionTextCritic:
    """TeacherEvaluator that splits visual perception from text-only criticism."""

    def __init__(
        self,
        config: BridgeConfig,
        transport: Transport | None = None,
    ) -> None:
        self.config = config
        self.transport = transport or NativeOllamaTransport(
            config.base_url, config.timeout_s
        )
        self._cache: OrderedDict[str, dict[str, object]] = OrderedDict()
        self._validated = False

    @classmethod
    def from_environment(
        cls,
        cwd: str | Path | None = None,
        transport: Transport | None = None,
    ) -> "QwenVisionTextCritic":
        return cls(BridgeConfig.from_environment(cwd), transport=transport)

    @property
    def teacher_id(self) -> str:
        return f"qwen-vl-text-bridge:{self.config.vision_model}->{self.config.critic_model}"

    def validate_models(self) -> None:
        vision_caps = _capabilities(
            self.transport(
                "/api/show", {"model": self.config.vision_model, "verbose": False}
            )
        )
        critic_caps = _capabilities(
            self.transport(
                "/api/show", {"model": self.config.critic_model, "verbose": False}
            )
        )
        vision_missing = sorted({"completion", "vision"} - set(vision_caps))
        critic_missing = sorted({"completion"} - set(critic_caps))
        if vision_missing:
            raise TextCriticBridgeError(
                f"Ollama vision model {self.config.vision_model} lacks required "
                f"capabilities: {vision_missing}; reported={list(vision_caps)}"
            )
        if critic_missing:
            raise TextCriticBridgeError(
                f"Ollama text critic {self.config.critic_model} lacks required "
                f"capabilities: {critic_missing}; reported={list(critic_caps)}"
            )
        self._validated = True

    def evaluate(
        self,
        task: CurriculumTask,
        profile: EmulatorProfile,
        observation: RolloutObservation,
        proposal: StudentProposal,
        outcome: EnvironmentOutcome,
    ) -> TeacherJudgement:
        if not self._validated:
            self.validate_models()
        perception = self._perceive(task, profile, observation)
        payload = self._critic_payload(
            task, profile, observation, proposal, outcome, perception
        )
        response = self.transport(
            "/api/chat",
            {
                "model": self.config.critic_model,
                "messages": [
                    {"role": "system", "content": _CRITIC_SYSTEM},
                    {
                        "role": "user",
                        "content": json.dumps(payload, ensure_ascii=False, sort_keys=True),
                    },
                ],
                "format": JUDGEMENT_SCHEMA,
                "stream": False,
                "think": False,
                "options": {"temperature": 0, "seed": 0},
            },
        )
        raw = _parse_message_json(response, "UI-TARS critic")
        return self._judgement(raw)

    def _perceive(
        self,
        task: CurriculumTask,
        profile: EmulatorProfile,
        observation: RolloutObservation,
    ) -> dict[str, object]:
        image, digest = self._read_image(observation.screenshot_reference)
        cached = self._cache.get(digest)
        if cached is not None:
            self._cache.move_to_end(digest)
            return cached

        prompt = {
            "task": _clip_text(getattr(task, "prompt", ""), 1_000),
            "os_family": _clip_text(getattr(profile, "os_family", ""), 120),
            "package": _clip_text(getattr(observation, "package", ""), 200),
            "activity": _clip_text(getattr(observation, "activity", ""), 200),
            "known_accessibility_labels": [
                _clip_text(value, _MAX_LABEL)
                for value in list(getattr(observation, "visible_labels", ()) or ())[:64]
            ],
            "instruction": (
                "Describe only the currently visible Android UI. Accessibility labels "
                "are hints and must not override the screenshot."
            ),
        }
        response = self.transport(
            "/api/chat",
            {
                "model": self.config.vision_model,
                "messages": [
                    {"role": "system", "content": _PERCEPTION_SYSTEM},
                    {
                        "role": "user",
                        "content": json.dumps(prompt, ensure_ascii=False, sort_keys=True),
                        "images": [base64.b64encode(image).decode("ascii")],
                    },
                ],
                "format": PERCEPTION_SCHEMA,
                "stream": False,
                "think": False,
                "options": {"temperature": 0, "seed": 0},
            },
        )
        perception = _normalise_perception(
            _parse_message_json(response, "Qwen-VL perception")
        )
        self._cache[digest] = perception
        self._cache.move_to_end(digest)
        while len(self._cache) > _MAX_CACHE_ENTRIES:
            self._cache.popitem(last=False)
        return perception

    def _read_image(self, reference: str) -> tuple[bytes, str]:
        raw_reference = str(reference or "").strip()
        if not raw_reference:
            raise TextCriticBridgeError("observation screenshot_reference is empty")
        parsed = urlparse(raw_reference)
        if parsed.scheme not in {"", "file"}:
            raise TextCriticBridgeError(
                "only local screenshot paths are accepted by the text critic bridge"
            )
        path = Path(parsed.path if parsed.scheme == "file" else raw_reference).expanduser()
        if path.is_symlink():
            raise TextCriticBridgeError("screenshot must be a regular non-symlink file")
        try:
            resolved = path.resolve(strict=True)
        except OSError as exc:
            raise TextCriticBridgeError(f"screenshot is unavailable: {path}") from exc
        if not resolved.is_file():
            raise TextCriticBridgeError("screenshot must be a regular non-symlink file")
        roots = self.config.allowed_image_roots
        if roots and not any(_is_relative_to(resolved, root) for root in roots):
            raise TextCriticBridgeError(
                f"screenshot is outside allowed image roots: {resolved}"
            )
        try:
            size = resolved.stat().st_size
        except OSError as exc:
            raise TextCriticBridgeError(f"cannot stat screenshot: {resolved}") from exc
        if not 0 < size <= self.config.max_image_bytes:
            raise TextCriticBridgeError(
                f"screenshot size {size} is outside the allowed range"
            )
        try:
            image = resolved.read_bytes()
        except OSError as exc:
            raise TextCriticBridgeError(f"cannot read screenshot: {resolved}") from exc
        if not (image.startswith(_PNG_MAGIC) or image.startswith(_JPEG_MAGIC)):
            raise TextCriticBridgeError("screenshot must be PNG or JPEG")
        return image, hashlib.sha256(image).hexdigest()

    @staticmethod
    def _critic_payload(
        task: CurriculumTask,
        profile: EmulatorProfile,
        observation: RolloutObservation,
        proposal: StudentProposal,
        outcome: EnvironmentOutcome,
        perception: Mapping[str, object],
    ) -> dict[str, object]:
        return {
            "rubric": {
                "grounding": "target is supported by Qwen-VL perception/accessibility",
                "planning": "next step follows task evidence and current screen",
                "language": "message is concise and user-facing",
                "safety": "no unsupported execution or sensitive leakage",
            },
            "task": {
                "task_id": _clip_text(getattr(task, "task_id", ""), 200),
                "kind": _clip_text(getattr(getattr(task, "kind", None), "value", ""), 120),
                "prompt": _clip_text(getattr(task, "prompt", ""), 1_500),
                "expected_output": _bounded_json(
                    getattr(task, "expected_output", {}) or {}
                ),
                "evidence": _bounded_json(getattr(task, "evidence", {}) or {}),
            },
            "profile": {
                "profile_id": _clip_text(getattr(profile, "profile_id", ""), 200),
                "os_family": _clip_text(getattr(profile, "os_family", ""), 120),
                "android_version": _clip_text(
                    getattr(profile, "android_version", ""), 80
                ),
                "api_level": _bounded_json(getattr(profile, "api_level", None)),
            },
            "environment": {
                "package": _clip_text(getattr(observation, "package", ""), 200),
                "activity": _clip_text(getattr(observation, "activity", ""), 200),
                "step_index": int(getattr(observation, "step_index", 0)),
                "visible_labels": [
                    _clip_text(value, _MAX_LABEL)
                    for value in list(
                        getattr(observation, "visible_labels", ()) or ()
                    )[:64]
                ],
                "accessibility_summary": _bounded_json(
                    getattr(observation, "accessibility_summary", {}) or {}
                ),
                "qwen_vl_perception": dict(perception),
            },
            "student": {
                "kind": _clip_text(
                    getattr(getattr(proposal, "kind", None), "value", ""), 80
                ),
                "target_label": _clip_text(
                    getattr(proposal, "target_label", None), _MAX_LABEL
                )
                or None,
                "bbox_norm": list(box)
                if (box := _bbox(getattr(proposal, "bbox_norm", None)))
                else None,
                "message": _clip_text(getattr(proposal, "message", ""), 1_000),
                "confidence": float(getattr(proposal, "confidence", 0.0)),
            },
            "outcome": {
                "reward": float(getattr(outcome, "reward", 0.0)),
                "terminal": bool(getattr(outcome, "terminal", False)),
                "verified": bool(getattr(outcome, "verified", False)),
                "reason": _clip_text(getattr(outcome, "reason", ""), 1_000),
            },
        }

    def _judgement(self, raw: Mapping[str, object]) -> TeacherJudgement:
        allowed = {"confidence", "scores", "should_update", "target", "tags"}
        extra = set(raw) - allowed
        if extra:
            raise TextCriticBridgeError(
                f"UI-TARS critic contains unexpected fields: {sorted(extra)}"
            )
        scores = raw.get("scores")
        target = raw.get("target")
        tags = raw.get("tags") or ()
        if not isinstance(scores, Mapping) or not isinstance(target, Mapping):
            raise TextCriticBridgeError("UI-TARS critic lacks scores or target object")
        if not isinstance(tags, Sequence) or isinstance(tags, (str, bytes, bytearray)):
            raise TextCriticBridgeError("UI-TARS critic tags must be an array")
        kind_value = _clip_text(target.get("kind"), 80)
        try:
            kind = ProposalKind(kind_value)
        except ValueError as exc:
            raise TextCriticBridgeError(
                f"UI-TARS critic returned an invalid proposal kind: {kind_value!r}"
            ) from exc
        response = _clip_text(target.get("response"), 2_000)
        folded = response.casefold()
        if any(fragment in folded for fragment in _FORBIDDEN):
            raise TextCriticBridgeError("UI-TARS critic response violates safety boundary")
        normalized_tags = tuple(
            dict.fromkeys(
                [
                    _clip_text(value, 120)
                    for value in list(tags)[:_MAX_TAGS]
                    if _clip_text(value, 120)
                ]
                + ["qwen-vl-text-bridge"]
            )
        )
        return TeacherJudgement(
            teacher_id=self.teacher_id,
            confidence=_score(raw.get("confidence"), "confidence"),
            total_score=_score(scores.get("total"), "total"),
            grounding_score=_score(scores.get("grounding"), "grounding"),
            planning_score=_score(scores.get("planning"), "planning"),
            language_score=_score(scores.get("language"), "language"),
            safety_score=_score(scores.get("safety"), "safety"),
            should_update=bool(raw.get("should_update", False)),
            target=TeacherTarget(
                preferred_kind=kind,
                target_label=_clip_text(target.get("target_label"), _MAX_LABEL)
                or None,
                bbox_norm=_bbox(target.get("bbox_norm")),
                response=response,
            ),
            tags=normalized_tags,
        )


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root.resolve())
        return True
    except ValueError:
        return False


__all__ = [
    "BridgeConfig",
    "JUDGEMENT_SCHEMA",
    "NativeOllamaTransport",
    "PERCEPTION_SCHEMA",
    "QwenVisionTextCritic",
    "TextCriticBridgeError",
]
