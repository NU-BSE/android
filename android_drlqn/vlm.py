"""Strict multimodal grounding bridge for Android settings navigation.

The bridge accepts a screenshot, a timestamp-aligned accessibility snapshot and
one semantic target. It prefers stable accessibility nodes and uses a local VLM
only when the tree cannot identify the target. VLM output is parsed as a small
JSON contract; it is never treated as executable text.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Protocol, Sequence

import numpy as np

from .advisor import SensitiveTextRedactor
from .core import AndroidState, Bounds

_TOKEN = re.compile(r"[\wа-яё]+", re.I)
_FORBIDDEN_TEXT = (
    "adb shell",
    "input tap",
    "input swipe",
    "dispatchgesture",
    "performaction",
    "uiautomator",
    "am start",
    "node_id=",
)


class GroundingSource(str, Enum):
    ACCESSIBILITY = "accessibility"
    DRLQN = "drlqn"
    VLM = "vlm"
    NONE = "none"


@dataclass(frozen=True)
class NormalizedBox:
    left: float
    top: float
    right: float
    bottom: float

    def __post_init__(self) -> None:
        values = (self.left, self.top, self.right, self.bottom)
        if any(not math.isfinite(value) for value in values):
            raise ValueError("box coordinates must be finite")
        if not (0.0 <= self.left < self.right <= 1.0):
            raise ValueError("box horizontal coordinates must satisfy 0 <= left < right <= 1")
        if not (0.0 <= self.top < self.bottom <= 1.0):
            raise ValueError("box vertical coordinates must satisfy 0 <= top < bottom <= 1")

    @property
    def center(self) -> tuple[float, float]:
        return ((self.left + self.right) / 2.0, (self.top + self.bottom) / 2.0)

    @classmethod
    def from_device_bounds(cls, bounds: Bounds, width: int, height: int) -> "NormalizedBox":
        width = max(1, int(width))
        height = max(1, int(height))
        left = min(1.0, max(0.0, float(bounds.left) / width))
        top = min(1.0, max(0.0, float(bounds.top) / height))
        right = min(1.0, max(0.0, float(bounds.right) / width))
        bottom = min(1.0, max(0.0, float(bounds.bottom) / height))
        if right <= left or bottom <= top:
            raise ValueError("device bounds are empty")
        return cls(left, top, right, bottom)


@dataclass(frozen=True)
class VisibleElement:
    label: str
    role: str
    box: NormalizedBox
    node_id: str | None = None
    clickable: bool = False
    enabled: bool = True
    checked: bool | None = None


@dataclass(frozen=True)
class VLMRequest:
    image: np.ndarray
    target: str
    screen_description: str
    visible_elements: tuple[VisibleElement, ...]
    locale: str = "ru"
    # The contract carries label, role, four box floats, confidence and a short
    # explanation. Below ~160 tokens the JSON is truncated mid-object and every
    # answer is rejected as a contract violation.
    max_new_tokens: int = 256


class LocalVisionLanguageModel(Protocol):
    """Runtime adapter implemented by LiteRT-LM, MediaPipe or ONNX Runtime code.

    The model must return JSON only. Image preprocessing/tokenization remains a
    responsibility of the concrete Android runtime adapter.
    """

    def ground(self, request: VLMRequest) -> str | Mapping[str, object]:
        ...


class MultimodalGenerator(Protocol):
    """Low-level multimodal text generator supplied by an Android runtime."""

    def generate(
        self,
        image: np.ndarray,
        messages: Sequence[Mapping[str, str]],
        *,
        max_new_tokens: int,
    ) -> str:
        ...


_VLM_SYSTEM_RU = """Ты — локальный visual grounding модуль Android.
Найди один элемент интерфейса, соответствующий target. Не предлагай команды,
жесты, координаты касания, shell или Accessibility API. Верни только JSON:
{"label":"...","role":"...","bbox_norm":[l,t,r,b],
 "confidence":0.0,"explanation":"краткая визуальная причина","metadata":{}}
Все bbox_norm должны находиться в диапазоне [0,1].
"""

_VLM_SYSTEM_EN = """You are an on-device Android visual-grounding module.
Locate exactly one UI element matching target. Do not propose commands, gestures,
tap coordinates, shell, or Accessibility APIs. Return JSON only:
{"label":"...","role":"...","bbox_norm":[l,t,r,b],
 "confidence":0.0,"explanation":"brief visual reason","metadata":{}}
All bbox_norm values must be in [0,1].
"""


class JsonGroundingVLM:
    """Concrete bridge from a multimodal generator to LocalVisionLanguageModel."""

    def __init__(self, generator: MultimodalGenerator):
        self.generator = generator

    def ground(self, request: VLMRequest) -> str:
        visible = [
            {
                "label": item.label,
                "role": item.role,
                "bbox_norm": [item.box.left, item.box.top, item.box.right, item.box.bottom],
                "clickable": item.clickable,
                "checked": item.checked,
            }
            for item in request.visible_elements
        ]
        context = json.dumps(
            {
                "target": request.target,
                "screen": request.screen_description,
                "tree_candidates": visible,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        system = _VLM_SYSTEM_RU if request.locale.startswith("ru") else _VLM_SYSTEM_EN
        return self.generator.generate(
            request.image,
            (
                {"role": "system", "content": system},
                {"role": "user", "content": context},
            ),
            max_new_tokens=request.max_new_tokens,
        )


@dataclass(frozen=True)
class GroundingDecision:
    source: GroundingSource
    confidence: float
    label: str | None = None
    node_id: str | None = None
    box: NormalizedBox | None = None
    role: str | None = None
    explanation: str = ""
    model_metadata: Mapping[str, object] = field(default_factory=dict)

    @property
    def grounded(self) -> bool:
        return self.source is not GroundingSource.NONE and self.box is not None


@dataclass(frozen=True)
class GrounderConfig:
    accessibility_threshold: float = 0.72
    drlqn_threshold: float = 0.55
    vlm_threshold: float = 0.60
    max_visible_elements: int = 64
    redact_sensitive_text: bool = True
    vlm_max_new_tokens: int = 256

    def __post_init__(self) -> None:
        if self.vlm_max_new_tokens < 160:
            raise ValueError(
                "vlm_max_new_tokens below 160 truncates the grounding JSON contract"
            )


class VLMContractError(ValueError):
    pass


def _tokens(value: str) -> set[str]:
    return {match.group(0).casefold() for match in _TOKEN.finditer(value or "") if len(match.group(0)) > 1}


def _text_score(target: str, label: str) -> float:
    target_norm = " ".join((target or "").casefold().split())
    label_norm = " ".join((label or "").casefold().split())
    if not target_norm or not label_norm:
        return 0.0
    if target_norm == label_norm:
        return 1.0
    if target_norm in label_norm or label_norm in target_norm:
        return 0.88
    target_tokens = _tokens(target_norm)
    label_tokens = _tokens(label_norm)
    if not target_tokens or not label_tokens:
        return 0.0
    overlap = len(target_tokens & label_tokens) / len(target_tokens | label_tokens)
    coverage = len(target_tokens & label_tokens) / len(target_tokens)
    return min(0.86, 0.45 * overlap + 0.55 * coverage)


class HybridGrounder:
    """Accessibility-first grounding with DRLQN ranking and VLM fallback."""

    def __init__(self, config: GrounderConfig | None = None):
        self.config = config or GrounderConfig()

    def ground(
        self,
        state: AndroidState,
        target: str,
        *,
        vlm: LocalVisionLanguageModel | None = None,
        drlqn_node_scores: Sequence[float] | np.ndarray | None = None,
    ) -> GroundingDecision:
        elements = self.visible_elements(state)
        accessibility = self._ground_accessibility(elements, target)
        if accessibility.confidence >= self.config.accessibility_threshold:
            return accessibility

        drlqn = self._ground_drlqn(elements, target, drlqn_node_scores)
        if drlqn.confidence >= self.config.drlqn_threshold:
            return drlqn

        if vlm is None:
            best = max((accessibility, drlqn), key=lambda item: item.confidence)
            return best if best.confidence > 0 else GroundingDecision(GroundingSource.NONE, 0.0)

        request = self.build_request(state, target, elements=elements)
        try:
            decision = self.parse_vlm_response(vlm.ground(request))
        except (TypeError, ValueError, json.JSONDecodeError, VLMContractError):
            best = max((accessibility, drlqn), key=lambda item: item.confidence)
            return best if best.confidence > 0 else GroundingDecision(GroundingSource.NONE, 0.0)
        if decision.confidence < self.config.vlm_threshold:
            return GroundingDecision(
                GroundingSource.NONE,
                decision.confidence,
                explanation="VLM confidence below threshold",
                model_metadata=decision.model_metadata,
            )
        return decision

    def build_request(
        self,
        state: AndroidState,
        target: str,
        *,
        elements: Sequence[VisibleElement] | None = None,
        max_new_tokens: int | None = None,
    ) -> VLMRequest:
        """Assemble exactly what a local VLM is allowed to see for one target."""
        return VLMRequest(
            image=np.asarray(state.screenshot.pixels),
            target=self._redact(target),
            screen_description=self._screen_description(state),
            visible_elements=tuple(
                self.visible_elements(state) if elements is None else elements
            ),
            max_new_tokens=max_new_tokens or self.config.vlm_max_new_tokens,
        )

    def visible_elements(self, state: AndroidState) -> tuple[VisibleElement, ...]:
        output: list[VisibleElement] = []
        for node in state.accessibility.nodes:
            if not node.visible or not node.enabled:
                continue
            label = " ".join((node.label or "").split())
            if not label:
                continue
            try:
                box = NormalizedBox.from_device_bounds(
                    node.bounds,
                    state.viewport.width_px,
                    state.viewport.height_px,
                )
            except ValueError:
                continue
            output.append(
                VisibleElement(
                    label=self._redact(label)[:120],
                    role=node.class_name or "unknown",
                    box=box,
                    node_id=node.node_id,
                    clickable=node.clickable,
                    enabled=node.enabled,
                    checked=node.checked,
                )
            )
            if len(output) >= self.config.max_visible_elements:
                break
        return tuple(output)

    def _ground_accessibility(
        self,
        elements: Sequence[VisibleElement],
        target: str,
    ) -> GroundingDecision:
        best: tuple[float, VisibleElement] | None = None
        for element in elements:
            score = _text_score(target, element.label)
            if element.clickable:
                score = min(1.0, score + 0.04)
            if best is None or score > best[0]:
                best = (score, element)
        if best is None or best[0] <= 0:
            return GroundingDecision(GroundingSource.NONE, 0.0)
        score, element = best
        return GroundingDecision(
            source=GroundingSource.ACCESSIBILITY,
            confidence=score,
            label=element.label,
            node_id=element.node_id,
            box=element.box,
            role=element.role,
            explanation="Matched a visible accessibility element",
        )

    def _ground_drlqn(
        self,
        elements: Sequence[VisibleElement],
        target: str,
        scores: Sequence[float] | np.ndarray | None,
    ) -> GroundingDecision:
        if scores is None or not elements:
            return GroundingDecision(GroundingSource.NONE, 0.0)
        values = np.asarray(scores, dtype=float).reshape(-1)
        if values.size == 0:
            return GroundingDecision(GroundingSource.NONE, 0.0)
        values = values[: len(elements)]
        finite = np.isfinite(values)
        if not finite.any():
            return GroundingDecision(GroundingSource.NONE, 0.0)
        safe = np.where(finite, values, -1e9)
        safe = safe - safe.max()
        probabilities = np.exp(np.clip(safe, -60, 0))
        probabilities /= max(probabilities.sum(), 1e-9)
        lexical = np.asarray([_text_score(target, item.label) for item in elements[: len(values)]])
        combined = 0.72 * probabilities + 0.28 * lexical
        index = int(np.argmax(combined))
        element = elements[index]
        return GroundingDecision(
            source=GroundingSource.DRLQN,
            confidence=float(combined[index]),
            label=element.label,
            node_id=element.node_id,
            box=element.box,
            role=element.role,
            explanation="Combined DRLQN node ranking with visible labels",
        )

    @staticmethod
    def parse_vlm_response(raw: str | Mapping[str, object]) -> GroundingDecision:
        payload = json.loads(raw) if isinstance(raw, str) else dict(raw)
        allowed = {"label", "role", "bbox_norm", "confidence", "explanation", "metadata"}
        unknown = set(payload) - allowed
        if unknown:
            raise VLMContractError(f"unknown VLM fields: {sorted(unknown)}")
        bbox = payload.get("bbox_norm")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise VLMContractError("bbox_norm must contain four normalized numbers")
        box = NormalizedBox(*(float(value) for value in bbox))
        confidence = float(payload.get("confidence", 0.0))
        if not 0.0 <= confidence <= 1.0:
            raise VLMContractError("confidence must be in [0, 1]")
        label = str(payload.get("label") or "").strip() or None
        explanation = str(payload.get("explanation") or "").strip()
        check_text = " ".join(filter(None, (label, explanation))).casefold()
        if any(token in check_text for token in _FORBIDDEN_TEXT):
            raise VLMContractError("VLM response contains executable automation text")
        metadata = payload.get("metadata") or {}
        if not isinstance(metadata, Mapping):
            raise VLMContractError("metadata must be an object")
        return GroundingDecision(
            source=GroundingSource.VLM,
            confidence=confidence,
            label=label,
            box=box,
            role=str(payload.get("role") or "") or None,
            explanation=explanation,
            model_metadata=dict(metadata),
        )

    def _screen_description(self, state: AndroidState) -> str:
        labels = [element.label for element in self.visible_elements(state)[:20]]
        return json.dumps(
            {
                "package": state.package,
                "activity": state.activity,
                "visible_labels": labels,
                "width": state.viewport.width_px,
                "height": state.viewport.height_px,
                "rotation": state.viewport.rotation,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )

    def _redact(self, value: str) -> str:
        return SensitiveTextRedactor.redact(value) if self.config.redact_sensitive_text else value
