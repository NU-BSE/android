"""Generate natural screen-search chat SFT data with Alibaba Qwen3.7-Plus.

The module deliberately exposes no Android execution tool. It sends only bounded
screen metadata to the OpenAI-compatible Model Studio endpoint and accepts a
strict JSON object containing natural user questions and assistant answers.
API keys are read from the environment and are never serialized into outputs.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, Mapping, Sequence
from urllib.parse import urlparse

DEFAULT_MODEL = "qwen3.7-plus"
DEFAULT_API_KEY_ENV = "DASHSCOPE_API_KEY"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
_ALLOWED_INTENTS = {"find", "open", "understand", "confirm"}
_FORBIDDEN = (
    "adb shell",
    "shell command",
    "settings put",
    "dispatchgesture",
    "accessibilityservice",
    "devicepolicymanager",
    "root command",
)


@dataclass(frozen=True)
class ScreenContext:
    screen_id: str
    package: str
    screen_name: str
    goal: str
    visible_labels: tuple[str, ...]
    ui_path: tuple[str, ...] = ()
    os_family: str = "android"


@dataclass(frozen=True)
class ScreenQA:
    question: str
    answer: str
    intent: str


Transport = Callable[[str, Mapping[str, str], Mapping[str, object], float], Mapping[str, object]]


def _default_transport(
    endpoint: str,
    headers: Mapping[str, str],
    payload: Mapping[str, object],
    timeout_s: float,
) -> Mapping[str, object]:
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=dict(headers),
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        value = json.loads(response.read().decode("utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("Qwen response must be a JSON object")
    return value


def _endpoint(base_url: str) -> str:
    base = base_url.rstrip("/")
    parsed = urlparse(base)
    if parsed.scheme != "https":
        raise ValueError("Qwen API base URL must use HTTPS")
    host = (parsed.hostname or "").casefold()
    if not (host == "aliyuncs.com" or host.endswith(".aliyuncs.com")):
        raise ValueError("Qwen API host must be an Alibaba Cloud aliyuncs.com host")
    if parsed.query or parsed.fragment or parsed.username or parsed.password:
        raise ValueError("Qwen API base URL must not contain credentials, query or fragment")
    suffix = "/chat/completions"
    return base if parsed.path.endswith(suffix) else base + suffix


def _clean_text(value: object, *, max_chars: int) -> str:
    text = " ".join(str(value or "").split())
    return text[:max_chars].strip()


def _assert_safe(text: str) -> None:
    lowered = text.casefold()
    if any(marker in lowered for marker in _FORBIDDEN):
        raise ValueError("generated answer contains executable or privileged guidance")


def screen_prompt(
    context: ScreenContext,
    *,
    count: int,
    language: str = "English",
) -> str:
    if count < 1 or count > 12:
        raise ValueError("questions per screen must be in [1, 12]")
    route = " > ".join(context.ui_path) or "not supplied"
    labels = ", ".join(context.visible_labels[:40]) or "not supplied"
    return (
        "Create natural user-chat training examples about locating or understanding "
        "one Android screen. Use only the supplied screen facts. Do not invent menu "
        "labels, device state, success, shell commands, ADB, root, accessibility "
        "automation, policy APIs, or settings-provider writes. Answers should be "
        "brief, friendly, and describe manual navigation or what the screen means. "
        f"Write in {language}. Return exactly one JSON object with this schema: "
        '{"items":[{"question":"...","answer":"...",'
        '"intent":"find|open|understand|confirm"}]}. '
        f"Return exactly {count} items.\n"
        f"OS family: {context.os_family}\n"
        f"Package: {context.package}\n"
        f"Screen: {context.screen_name}\n"
        f"Exploration goal: {context.goal or 'not supplied'}\n"
        f"Documented UI path: {route}\n"
        f"Visible labels: {labels}"
    )


class QwenScreenQAClient:
    """Small strict client for Qwen3.7-Plus structured screen QA generation."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        api_key_env: str = DEFAULT_API_KEY_ENV,
        base_url: str | None = None,
        timeout_s: float = 60.0,
        max_retries: int = 3,
        transport: Transport | None = None,
    ) -> None:
        self.model = str(model or DEFAULT_MODEL)
        self.api_key_env = str(api_key_env or DEFAULT_API_KEY_ENV)
        self.base_url = str(base_url or os.getenv("DASHSCOPE_BASE_URL") or DEFAULT_BASE_URL)
        self.endpoint = _endpoint(self.base_url)
        self.timeout_s = float(timeout_s)
        self.max_retries = int(max_retries)
        self.transport = transport or _default_transport
        if self.timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if self.max_retries < 0 or self.max_retries > 8:
            raise ValueError("max_retries must be in [0, 8]")

    def generate(
        self,
        context: ScreenContext,
        *,
        count: int = 4,
        language: str = "English",
    ) -> tuple[ScreenQA, ...]:
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"missing Qwen API key environment variable: {self.api_key_env}")
        payload: dict[str, object] = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You create concise, factual Android settings chat examples. "
                        "Return JSON only and never provide executable automation."
                    ),
                },
                {"role": "user", "content": screen_prompt(context, count=count, language=language)},
            ],
            "response_format": {"type": "json_object"},
            "enable_thinking": False,
            "temperature": 0.7,
            "max_tokens": 1600,
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        last_error: BaseException | None = None
        for attempt in range(self.max_retries + 1):
            try:
                raw = self.transport(self.endpoint, headers, payload, self.timeout_s)
                return self._parse(raw, expected_count=count)
            except (urllib.error.URLError, TimeoutError, ValueError, KeyError, TypeError) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(min(8.0, 0.75 * (2**attempt)))
        raise RuntimeError(f"Qwen screen-QA generation failed: {last_error}")

    @staticmethod
    def _parse(raw: Mapping[str, object], *, expected_count: int) -> tuple[ScreenQA, ...]:
        choices = raw.get("choices")
        if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)) or not choices:
            raise ValueError("Qwen response has no choices")
        first = choices[0]
        if not isinstance(first, Mapping):
            raise ValueError("Qwen choice must be an object")
        message = first.get("message")
        if not isinstance(message, Mapping):
            raise ValueError("Qwen choice has no message")
        content = message.get("content")
        if not isinstance(content, str):
            raise ValueError("Qwen message content must be a JSON string")
        value = json.loads(content)
        if not isinstance(value, Mapping) or set(value) != {"items"}:
            raise ValueError("Qwen output must contain only the items field")
        items = value["items"]
        if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
            raise ValueError("Qwen items must be an array")
        if len(items) != expected_count:
            raise ValueError("Qwen returned the wrong number of items")
        output: list[ScreenQA] = []
        seen: set[tuple[str, str]] = set()
        for item in items:
            if not isinstance(item, Mapping) or set(item) != {"question", "answer", "intent"}:
                raise ValueError("each Qwen item must match the closed schema")
            question = _clean_text(item["question"], max_chars=500)
            answer = _clean_text(item["answer"], max_chars=1200)
            intent = _clean_text(item["intent"], max_chars=32).casefold()
            if not question or not answer or intent not in _ALLOWED_INTENTS:
                raise ValueError("Qwen item has empty text or unsupported intent")
            _assert_safe(question)
            _assert_safe(answer)
            key = (question.casefold(), answer.casefold())
            if key in seen:
                raise ValueError("Qwen returned a duplicate item")
            seen.add(key)
            output.append(ScreenQA(question, answer, intent))
        return tuple(output)


def load_screen_contexts(path: str | Path) -> Iterator[ScreenContext]:
    """Read screen metadata from the prepared MobileForge GUI-policy JSONL."""
    source = Path(path)
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, Mapping):
            raise ValueError(f"{source}:{line_number}: record must be an object")
        screen_id = _clean_text(value.get("id") or value.get("screen_id"), max_chars=128)
        package = _clean_text(value.get("package"), max_chars=200)
        screen_name = _clean_text(value.get("screen_name"), max_chars=300)
        goal = _clean_text(value.get("goal"), max_chars=1000)
        visible = value.get("visible_labels") or ()
        route = value.get("ui_path") or ()
        if not screen_id or not package or not screen_name:
            raise ValueError(f"{source}:{line_number}: missing screen identity")
        if not isinstance(visible, Sequence) or isinstance(visible, (str, bytes)):
            visible = ()
        if not isinstance(route, Sequence) or isinstance(route, (str, bytes)):
            route = ()
        yield ScreenContext(
            screen_id=screen_id,
            package=package,
            screen_name=screen_name,
            goal=goal,
            visible_labels=tuple(_clean_text(item, max_chars=200) for item in visible if _clean_text(item, max_chars=200)),
            ui_path=tuple(_clean_text(item, max_chars=200) for item in route if _clean_text(item, max_chars=200)),
            os_family=_clean_text(value.get("os_family") or "android", max_chars=100),
        )


def chat_sft_records(
    context: ScreenContext,
    items: Iterable[ScreenQA],
    *,
    model: str = DEFAULT_MODEL,
) -> Iterator[dict[str, object]]:
    for index, item in enumerate(items):
        yield {
            "id": f"{context.screen_id}-chat-{index:02d}",
            "screen_id": context.screen_id,
            "package": context.package,
            "screen_name": context.screen_name,
            "generator": {"provider": "alibaba_model_studio", "model": model},
            "intent": item.intent,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Explain Android screens clearly. Give only manual, "
                        "evidence-bounded guidance and never claim an action succeeded."
                    ),
                },
                {"role": "user", "content": item.question},
                {"role": "assistant", "content": item.answer},
            ],
        }


def append_chat_jsonl(
    output: str | Path,
    records: Iterable[Mapping[str, object]],
) -> int:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), ensure_ascii=False) + "\n")
            count += 1
    return count


__all__ = [
    "DEFAULT_API_KEY_ENV",
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "QwenScreenQAClient",
    "ScreenContext",
    "ScreenQA",
    "append_chat_jsonl",
    "chat_sft_records",
    "load_screen_contexts",
    "screen_prompt",
]
