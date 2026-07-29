"""Local Ollama adapters for the VLM teacher and the chat advisor.

Ollama is used as a *teacher* runtime on the workstation that drives the
emulators. It never runs on the phone: on-device inference stays with the
LiteRT-LM/ONNX Runtime profiles in :mod:`android_drlqn.device_profiles`.

The module speaks the Ollama HTTP API through the standard library only, so
the package keeps its single numpy dependency. Teacher output is still parsed
through the strict :class:`~android_drlqn.vlm.HybridGrounder` JSON contract:
a teacher is untrusted evidence, not an instruction source.
"""
from __future__ import annotations

import base64
import json
import struct
import urllib.error
import urllib.request
import zlib
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from .advisor import ChatMessage, GenerationOptions
from .vlm import JsonGroundingVLM

DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"

# Vision models that are known to follow a strict JSON grounding contract well
# enough to act as a teacher. The list is a hint for operators, not a gate.
SUGGESTED_TEACHERS = (
    "qwen2.5vl:7b",
    "qwen2.5vl:32b",
    "llama3.2-vision:11b",
    "minicpm-v:8b",
)


class OllamaError(RuntimeError):
    """Raised when the local Ollama daemon is unreachable or returns an error."""


def encode_png(pixels: np.ndarray) -> bytes:
    """Encode an ``HxW[x3|x4]`` array as a PNG without a third-party imaging library."""
    image = np.asarray(pixels)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    if image.ndim != 3 or image.shape[-1] not in (3, 4):
        raise ValueError("image must be HxW, HxWx3 or HxWx4")
    image = image[..., :3]
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image, 0.0, 1.0) * 255.0
    elif np.issubdtype(image.dtype, np.integer) and image.dtype != np.uint8:
        maximum = float(np.iinfo(image.dtype).max)
        image = image.astype(np.float64) / maximum * 255.0
    image = np.ascontiguousarray(image.astype(np.uint8))

    height, width = image.shape[:2]
    if height < 1 or width < 1:
        raise ValueError("image must not be empty")
    # Each scanline is prefixed with filter type 0 (None).
    raw = b"".join(b"\x00" + image[row].tobytes() for row in range(height))

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + tag
            + payload
            + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
        )

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )


@dataclass(frozen=True)
class OllamaConfig:
    """Connection and decoding settings for one local Ollama model."""

    model: str
    host: str = DEFAULT_OLLAMA_HOST
    timeout_s: float = 180.0
    temperature: float = 0.0
    top_p: float = 0.9
    num_ctx: int = 4096
    keep_alive: str = "10m"
    seed: int | None = 0
    force_json: bool = True
    max_image_side: int = 1024
    extra_options: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("model must be a non-empty Ollama tag")
        if self.timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if self.max_image_side < 64:
            raise ValueError("max_image_side must be at least 64")

    @property
    def base_url(self) -> str:
        return self.host.rstrip("/")

    def options(self, num_predict: int) -> dict:
        payload = {
            "temperature": float(self.temperature),
            "top_p": float(self.top_p),
            "num_ctx": int(self.num_ctx),
            "num_predict": int(num_predict),
        }
        if self.seed is not None:
            payload["seed"] = int(self.seed)
        payload.update(dict(self.extra_options))
        return payload


class OllamaClient:
    """Thin JSON-over-HTTP client for a local Ollama daemon."""

    def __init__(self, config: OllamaConfig, *, opener=None):
        self.config = config
        self._opener = opener or urllib.request.urlopen

    def _request(self, path: str, payload: Mapping[str, object] | None = None) -> dict:
        url = f"{self.config.base_url}{path}"
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="GET" if data is None else "POST",
        )
        try:
            with self._opener(request, timeout=self.config.timeout_s) as response:
                body = response.read()
        except urllib.error.HTTPError as error:  # pragma: no cover - network path
            detail = error.read().decode("utf-8", "replace")[:400]
            raise OllamaError(f"Ollama {path} failed with HTTP {error.code}: {detail}") from error
        except OSError as error:  # pragma: no cover - network path
            raise OllamaError(
                f"cannot reach Ollama at {self.config.base_url}: {error}. "
                "Start it with `ollama serve`."
            ) from error
        try:
            return json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise OllamaError(f"Ollama {path} returned a non-JSON body") from error

    def installed_models(self) -> tuple[str, ...]:
        payload = self._request("/api/tags")
        return tuple(str(item.get("name", "")) for item in payload.get("models") or ())

    def ensure_model(self) -> None:
        """Fail early with an actionable message instead of mid-run."""
        names = self.installed_models()
        if self.config.model in names:
            return
        stem = self.config.model.split(":")[0]
        if any(name.split(":")[0] == stem for name in names):
            raise OllamaError(
                f"Ollama has {stem} but not the exact tag {self.config.model}; "
                f"run `ollama pull {self.config.model}`"
            )
        raise OllamaError(
            f"model {self.config.model} is not installed; run `ollama pull {self.config.model}`"
        )

    def chat(
        self,
        messages: Sequence[Mapping[str, object]],
        *,
        num_predict: int,
        images: Sequence[bytes] = (),
        force_json: bool | None = None,
        schema: Mapping[str, object] | None = None,
    ) -> str:
        payload: dict[str, object] = {
            "model": self.config.model,
            "messages": _attach_images(messages, images),
            "stream": False,
            "keep_alive": self.config.keep_alive,
            "options": self.config.options(num_predict),
        }
        # A JSON *schema* constrains decoding to the contract's exact keys and
        # types. Plain `format: json` only guarantees syntactic JSON, and small
        # vision models routinely emit stray keys under it.
        if schema is not None:
            payload["format"] = dict(schema)
        elif self.config.force_json if force_json is None else force_json:
            payload["format"] = "json"
        response = self._request("/api/chat", payload)
        message = response.get("message") or {}
        content = message.get("content")
        if not isinstance(content, str):
            raise OllamaError("Ollama response did not contain message.content")
        return content


def _attach_images(
    messages: Sequence[Mapping[str, object]],
    images: Sequence[bytes],
) -> list[dict]:
    output = [dict(message) for message in messages]
    if not images:
        return output
    encoded = [base64.b64encode(item).decode("ascii") for item in images]
    for message in reversed(output):
        if message.get("role") == "user":
            message["images"] = encoded
            return output
    output.append({"role": "user", "content": "", "images": encoded})
    return output


def downscale(image: np.ndarray, max_side: int) -> np.ndarray:
    """Nearest-neighbour downscale so a 1080x2400 frame stays a cheap prompt."""
    array = np.asarray(image)
    if array.ndim < 2:
        raise ValueError("image must have at least two dimensions")
    height, width = array.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return array
    scale = max_side / float(longest)
    new_h = max(1, int(round(height * scale)))
    new_w = max(1, int(round(width * scale)))
    rows = np.minimum(height - 1, (np.arange(new_h) * height / new_h).astype(int))
    cols = np.minimum(width - 1, (np.arange(new_w) * width / new_w).astype(int))
    return array[rows[:, None], cols[None, :]]


class OllamaMultimodalGenerator:
    """`MultimodalGenerator` backed by a local Ollama vision model.

    Wrap it with :class:`~android_drlqn.vlm.JsonGroundingVLM` to obtain a
    `LocalVisionLanguageModel`; the grounding JSON contract is enforced by
    `HybridGrounder.parse_vlm_response`, not by this adapter.
    """

    def __init__(
        self,
        config: OllamaConfig,
        *,
        client: OllamaClient | None = None,
        response_schema: Mapping[str, object] | None = None,
    ):
        self.config = config
        self.client = client or OllamaClient(config)
        self.response_schema = response_schema

    def generate(
        self,
        image: np.ndarray,
        messages: Sequence[Mapping[str, str]],
        *,
        max_new_tokens: int,
    ) -> str:
        png = encode_png(downscale(image, self.config.max_image_side))
        return self.client.chat(
            messages,
            num_predict=max_new_tokens,
            images=(png,),
            schema=self.response_schema,
        )


CONTRACT_FIELDS = ("label", "role", "bbox_norm", "confidence", "explanation", "metadata")

# Ollama structured-output schema for the grounding contract. Constrained
# decoding removes the stray-key and truncated-object failures that plain
# `format: json` leaves in place on small vision models.
GROUNDING_SCHEMA: Mapping[str, object] = {
    "type": "object",
    "properties": {
        "label": {"type": "string"},
        "role": {"type": "string"},
        "bbox_norm": {
            "type": "array",
            "items": {"type": "number"},
            "minItems": 4,
            "maxItems": 4,
        },
        "confidence": {"type": "number"},
        # Bounded so a verbose teacher cannot run past num_predict and emit an
        # unterminated object; Ollama enforces maxLength during decoding.
        "explanation": {"type": "string", "maxLength": 160},
    },
    "required": ["label", "role", "bbox_norm", "confidence", "explanation"],
}


def normalize_grounding_payload(raw: str | Mapping[str, object]) -> dict:
    """Translate one runtime's JSON dialect into the strict grounding contract.

    Observed teacher dialects this repairs:

    * stray or blank keys emitted alongside the contract fields;
    * ``bbox_norm`` on the Qwen-style 0-1000 scale instead of 0-1;
    * a fenced ```json block around the object.

    The function may only drop, rescale or clamp fields. It never invents a
    label, a box or a confidence, and the result is still validated by
    `HybridGrounder.parse_vlm_response`, including the executable-text check.
    """
    if isinstance(raw, str):
        text = raw.strip()
        if text.startswith("```"):
            text = text.strip("`")
            text = text[4:] if text[:4].casefold() == "json" else text
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end < start:
            raise ValueError("teacher response does not contain a JSON object")
        payload = json.loads(text[start : end + 1])
    else:
        payload = dict(raw)
    if not isinstance(payload, Mapping):
        raise ValueError("teacher response is not a JSON object")

    output = {key: payload[key] for key in CONTRACT_FIELDS if key in payload}
    bbox = output.get("bbox_norm")
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        values = [float(item) for item in bbox]
        largest = max(abs(value) for value in values)
        if 1.0 < largest <= 1000.0:
            values = [value / 1000.0 for value in values]
        output["bbox_norm"] = [min(1.0, max(0.0, value)) for value in values]
    if "explanation" in output:
        output["explanation"] = str(output["explanation"])[:240]
    return output


class OllamaGroundingVLM:
    """`LocalVisionLanguageModel` teacher: Ollama generation plus dialect repair."""

    def __init__(self, generator: OllamaMultimodalGenerator):
        self.inner = JsonGroundingVLM(generator)

    @property
    def generator(self) -> OllamaMultimodalGenerator:
        return self.inner.generator

    def ground(self, request) -> Mapping[str, object]:
        return normalize_grounding_payload(self.inner.ground(request))


class OllamaChatModel:
    """`LocalChatModel` backed by a local Ollama text model.

    Used to rehearse the advisor prompt against the teacher stack; production
    devices run the on-device profile instead.
    """

    def __init__(self, config: OllamaConfig, *, client: OllamaClient | None = None):
        self.config = config
        self.client = client or OllamaClient(config)

    def generate(
        self,
        messages: Sequence[ChatMessage],
        options: GenerationOptions,
    ) -> str:
        payload = [{"role": item.role, "content": item.content} for item in messages]
        return self.client.chat(payload, num_predict=options.max_new_tokens)


def ollama_grounding_vlm(
    config: OllamaConfig,
    *,
    client: OllamaClient | None = None,
    verify_installed: bool = True,
) -> OllamaGroundingVLM:
    """Build the teacher `LocalVisionLanguageModel` used by the distiller."""
    generator = OllamaMultimodalGenerator(
        config, client=client, response_schema=GROUNDING_SCHEMA
    )
    if verify_installed:
        generator.client.ensure_model()
    return OllamaGroundingVLM(generator)
