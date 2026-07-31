from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from android_drlqn.distillation import (
    EnvironmentOutcome,
    ProposalKind,
    RolloutObservation,
    StudentProposal,
)
from android_drlqn.document_curriculum import CurriculumTask, CurriculumTaskKind
from android_drlqn.emulator_lab import selected_emulator_matrix
from android_drlqn.ollama_text_critic import (
    BridgeConfig,
    QwenVisionTextCritic,
    TextCriticBridgeError,
)


def task() -> CurriculumTask:
    return CurriculumTask(
        "task",
        "coverage",
        "android_settings_api",
        "android-aosp",
        "abc",
        "root",
        CurriculumTaskKind.GROUND_UI,
        "Ground Bluetooth",
        {"display_name": "Bluetooth"},
        {"snapshot": "abc"},
        setting_id="bluetooth",
    )


def proposal() -> StudentProposal:
    return StudentProposal(
        ProposalKind.TAP,
        "Bluetooth",
        (0.1, 0.1, 0.4, 0.2),
        "Tap Bluetooth",
        0.8,
        (0.8, 0.04, 0.04, 0.04, 0.04, 0.04),
        -0.2,
        0,
    )


def outcome() -> EnvironmentOutcome:
    return EnvironmentOutcome(1.0, True, True, None, "Bluetooth became visible")


def judgement_payload() -> dict[str, object]:
    return {
        "confidence": 0.92,
        "scores": {
            "total": 0.9,
            "grounding": 0.95,
            "planning": 0.9,
            "language": 0.85,
            "safety": 1.0,
        },
        "should_update": True,
        "target": {
            "kind": "tap",
            "target_label": "Bluetooth",
            "bbox_norm": [0.1, 0.1, 0.4, 0.2],
            "response": "Tap Bluetooth",
        },
        "tags": ["grounded"],
    }


class FakeTransport:
    def __init__(self, critic_capabilities=("tools", "completion")) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.critic_capabilities = critic_capabilities

    def __call__(self, endpoint, payload):
        payload = dict(payload)
        self.calls.append((endpoint, payload))
        if endpoint == "/api/show":
            if payload["model"] == "qwen3-vl:4b":
                return {"capabilities": ["completion", "vision"]}
            return {"capabilities": list(self.critic_capabilities)}
        if payload["model"] == "qwen3-vl:4b":
            return {
                "message": {
                    "content": json.dumps(
                        {
                            "screen_summary": "Android Settings showing Bluetooth",
                            "visible_elements": [
                                {
                                    "label": "Bluetooth",
                                    "role": "row",
                                    "bbox_norm": [0.1, 0.1, 0.4, 0.2],
                                    "state": "visible",
                                }
                            ],
                            "uncertainties": [],
                        }
                    )
                }
            }
        return {"message": {"content": json.dumps(judgement_payload())}}


def bridge(tmp_path: Path, transport: FakeTransport) -> QwenVisionTextCritic:
    return QwenVisionTextCritic(
        BridgeConfig(
            vision_model="qwen3-vl:4b",
            critic_model="android-ui-tars-critic:latest",
            allowed_image_roots=(tmp_path.resolve(),),
        ),
        transport=transport,
    )


def observation(tmp_path: Path) -> RolloutObservation:
    image = tmp_path / "screen.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\nfixture")
    return RolloutObservation(
        str(image),
        ("Network & internet", "Bluetooth", "Privacy"),
        {"node_count": 12},
        "com.android.settings",
        ".Settings",
        0,
    )


def test_qwen_receives_image_but_text_critic_does_not(tmp_path):
    transport = FakeTransport()
    evaluator = bridge(tmp_path, transport)
    result = evaluator.evaluate(
        task(), selected_emulator_matrix().profiles[0], observation(tmp_path), proposal(), outcome()
    )

    chat_calls = [payload for endpoint, payload in transport.calls if endpoint == "/api/chat"]
    assert len(chat_calls) == 2
    qwen, critic = chat_calls
    qwen_user = qwen["messages"][1]
    critic_user = critic["messages"][1]
    assert "images" in qwen_user
    assert base64.b64decode(qwen_user["images"][0]).startswith(b"\x89PNG")
    assert "images" not in critic_user
    critic_state = json.loads(critic_user["content"])
    assert critic_state["environment"]["qwen_vl_perception"]["screen_summary"]
    assert "screenshot_reference" not in critic_user["content"]
    assert result.teacher_id.startswith("qwen-vl-text-bridge:")
    assert result.target.preferred_kind is ProposalKind.TAP
    assert "qwen-vl-text-bridge" in result.tags


def test_text_critic_requires_completion_not_vision(tmp_path):
    transport = FakeTransport(critic_capabilities=("tools", "completion"))
    evaluator = bridge(tmp_path, transport)
    evaluator.validate_models()


def test_text_critic_without_completion_fails(tmp_path):
    transport = FakeTransport(critic_capabilities=("tools",))
    evaluator = bridge(tmp_path, transport)
    with pytest.raises(TextCriticBridgeError, match="completion"):
        evaluator.validate_models()


def test_perception_is_cached_by_image_digest(tmp_path):
    transport = FakeTransport()
    evaluator = bridge(tmp_path, transport)
    obs = observation(tmp_path)
    profile = selected_emulator_matrix().profiles[0]
    evaluator.evaluate(task(), profile, obs, proposal(), outcome())
    evaluator.evaluate(task(), profile, obs, proposal(), outcome())
    qwen_calls = [
        payload
        for endpoint, payload in transport.calls
        if endpoint == "/api/chat" and payload["model"] == "qwen3-vl:4b"
    ]
    assert len(qwen_calls) == 1


def test_rejects_image_outside_allowed_roots(tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"\x89PNG\r\n\x1a\nfixture")
    evaluator = QwenVisionTextCritic(
        BridgeConfig(allowed_image_roots=(allowed.resolve(),)),
        transport=FakeTransport(),
    )
    obs = RolloutObservation(str(outside), (), {}, "pkg", "activity", 0)
    with pytest.raises(TextCriticBridgeError, match="outside allowed"):
        evaluator.evaluate(
            task(), selected_emulator_matrix().profiles[0], obs, proposal(), outcome()
        )


def test_rejects_symlinked_screenshot(tmp_path):
    target = tmp_path / "screen.png"
    target.write_bytes(b"\x89PNG\r\n\x1a\nfixture")
    link = tmp_path / "linked.png"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlinks are unavailable")
    evaluator = bridge(tmp_path, FakeTransport())
    obs = RolloutObservation(str(link), (), {}, "pkg", "activity", 0)
    with pytest.raises(TextCriticBridgeError, match="non-symlink"):
        evaluator.evaluate(
            task(), selected_emulator_matrix().profiles[0], obs, proposal(), outcome()
        )
