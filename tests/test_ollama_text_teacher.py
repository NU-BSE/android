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
from android_drlqn.ollama_text_teacher import (
    TextTeacherConfig,
    TextOnlyOllamaTeacher,
    TextOnlyTeacherError,
    install_text_only_teacher,
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


def teacher(tmp_path: Path, transport: FakeTransport) -> TextOnlyOllamaTeacher:
    return TextOnlyOllamaTeacher(
        config=TextTeacherConfig(
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
    evaluator = teacher(tmp_path, transport)
    result = evaluator.evaluate(
        task(), selected_emulator_matrix().profiles[0], observation(tmp_path), proposal(), outcome()
    )

    chat_calls = [
        payload
        for endpoint, payload in transport.calls
        if endpoint == "/api/chat"
    ]
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
    assert result.teacher_id.startswith("qwen-vl-text-teacher:")
    assert result.target.preferred_kind is ProposalKind.TAP
    assert "qwen-vl-perception" in result.tags
    assert "text-only-ui-tars" in result.tags


def test_text_critic_requires_completion_not_vision(tmp_path):
    transport = FakeTransport(critic_capabilities=("tools", "completion"))
    evaluator = teacher(tmp_path, transport)
    evaluator.validate_models()


def test_text_critic_without_completion_fails(tmp_path):
    transport = FakeTransport(critic_capabilities=("tools",))
    with pytest.raises(TextOnlyTeacherError, match="completion"):
        teacher(tmp_path, transport)


def test_perception_is_cached_by_image_digest(tmp_path):
    transport = FakeTransport()
    evaluator = teacher(tmp_path, transport)
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
    evaluator = TextOnlyOllamaTeacher(
        TextTeacherConfig(allowed_image_roots=(allowed.resolve(),)),
        transport=FakeTransport(),
    )
    obs = RolloutObservation(str(outside), (), {}, "pkg", "activity", 0)
    with pytest.raises(TextOnlyTeacherError, match="outside allowed"):
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
    evaluator = teacher(tmp_path, FakeTransport())
    obs = RolloutObservation(str(link), (), {}, "pkg", "activity", 0)
    with pytest.raises(TextOnlyTeacherError, match="non-symlink"):
        evaluator.evaluate(
            task(), selected_emulator_matrix().profiles[0], obs, proposal(), outcome()
        )


def test_install_replaces_only_ollama_teacher():
    class OriginalTeacher:
        def evaluate(self):
            raise NotImplementedError

    class Runtime:
        OllamaTeacher = OriginalTeacher

    install_text_only_teacher(Runtime)
    assert Runtime.OllamaTeacher is TextOnlyOllamaTeacher
    assert Runtime.VisionOllamaTeacher is OriginalTeacher
    install_text_only_teacher(Runtime)
    assert Runtime.VisionOllamaTeacher is OriginalTeacher


def test_install_supports_runtime_without_ollama_teacher_export():
    class Runtime:
        pass

    install_text_only_teacher(Runtime)

    assert Runtime.OllamaTeacher is TextOnlyOllamaTeacher
    assert Runtime._text_only_teacher_replacements == ("OllamaTeacher",)


def test_install_replaces_actual_evaluator_name_without_touching_config():
    class OllamaTeacherConfig:
        pass

    class OllamaMultimodalEvaluator:
        def evaluate(self):
            raise NotImplementedError

    # The real evaluator name need not be exactly OllamaTeacher; an evaluator
    # export is enough. This mirrors runtime version drift.
    OllamaMultimodalEvaluator.__name__ = "OllamaVisionTeacher"

    class Runtime:
        TeacherConfig = OllamaTeacherConfig
        OllamaVisionEvaluator = OllamaMultimodalEvaluator

    install_text_only_teacher(Runtime)

    assert Runtime.OllamaVisionEvaluator is TextOnlyOllamaTeacher
    assert Runtime.VisionOllamaVisionEvaluator is OllamaMultimodalEvaluator
    assert Runtime.TeacherConfig is OllamaTeacherConfig
    assert Runtime.OllamaTeacher is TextOnlyOllamaTeacher


def test_runtime_config_supplies_native_url_and_model_names(monkeypatch, tmp_path):
    for name in (
        "ANDROID_DRLQN_OLLAMA_BASE_URL",
        "ANDROID_DRLQN_QWEN_VL_MODEL",
        "ANDROID_DRLQN_UI_TARS_CRITIC_MODEL",
        "ANDROID_DRLQN_OLLAMA_TIMEOUT_S",
    ):
        monkeypatch.delenv(name, raising=False)

    runtime_config = {
        "ollama": {
            "base_url": "http://127.0.0.1:11434/api/",
            "timeout_s": 45,
            "models": {
                "vlm": {
                    "name": "local-qwen-vl:latest",
                    "required_capabilities": ["vision", "completion"],
                },
                "critic": {
                    "name": "local-ui-tars-critic:latest",
                    "required_capabilities": ["completion"],
                },
            },
        }
    }

    config = TextTeacherConfig.from_runtime(runtime_config, tmp_path)

    assert config.base_url == "http://127.0.0.1:11434"
    assert config.vision_model == "local-qwen-vl:latest"
    assert config.critic_model == "local-ui-tars-critic:latest"
    assert config.timeout_s == 45


def test_environment_model_names_override_runtime_config(monkeypatch, tmp_path):
    monkeypatch.setenv("ANDROID_DRLQN_QWEN_VL_MODEL", "env-vl:latest")
    monkeypatch.setenv("ANDROID_DRLQN_UI_TARS_CRITIC_MODEL", "env-critic:latest")
    runtime_config = {
        "models": {
            "vlm": {"name": "runtime-vl:latest"},
            "critic": {"name": "runtime-critic:latest"},
        }
    }

    config = TextTeacherConfig.from_runtime(runtime_config, tmp_path)

    assert config.vision_model == "env-vl:latest"
    assert config.critic_model == "env-critic:latest"


def test_teacher_constructor_uses_runtime_config(monkeypatch, tmp_path):
    for name in (
        "ANDROID_DRLQN_OLLAMA_BASE_URL",
        "ANDROID_DRLQN_QWEN_VL_MODEL",
        "ANDROID_DRLQN_UI_TARS_CRITIC_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)

    calls = []

    def transport(endpoint, payload):
        calls.append((endpoint, dict(payload)))
        if payload["model"] == "runtime-vl:latest":
            return {"capabilities": ["vision", "completion"]}
        return {"capabilities": ["tools", "completion"]}

    runtime_config = {
        "base_url": "http://127.0.0.1:11434/v1",
        "models": {
            "vlm": {"model": "runtime-vl:latest"},
            "critic": {"model": "runtime-critic:latest"},
        },
    }
    evaluator = TextOnlyOllamaTeacher(
        runtime_config,
        object(),
        transport=transport,
        cwd=tmp_path,
    )

    assert evaluator.config.base_url == "http://127.0.0.1:11434"
    assert evaluator.config.vision_model == "runtime-vl:latest"
    assert evaluator.config.critic_model == "runtime-critic:latest"
    assert [payload["model"] for _, payload in calls] == [
        "runtime-vl:latest",
        "runtime-critic:latest",
    ]


def test_native_transport_reports_ollama_error_body(monkeypatch):
    from io import BytesIO
    from urllib.error import HTTPError

    import android_drlqn.ollama_text_teacher as module
    from android_drlqn.ollama_text_teacher import NativeOllamaTransport

    def fail(request, timeout):
        del timeout
        raise HTTPError(
            request.full_url,
            404,
            "Not Found",
            {},
            BytesIO(b'{"error":"model \'missing:latest\' not found"}'),
        )

    monkeypatch.setattr(module, "urlopen", fail)
    transport = NativeOllamaTransport("http://127.0.0.1:11434", 2)

    with pytest.raises(TextOnlyTeacherError) as raised:
        transport("/api/show", {"model": "missing:latest"})

    message = str(raised.value)
    assert "HTTP 404" in message
    assert "missing:latest" in message
    assert "model 'missing:latest' not found" in message
    assert "http://127.0.0.1:11434/api/show" in message



def test_preflight_returns_cached_dataclass_statuses(tmp_path):
    from dataclasses import asdict

    calls = []

    def transport(endpoint, payload):
        calls.append((endpoint, dict(payload)))
        if payload["model"] == "qwen3-vl:4b":
            return {"capabilities": ["completion", "vision"]}
        return {"capabilities": ["tools", "completion"]}

    evaluator = TextOnlyOllamaTeacher(
        TextTeacherConfig(allowed_image_roots=(tmp_path.resolve(),)),
        transport=transport,
    )
    constructor_calls = len(calls)

    statuses = evaluator.preflight()

    assert len(statuses) == 2
    assert len(calls) == constructor_calls
    assert [status.role for status in statuses] == ["vision", "text_critic"]
    assert all(status.available and status.ready for status in statuses)
    assert statuses[0].required_capabilities == ("completion", "vision")
    assert statuses[1].required_capabilities == ("completion",)
    assert asdict(statuses[1])["model"] == "android-ui-tars-critic:latest"


def test_preflight_revalidates_when_cache_is_cleared(tmp_path):
    calls = []

    def transport(endpoint, payload):
        calls.append((endpoint, dict(payload)))
        if payload["model"] == "qwen3-vl:4b":
            return {"capabilities": ["completion", "vision"]}
        return {"capabilities": ["completion"]}

    evaluator = TextOnlyOllamaTeacher(
        TextTeacherConfig(allowed_image_roots=(tmp_path.resolve(),)),
        transport=transport,
    )
    evaluator._validated = False
    evaluator._preflight_statuses = ()
    before = len(calls)

    statuses = evaluator.preflight()

    assert len(calls) == before + 2
    assert len(statuses) == 2


def test_judgement_recomputes_total_when_critic_returns_component_sum(tmp_path):
    evaluator = TextOnlyOllamaTeacher(
        TextTeacherConfig(allowed_image_roots=(tmp_path.resolve(),)),
        transport=lambda endpoint, payload: {
            "capabilities": ["completion", "vision"]
            if payload["model"] == "qwen3-vl:4b"
            else ["completion"]
        },
    )
    raw = judgement_payload()
    raw["scores"]["total"] = 3.7

    judgement = evaluator._judgement(raw)

    assert judgement.total_score == pytest.approx(0.925)
    assert "critic-total-recomputed" in judgement.tags


def test_judgement_normalizes_common_ten_point_scale(tmp_path):
    evaluator = TextOnlyOllamaTeacher(
        TextTeacherConfig(allowed_image_roots=(tmp_path.resolve(),)),
        transport=lambda endpoint, payload: {
            "capabilities": ["completion", "vision"]
            if payload["model"] == "qwen3-vl:4b"
            else ["completion"]
        },
    )
    raw = judgement_payload()
    raw["confidence"] = 9
    raw["scores"] = {
        "total": 8.5,
        "grounding": 9,
        "planning": 8,
        "language": 7,
        "safety": 10,
    }

    judgement = evaluator._judgement(raw)

    assert judgement.confidence == pytest.approx(0.9)
    assert judgement.total_score == pytest.approx(0.85)
    assert judgement.grounding_score == pytest.approx(0.9)
    assert judgement.safety_score == pytest.approx(1.0)
    assert "critic-component-scale-10" in judgement.tags
    assert "critic-total-scale-10" in judgement.tags
    assert "critic-confidence-scale-10" in judgement.tags


def test_judgement_rejects_mixed_component_scales(tmp_path):
    evaluator = TextOnlyOllamaTeacher(
        TextTeacherConfig(allowed_image_roots=(tmp_path.resolve(),)),
        transport=lambda endpoint, payload: {
            "capabilities": ["completion", "vision"]
            if payload["model"] == "qwen3-vl:4b"
            else ["completion"]
        },
    )
    raw = judgement_payload()
    raw["scores"] = {
        "total": 0.8,
        "grounding": 0.9,
        "planning": 8,
        "language": 7,
        "safety": 10,
    }

    with pytest.raises(TextOnlyTeacherError, match="inconsistent"):
        evaluator._judgement(raw)


def test_judgement_rejects_non_finite_score(tmp_path):
    evaluator = TextOnlyOllamaTeacher(
        TextTeacherConfig(allowed_image_roots=(tmp_path.resolve(),)),
        transport=lambda endpoint, payload: {
            "capabilities": ["completion", "vision"]
            if payload["model"] == "qwen3-vl:4b"
            else ["completion"]
        },
    )
    raw = judgement_payload()
    raw["scores"]["total"] = float("nan")

    with pytest.raises(TextOnlyTeacherError, match="not finite"):
        evaluator._judgement(raw)


def test_truncated_critic_json_is_retried_once_without_images(tmp_path):
    class RetryTransport(FakeTransport):
        def __init__(self):
            super().__init__()
            self.critic_attempts = 0

        def __call__(self, endpoint, payload):
            if endpoint == "/api/show" or payload["model"] == "qwen3-vl:4b":
                return super().__call__(endpoint, payload)
            payload = dict(payload)
            self.calls.append((endpoint, payload))
            self.critic_attempts += 1
            if self.critic_attempts == 1:
                return {
                    "message": {
                        "content": '{"confidence":0.9,"scores":{"total":0.8'
                    },
                    "done": True,
                    "done_reason": "length",
                }
            return {
                "message": {"content": json.dumps(judgement_payload())},
                "done": True,
                "done_reason": "stop",
            }

    transport = RetryTransport()
    evaluator = teacher(tmp_path, transport)

    result = evaluator.evaluate(
        task(),
        selected_emulator_matrix().profiles[0],
        observation(tmp_path),
        proposal(),
        outcome(),
    )

    chat_calls = [
        payload
        for endpoint, payload in transport.calls
        if endpoint == "/api/chat"
    ]
    assert len(chat_calls) == 3
    vision_call, first_critic, retry_critic = chat_calls
    assert "images" in vision_call["messages"][1]
    assert "images" not in first_critic["messages"][1]
    assert "images" not in retry_critic["messages"][1]
    assert first_critic["options"]["num_predict"] == 1024
    assert retry_critic["options"]["num_predict"] == 1024
    retry_state = json.loads(retry_critic["messages"][1]["content"])
    assert retry_state["retry"]["done_reason"] == "length"
    assert "partial_response" in retry_state["retry"]
    assert "screenshot_reference" not in retry_critic["messages"][1]["content"]
    assert "critic-json-retried" in result.tags


def test_invalid_json_twice_reports_response_metadata(tmp_path):
    class BrokenTransport(FakeTransport):
        def __call__(self, endpoint, payload):
            if endpoint == "/api/show" or payload["model"] == "qwen3-vl:4b":
                return super().__call__(endpoint, payload)
            self.calls.append((endpoint, dict(payload)))
            return {
                "message": {"content": '{"confidence":"unterminated'},
                "done": True,
                "done_reason": "length",
            }

    evaluator = teacher(tmp_path, BrokenTransport())
    with pytest.raises(TextOnlyTeacherError) as raised:
        evaluator.evaluate(
            task(),
            selected_emulator_matrix().profiles[0],
            observation(tmp_path),
            proposal(),
            outcome(),
        )

    message = str(raised.value)
    assert "invalid JSON twice" in message
    assert "done_reason='length'" in message
    assert "chars=" in message
    assert "column=" in message


def test_fenced_json_content_is_accepted(tmp_path):
    class FencedTransport(FakeTransport):
        def __call__(self, endpoint, payload):
            if endpoint == "/api/show" or payload["model"] == "qwen3-vl:4b":
                return super().__call__(endpoint, payload)
            self.calls.append((endpoint, dict(payload)))
            return {
                "message": {
                    "content": "```json\n" + json.dumps(judgement_payload()) + "\n```"
                }
            }

    evaluator = teacher(tmp_path, FencedTransport())
    result = evaluator.evaluate(
        task(),
        selected_emulator_matrix().profiles[0],
        observation(tmp_path),
        proposal(),
        outcome(),
    )
    assert result.total_score == pytest.approx(0.9)


def test_critic_num_predict_can_be_overridden(monkeypatch, tmp_path):
    monkeypatch.setenv("ANDROID_DRLQN_CRITIC_NUM_PREDICT", "1536")
    monkeypatch.setenv("ANDROID_DRLQN_CRITIC_RETRY_NUM_PREDICT", "2048")

    config = TextTeacherConfig.from_environment(tmp_path)

    assert config.critic_num_predict == 1536
    assert config.critic_retry_num_predict == 2048


def test_critic_payload_is_compact_for_large_nested_state():
    from types import SimpleNamespace

    huge = {
        f"key-{index}": {
            f"nested-{child}": "x" * 5_000
            for child in range(40)
        }
        for index in range(80)
    }
    fake_task = SimpleNamespace(
        task_id="task",
        kind=SimpleNamespace(value="ground_ui"),
        prompt="p" * 10_000,
        expected_output=huge,
        evidence=huge,
    )
    fake_profile = SimpleNamespace(
        profile_id="profile",
        os_family="android-aosp",
        android_version="14",
        api_level=34,
    )
    fake_observation = SimpleNamespace(
        package="pkg",
        activity="activity",
        step_index=0,
        visible_labels=tuple(
            "label-" + str(i) + "-" + "y" * 500 for i in range(100)
        ),
        accessibility_summary=huge,
    )
    fake_proposal = SimpleNamespace(
        kind=SimpleNamespace(value="tap"),
        target_label="Bluetooth",
        bbox_norm=(0.1, 0.1, 0.4, 0.2),
        message="Tap Bluetooth",
        confidence=0.8,
    )
    fake_outcome = SimpleNamespace(
        reward=1.0,
        terminal=True,
        verified=True,
        reason="verified",
    )
    perception = {
        "screen_summary": "summary" * 500,
        "visible_elements": [
            {"label": "z" * 1_000, "role": "row", "state": "visible"}
            for _ in range(100)
        ],
        "uncertainties": ["u" * 1_000 for _ in range(50)],
    }

    payload = TextOnlyOllamaTeacher._critic_payload(
        fake_task,
        fake_profile,
        fake_observation,
        fake_proposal,
        fake_outcome,
        perception,
    )
    encoded = json.dumps(payload, ensure_ascii=False)

    assert len(encoded) < 45_000
    assert len(payload["environment"]["visible_labels"]) == 24
