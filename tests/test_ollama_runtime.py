import json

import pytest

from android_drlqn.distillation import (
    EnvironmentOutcome,
    ProposalKind,
    RolloutObservation,
    StudentProposal,
)
from android_drlqn.document_curriculum import CurriculumTask, CurriculumTaskKind
from android_drlqn.emulator_lab import (
    DisplaySpec,
    EmulatorDeviceTier,
    EmulatorKind,
    EmulatorProfile,
    FormFactor,
)
from android_drlqn.ollama_runtime import (
    OllamaClient,
    OllamaEnsembleConfig,
    OllamaRuntimeError,
    OllamaServerConfig,
    OllamaTeacherEnsemble,
    parse_ui_tars_action,
    smoke_image_base64,
)


class FakeOllama:
    def __init__(self, *, critic_action=None, ui_capabilities=("completion", "vision")):
        self.critic_action = critic_action or (
            "Action_Summary: click Bluetooth\n"
            "Action: click(start_box='<|box_start|>(500,250)<|box_end|>')"
        )
        self.ui_capabilities = ui_capabilities
        self.calls = []

    def __call__(self, method, path, payload):
        self.calls.append((method, path, payload))
        if path == "/api/tags":
            return {
                "models": [
                    {"name": "qwen3-vl:30b", "digest": "qwen-digest"},
                    {"name": "ui-tars:latest", "digest": "ui-digest"},
                ]
            }
        if path == "/api/show":
            model = payload["model"]
            capabilities = (
                ("completion", "vision")
                if model == "qwen3-vl:30b"
                else self.ui_capabilities
            )
            return {
                "capabilities": list(capabilities),
                "details": {
                    "parameter_size": "30B" if model == "qwen3-vl:30b" else "7B"
                },
            }
        if path == "/api/chat":
            if payload["model"] == "qwen3-vl:30b":
                content = json.dumps(
                    {
                        "confidence": 0.9,
                        "scores": {
                            "total": 0.8,
                            "grounding": 0.8,
                            "planning": 0.9,
                            "language": 0.8,
                            "safety": 1.0,
                        },
                        "should_update": True,
                        "target": {
                            "kind": "tap",
                            "target_label": "Bluetooth",
                            "bbox_norm": [0.4, 0.2, 0.6, 0.3],
                            "response": "Нажмите «Bluetooth».",
                        },
                        "tags": ["fixture"],
                    }
                )
            else:
                content = self.critic_action
            return {"message": {"role": "assistant", "content": content}, "done": True}
        raise AssertionError(path)


def config():
    return OllamaEnsembleConfig.from_mapping(
        {
            "server": {"base_url": "http://127.0.0.1:11434"},
            "qwen3_vl": {"model": "qwen3-vl:30b"},
            "ui_tars": {"model": "ui-tars:latest"},
            "ensemble": {"ui_tars_weight": 0.25, "min_critic_agreement": 0.35},
        }
    )


def task():
    return CurriculumTask(
        task_id="task-1",
        coverage_id="coverage-1",
        source_key="android_settings",
        os_family="android-aosp",
        snapshot_sha256="abc",
        section_id="bluetooth",
        kind=CurriculumTaskKind.GROUND_UI,
        prompt="Включите Bluetooth",
        expected_output={"display_name": "Bluetooth"},
        evidence={"source_id": "android_settings", "snapshot_sha256": "abc"},
    )


def profile():
    return EmulatorProfile(
        profile_id="aosp-16-phone",
        os_family="android-aosp",
        os_release="16",
        android_version=16,
        api_level=36,
        kind=EmulatorKind.ANDROID_AVD,
        device_class="phone",
        tier=EmulatorDeviceTier.MID,
        form_factor=FormFactor.PHONE,
        architecture="x86_64",
        ram_mb=6144,
        displays=(DisplaySpec(1080, 2400, 420),),
        locales=("ru-RU",),
        navigation_modes=("gesture",),
        system_image="system-images;android-36;google_apis;x86_64",
    )


def observation():
    return RolloutObservation(
        screenshot_reference="base64:" + smoke_image_base64(),
        visible_labels=("Bluetooth", "Wi-Fi"),
        accessibility_summary={"nodes": 2},
        package="com.android.settings",
        activity=".Settings",
        step_index=0,
    )


def proposal(box=(0.4, 0.2, 0.6, 0.3)):
    return StudentProposal(
        kind=ProposalKind.TAP,
        target_label="Bluetooth",
        bbox_norm=box,
        message="Нажмите Bluetooth",
        confidence=0.8,
        action_distribution=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        log_probability=0.0,
        policy_version=0,
    )


def outcome():
    return EnvironmentOutcome(1.0, True, True, None)


def test_server_is_restricted_to_loopback_port_11434():
    with pytest.raises(ValueError):
        OllamaServerConfig(base_url="http://192.168.1.4:11434")
    with pytest.raises(ValueError):
        OllamaServerConfig(base_url="http://127.0.0.1:11435")
    assert OllamaServerConfig().normalized_base_url == "http://127.0.0.1:11434"


def test_preflight_requires_multimodal_capability_for_both_models():
    fake = FakeOllama(ui_capabilities=("completion",))
    ensemble = OllamaTeacherEnsemble(config(), OllamaClient(config().server, fake))
    with pytest.raises(OllamaRuntimeError, match="vision"):
        ensemble.preflight()


def test_qwen_and_ui_tars_ensemble_uses_images_and_agreement():
    fake = FakeOllama()
    ensemble = OllamaTeacherEnsemble(config(), OllamaClient(config().server, fake))
    judgement = ensemble.evaluate(task(), profile(), observation(), proposal(), outcome())
    assert judgement.should_update
    assert judgement.teacher_id == "ollama:qwen3-vl:30b+ollama:ui-tars:latest"
    assert judgement.grounding_score > 0.8
    chat_payloads = [payload for _, path, payload in fake.calls if path == "/api/chat"]
    assert len(chat_payloads) == 2
    assert chat_payloads[0]["messages"][-1]["images"]
    assert chat_payloads[0]["format"]["additionalProperties"] is False
    assert chat_payloads[0]["think"] is False


def test_ui_tars_disagreement_blocks_weight_update():
    fake = FakeOllama(
        critic_action=(
            "Action_Summary: go back\n"
            "Action: navigate_back()"
        )
    )
    ensemble = OllamaTeacherEnsemble(config(), OllamaClient(config().server, fake))
    judgement = ensemble.evaluate(task(), profile(), observation(), proposal(), outcome())
    assert not judgement.should_update
    assert "teacher-disagreement" in judgement.tags


def test_ui_tars_native_coordinates_are_normalized():
    action = parse_ui_tars_action(
        "Action_Summary: select Bluetooth\n"
        "Action: click(start_box='<|box_start|>(750,125)<|box_end|>')",
        coordinate_scale=1000,
    )
    assert action.kind is ProposalKind.TAP
    assert action.point_norm == (0.75, 0.125)
