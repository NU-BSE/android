import json

import numpy as np

from android_drlqn.advisor import ChatMessage, PlannerAdvisor, SensitiveTextRedactor
from android_drlqn.core import (
    AccessibilityNode,
    AccessibilitySnapshot,
    AndroidState,
    Bounds,
    Caption,
    ScreenshotFrame,
    ViewportMetadata,
)
from android_drlqn.device_profiles import (
    BenchmarkSample,
    DeviceCapabilities,
    DeviceFamily,
    DeviceProfileSelector,
    DeviceTier,
    RuntimeBackend,
    SocVendor,
)
from android_drlqn.runtime_config import litert_export_manifest, onnxruntime_android_manifest
from android_drlqn.window_graph import PlannerSubgoal, SubgoalType


def _state(*labels):
    nodes = tuple(
        AccessibilityNode(
            node_id=str(index),
            text=label,
            bounds=Bounds(0, index * 40, 500, index * 40 + 36),
            clickable=True,
        )
        for index, label in enumerate(labels)
    )
    return AndroidState(
        screenshot=ScreenshotFrame(1000, np.zeros((80, 40, 3), dtype=np.uint8)),
        accessibility=AccessibilitySnapshot(1000, nodes, active_window_id=1),
        captions=(Caption(1000, "Settings"),),
        package="com.android.settings",
        activity=".Settings",
        viewport=ViewportMetadata(1080, 2400),
    )


class UnsafeModel:
    def generate(self, messages, options):
        assert isinstance(messages[0], ChatMessage)
        return json.dumps({
            "instruction": "adb shell input tap 100 200",
            "expected_result": "done",
            "confirmation_question": "ok?",
            "target_label": "Bluetooth",
            "caution": None,
        })


def test_advisor_emits_one_manual_step_and_waits_for_confirmation():
    plan = (
        PlannerSubgoal(SubgoalType.FIND_SETTING, "Bluetooth"),
        PlannerSubgoal(SubgoalType.CONFIRM),
    )
    advisor = PlannerAdvisor()
    reply = advisor.advise(plan, _state("Network & internet", "Bluetooth", "Save"), "Enable Bluetooth")
    assert reply.step is not None
    assert "Bluetooth" in reply.step.instruction
    assert reply.remaining_plan == plan
    assert advisor.confirm_step(plan, user_confirmed=False) == plan
    assert advisor.confirm_step(plan, user_confirmed=True) == plan[1:]


def test_unsafe_llm_output_falls_back_to_deterministic_guidance():
    plan = (PlannerSubgoal(SubgoalType.FIND_SETTING, "Bluetooth"),)
    reply = PlannerAdvisor().advise(plan, _state("Bluetooth"), "Enable Bluetooth", llm=UnsafeModel())
    assert reply.step is not None
    assert not reply.step.generated_by_llm
    assert "adb" not in reply.step.instruction.casefold()


def test_prompt_redacts_sensitive_ui_text():
    assert SensitiveTextRedactor.redact("OTP код 123456 user@example.com") == "OTP код [secret] [email]"
    messages = PlannerAdvisor().build_messages(
        PlannerSubgoal(SubgoalType.ENTER_VALUE, "123456"),
        _state("Account user@example.com", "OTP 123456"),
        "Enter code 123456",
    )
    assert "123456" not in messages[-1].content
    assert "user@example.com" not in messages[-1].content


def test_samsung_mid_range_does_not_assume_npu():
    device = DeviceCapabilities(
        android_version=14,
        family=DeviceFamily.SAMSUNG,
        tier=DeviceTier.MID,
        soc_vendor=SocVendor.SAMSUNG,
        soc_model="Exynos 1380",
        ram_gb=6,
        has_opencl=True,
    )
    profile = DeviceProfileSelector().select(device)
    assert RuntimeBackend.LITERT_LM_NPU_SAMSUNG not in profile.backend_priority
    assert RuntimeBackend.LITERT_LM_GPU in profile.backend_priority


def test_samsung_high_range_litecore_requires_supported_soc():
    device = DeviceCapabilities(
        android_version=16,
        family=DeviceFamily.SAMSUNG,
        tier=DeviceTier.HIGH,
        soc_vendor=SocVendor.SAMSUNG,
        soc_model="Exynos 2500",
        ram_gb=12,
        cpu_big_cores=4,
        has_samsung_litecore=True,
    )
    profile = DeviceProfileSelector().select(device)
    assert profile.backend_priority[0] is RuntimeBackend.LITERT_LM_NPU_SAMSUNG


def test_oppo_snapdragon_high_range_gets_qnn_with_fallbacks():
    device = DeviceCapabilities(
        android_version=15,
        family=DeviceFamily.OPPO,
        tier=DeviceTier.HIGH,
        soc_vendor=SocVendor.QUALCOMM,
        soc_model="SM8750",
        ram_gb=12,
        cpu_big_cores=4,
        has_qnn_runtime=True,
        has_opencl=True,
    )
    profile = DeviceProfileSelector().select(device)
    assert RuntimeBackend.ONNXRUNTIME_QNN in profile.backend_priority
    assert RuntimeBackend.LITERT_LM_CPU in profile.backend_priority


def test_android_10_and_17_profiles_bound_context_and_memory():
    selector = DeviceProfileSelector()
    old = selector.select(DeviceCapabilities(android_version=10, tier=DeviceTier.MID, ram_gb=6))
    new = selector.select(DeviceCapabilities(android_version=17, tier=DeviceTier.HIGH, ram_gb=12, cpu_big_cores=4))
    assert old.context_tokens <= 640
    assert new.context_tokens <= 1024
    assert new.memory_budget_mb <= int(12 * 1024 * 0.20)


def test_benchmark_rejects_thermal_and_memory_failures():
    profile = DeviceProfileSelector().select(
        DeviceCapabilities(android_version=15, tier=DeviceTier.HIGH, ram_gb=8, cpu_big_cores=4, has_opencl=True)
    )
    samples = [
        BenchmarkSample(RuntimeBackend.LITERT_LM_GPU, 300, 12, 900, thermal_severe=True),
        BenchmarkSample(RuntimeBackend.LITERT_LM_CPU, 700, 6, profile.memory_budget_mb - 20),
    ]
    decision = DeviceProfileSelector().choose_benchmark_winner(profile, samples)
    assert decision.selected is RuntimeBackend.LITERT_LM_CPU
    assert decision.rejected[RuntimeBackend.LITERT_LM_GPU] == "thermal status severe"


def test_build_manifests_are_reviewable_and_do_not_execute_tools(tmp_path):
    device = DeviceCapabilities(
        android_version=15,
        family=DeviceFamily.REALME,
        tier=DeviceTier.HIGH,
        soc_vendor=SocVendor.QUALCOMM,
        soc_model="SM8750",
        ram_gb=12,
        cpu_big_cores=4,
        has_qnn_runtime=True,
    )
    profile = DeviceProfileSelector().select(device)
    lite = litert_export_manifest(profile, device)
    ort = onnxruntime_android_manifest(profile, device)
    assert profile.model.value in lite.commands[0]
    assert "--android" in ort.commands[0]
    assert "-keep class ai.onnxruntime.**" in ort.proguard_rules[0]


def test_advisory_model_filters_actionable_outputs():
    from android_drlqn.advisory_model import AdvisoryDRLQN

    class FakeModel:
        def __call__(self, batch, next_state_latent=None):
            return {
                "planner_logits": "plan",
                "ground_node_logits": "nodes",
                "gesture_type_logits": "gesture",
                "control_points": "coordinates",
            }

    safe = AdvisoryDRLQN(model=FakeModel())({})
    assert safe == {"planner_logits": "plan", "ground_node_logits": "nodes"}
