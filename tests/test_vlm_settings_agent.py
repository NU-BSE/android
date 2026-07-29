import json

import numpy as np

from android_drlqn.core import (
    AccessibilityNode,
    AccessibilitySnapshot,
    AndroidState,
    Bounds,
    ScreenshotFrame,
    ViewportMetadata,
)
from android_drlqn.settings_agent import (
    ActionReceipt,
    ControlPlane,
    EvidenceRef,
    InMemorySettingsCatalog,
    RiskLevel,
    SettingCapability,
    SettingsCoordinator,
    SessionStatus,
    TargetScope,
    VariantScope,
)
from android_drlqn.vlm import GroundingSource, HybridGrounder


def _state(*labels, checked=False):
    nodes = tuple(
        AccessibilityNode(
            node_id=str(index),
            text=label,
            class_name="android.widget.Switch",
            bounds=Bounds(0, index * 100, 1000, index * 100 + 80),
            clickable=True,
            checked=checked,
        )
        for index, label in enumerate(labels)
    )
    return AndroidState(
        screenshot=ScreenshotFrame(1_000, np.zeros((1000, 1000, 3), dtype=np.uint8)),
        accessibility=AccessibilitySnapshot(1_000, nodes, active_window_id=1),
        captions=(),
        package="com.android.settings",
        activity=".Settings",
        viewport=ViewportMetadata(1000, 1000),
    )


class FakeVLM:
    def ground(self, request):
        return json.dumps(
            {
                "label": "Privacy dashboard",
                "role": "button",
                "bbox_norm": [0.1, 0.2, 0.8, 0.3],
                "confidence": 0.91,
                "explanation": "visual match",
                "metadata": {"model": "fake"},
            }
        )


def _capability(risk=RiskLevel.LOW, os_family="grapheneos"):
    return SettingCapability(
        setting_id="privacy_dashboard",
        canonical_name="privacy dashboard",
        display_name="Privacy dashboard",
        aliases=("privacy",),
        scope=VariantScope(os_family, min_android_version=14, max_android_version=17),
        operation="open",
        control_plane=ControlPlane.UI_ONLY,
        ui_path=("Settings", "Privacy", "Privacy dashboard"),
        required_role=None,
        requires_user_interaction=True,
        reversible=True,
        risk_level=risk,
        verification_text="Privacy dashboard",
        evidence=(EvidenceRef("grapheneos_usage", "abc123", 1),),
    )


def test_vlm_fallback_when_tree_has_no_target():
    decision = HybridGrounder().ground(_state("Network"), "Privacy dashboard", vlm=FakeVLM())
    assert decision.source is GroundingSource.VLM
    assert decision.box.center == (0.45, 0.25)


def test_contract_rejects_executable_text():
    raw = {
        "label": "adb shell input tap 1 2",
        "bbox_norm": [0.1, 0.1, 0.2, 0.2],
        "confidence": 0.9,
    }
    try:
        HybridGrounder.parse_vlm_response(raw)
    except ValueError:
        pass
    else:
        raise AssertionError("unsafe VLM output accepted")


def test_catalog_requires_exact_os_scope():
    catalog = InMemorySettingsCatalog([_capability()])
    wrong_target = TargetScope("lineageos", "21", 15, 35, "panther")
    assert catalog.resolve("open privacy dashboard", wrong_target) is None


def test_coordinator_combines_orm_grounding_and_chat():
    coordinator = SettingsCoordinator(InMemorySettingsCatalog([_capability(RiskLevel.HIGH)]))
    target = TargetScope("grapheneos", "2026.07", 15, 35, "panther")

    first = coordinator.start("open privacy dashboard", target, _state("Settings"))
    assert first.status is SessionStatus.WAITING_FOR_RECEIPT
    assert first.proposed_action.target_label == "Settings"

    receipt = ActionReceipt(
        action_id=first.proposed_action.action_id,
        success=True,
        before=_state("Settings"),
        after=_state("Privacy"),
        user_confirmed=True,
    )
    second = coordinator.submit_receipt(first.session, receipt)
    assert second.proposed_action.target_label == "Privacy"


def test_final_high_risk_action_requires_confirmation():
    capability = SettingCapability(
        setting_id="privacy",
        canonical_name="privacy",
        display_name="Privacy",
        aliases=(),
        scope=VariantScope("grapheneos", min_android_version=14),
        operation="enable",
        control_plane=ControlPlane.UI_ONLY,
        ui_path=("Privacy",),
        required_role=None,
        requires_user_interaction=True,
        reversible=True,
        risk_level=RiskLevel.HIGH,
        verification_text="Privacy",
        evidence=(EvidenceRef("source", "snapshot", 1),),
    )
    coordinator = SettingsCoordinator(InMemorySettingsCatalog([capability]))
    target = TargetScope("grapheneos", "1", 15, 35)

    turn = coordinator.start("privacy", target, _state("Privacy"))
    assert turn.status is SessionStatus.WAITING_FOR_CONFIRMATION
    denied = coordinator.confirm_action(turn.session, approved=False)
    assert denied.status is SessionStatus.BLOCKED


def test_perception_facade_exposes_grounding_but_filters_gestures():
    from android_drlqn.advisory_model import PerceptionDRLQN

    class FakeModel:
        def __call__(self, batch, next_state_latent=None):
            return {
                "planner_logits": "plan",
                "ground_node_logits": "nodes",
                "ground_box": "box",
                "ground_heatmap": "heatmap",
                "gesture_type_logits": "gesture",
                "control_points": "points",
            }

    output = PerceptionDRLQN(model=FakeModel())({})
    assert output == {
        "planner_logits": "plan",
        "ground_node_logits": "nodes",
        "ground_box": "box",
        "ground_heatmap": "heatmap",
    }
