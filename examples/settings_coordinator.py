"""Minimal evidence-backed VLM/chat coordination example."""
from __future__ import annotations

import json

import numpy as np

from android_drlqn import (
    AccessibilityNode,
    AccessibilitySnapshot,
    AndroidState,
    Bounds,
    Caption,
    InMemorySettingsCatalog,
    JsonGroundingVLM,
    ScreenshotFrame,
    SettingsCoordinator,
    TargetScope,
    ViewportMetadata,
)


class DemoMultimodalGenerator:
    """Replace with a LiteRT-LM, MediaPipe or ONNX Runtime VLM wrapper."""

    def generate(self, image, messages, *, max_new_tokens):
        del image, messages, max_new_tokens
        return json.dumps(
            {
                "label": "Privacy dashboard",
                "role": "android.widget.TextView",
                "bbox_norm": [0.08, 0.20, 0.92, 0.28],
                "confidence": 0.90,
                "explanation": "Visible privacy setting row",
                "metadata": {"runtime": "demo"},
            }
        )


def demo_state() -> AndroidState:
    return AndroidState(
        screenshot=ScreenshotFrame(1_000, np.zeros((2400, 1080, 3), dtype=np.uint8)),
        accessibility=AccessibilitySnapshot(
            1_000,
            (
                AccessibilityNode(
                    node_id="privacy",
                    text="Privacy",
                    class_name="android.widget.TextView",
                    bounds=Bounds(40, 420, 1040, 540),
                    clickable=True,
                ),
            ),
            active_window_id=1,
        ),
        captions=(Caption(1_000, "Settings"),),
        package="com.android.settings",
        activity=".Settings",
        viewport=ViewportMetadata(1080, 2400, density=3.0),
    )


def main() -> None:
    normalized_document = {
        "document": {
            "source_id": "grapheneos_usage",
            "snapshot_sha256": "replace-with-real-snapshot-sha256",
            "confidence_tier": 1,
        },
        "claims": [
            {
                "claim_kind": "setting_variant",
                "canonical_setting_hint": "privacy_dashboard",
                "display_name": "Privacy dashboard",
                "aliases": ["privacy dashboard", "панель конфиденциальности"],
                "scope": {
                    "os_slug": "grapheneos",
                    "android_version_min": 14,
                    "android_version_max": 17,
                    "device_codenames": [],
                },
                "ui_path": ["Settings", "Privacy", "Privacy dashboard"],
                "control_plane": "ui_only",
                "write_capability": "user_interaction_required",
                "risk_level": "low",
                "reversible": True,
                "operation": "open",
                "evidence": {"source_locator": {"heading": "Privacy"}},
            }
        ],
    }

    catalog = InMemorySettingsCatalog.from_normalized_claims([normalized_document])
    coordinator = SettingsCoordinator(catalog)
    vlm = JsonGroundingVLM(DemoMultimodalGenerator())
    target = TargetScope(
        os_family="grapheneos",
        os_release="2026.07",
        android_version=16,
        api_level=36,
        device_codename="example",
    )
    turn = coordinator.start(
        "Открой панель конфиденциальности",
        target,
        demo_state(),
        vlm=vlm,
    )
    print(turn.status.value, turn.message)
    if turn.proposed_action:
        print(turn.proposed_action.grounding)


if __name__ == "__main__":
    main()
