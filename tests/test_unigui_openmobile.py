from __future__ import annotations

import json
from pathlib import Path

import pytest

from android_drlqn.unigui_openmobile import (
    DATASET_ID,
    gui_owl_sft_record,
    iter_unigui_screen_records,
    write_gui_policy_jsonl,
)


def _episode(root: Path, *, app: str = "settings", package: str = "com.android.settings") -> Path:
    episode = root / "episode-settings-1"
    episode.mkdir(parents=True)
    for index in range(4):
        (episode / f"screenshot_step{index}.png").write_bytes(b"png")
    payload = {
        "task": "OpenMobile",
        "app": app,
        "app_package": package,
        "screen_resolution": [1080, 2400],
        "query": "Open the 'Notification history' screen.",
        "episode_id": "episode-settings-1",
        "device": "mobile",
        "data": [
            {
                "step": 0,
                "action": "I clicked 'Notifications' to open notification settings.",
                "thought": "The visible 'Notifications' row is the correct next screen.",
                "plan": {
                    "name": "mobile_use",
                    "arguments": {"action": "click", "coordinate": [540, 1200]},
                },
                "bbox": [[400, 450], [680, 550]],
                "screenshot": "screenshot_step0.png",
                "is_use": True,
                "is_reviewed": True,
                "train_test": "train",
                "code": "adb shell input tap 540 1200",
            },
            {
                "step": 1,
                "action": "I swiped upward to reveal more settings.",
                "plan": {
                    "name": "mobile_use",
                    "arguments": {
                        "action": "swipe",
                        "coordinate": [500, 800],
                        "coordinate2": [500, 200],
                    },
                },
                "screenshot": "screenshot_step1.png",
                "is_use": True,
                "is_reviewed": True,
                "train_test": "train",
            },
            {
                "step": 2,
                "action": "This unreviewed action must be skipped.",
                "plan": {
                    "name": "mobile_use",
                    "arguments": {"action": "click", "coordinate": [1, 2]},
                },
                "screenshot": "screenshot_step2.png",
                "is_use": True,
                "is_reviewed": False,
                "train_test": "train",
            },
            {
                "step": 3,
                "action": "The requested screen is visible, so terminate successfully.",
                "plan": {
                    "name": "mobile_use",
                    "arguments": {"action": "terminate", "status": "success"},
                },
                "screenshot": "screenshot_step3.png",
                "is_use": True,
                "is_reviewed": True,
                "train_test": "train",
            },
        ],
    }
    (episode / "task.json").write_text(json.dumps(payload), encoding="utf-8")
    return episode


def test_iter_unigui_records_maps_reviewed_actions_and_ignores_raw_code(tmp_path: Path) -> None:
    _episode(tmp_path)
    records = list(iter_unigui_screen_records(tmp_path))
    assert [record.action_name for record in records] == ["mobile.click", "mobile.swipe"]
    assert records[0].action_arguments == {"coordinate": [500, 500]}
    assert records[0].target_bbox == (400, 450, 680, 550)
    assert records[0].visible_labels[0] == "Notifications"
    assert "adb shell" not in json.dumps(gui_owl_sft_record(records[0])).casefold()


def test_include_terminate_and_write_gui_owl_jsonl(tmp_path: Path) -> None:
    _episode(tmp_path)
    records = list(iter_unigui_screen_records(tmp_path, include_terminate=True))
    assert records[-1].action_name == "mobile.terminate"
    output = tmp_path / "training" / "gui.jsonl"
    count = write_gui_policy_jsonl(records, output, source_revision="deadbeef")
    assert count == 3
    first = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
    assert first["dataset"] == DATASET_ID
    assert first["source_revision"] == "deadbeef"
    assert first["episode_id"] == "episode-settings-1"
    assert first["messages"][1]["content"][0]["type"] == "image"
    assert "<tool_call>" in first["messages"][-1]["content"]


def test_app_filter_and_missing_image_are_strict(tmp_path: Path) -> None:
    episode = _episode(tmp_path, app="tasks", package="org.tasks")
    assert list(iter_unigui_screen_records(tmp_path, app="settings")) == []
    (episode / "screenshot_step0.png").unlink()
    with pytest.raises(FileNotFoundError):
        list(iter_unigui_screen_records(tmp_path, app="tasks"))


def test_rejects_privileged_plan_text(tmp_path: Path) -> None:
    episode = _episode(tmp_path)
    payload = json.loads((episode / "task.json").read_text(encoding="utf-8"))
    payload["data"][0]["plan"]["arguments"]["text"] = "run adb shell settings put secure x y"
    (episode / "task.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="privileged"):
        list(iter_unigui_screen_records(tmp_path))
