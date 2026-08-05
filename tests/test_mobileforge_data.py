from __future__ import annotations

import json

from android_drlqn.mobileforge_data import (
    DEFAULT_PACKAGE,
    gui_owl_sft_record,
    iter_mobileforge_screen_records,
    screen_records_from_json,
)


def sample_record():
    return {
        "package": DEFAULT_PACKAGE,
        "goal": "Open Private DNS",
        "screen_name": "Network & internet",
        "screenshot": "screens/step-1.png",
        "visible_labels": ["Network & internet", "Private DNS"],
        "action": {
            "name": "tap",
            "arguments": {
                "coordinate": [0.25, 0.75],
                "target_label": "Private DNS",
            },
        },
    }


def test_screen_record_maps_click_and_coordinates():
    records = list(
        screen_records_from_json(
            sample_record(),
            source_file="/tmp/final/com.android.settings/trajectory.json",
            dataset_root="/tmp",
        )
    )
    assert len(records) == 1
    record = records[0]
    assert record.action_name == "mobile.click"
    assert record.action_arguments["coordinate"] == [250, 750]
    assert record.visible_labels == ("Network & internet", "Private DNS")
    sft = gui_owl_sft_record(record)
    assistant = sft["messages"][-1]["content"]
    assert "<tool_call>" in assistant and '"mobile.click"' in assistant


def test_iterator_never_deserializes_pickle_archives(tmp_path):
    app = tmp_path / "final" / DEFAULT_PACKAGE
    app.mkdir(parents=True)
    (app / "unsafe.pkl.zst").write_bytes(b"not a pickle and must never be read")
    (app / "trajectory.json").write_text(json.dumps(sample_record()), encoding="utf-8")
    records = list(iter_mobileforge_screen_records(tmp_path))
    assert len(records) == 1
    assert records[0].source_file.endswith("trajectory.json")


def test_executable_action_text_is_rejected():
    value = sample_record()
    value["action"] = "adb shell input tap 10 20"
    records = list(
        screen_records_from_json(
            value,
            source_file="/tmp/trajectory.json",
            dataset_root="/tmp",
        )
    )
    assert records == []
