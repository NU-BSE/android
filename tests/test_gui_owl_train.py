from pathlib import Path

import pytest

from android_drlqn.distillation import ProposalKind, RolloutObservation, TeacherTarget
from android_drlqn.document_curriculum import CurriculumTask, CurriculumTaskKind
from android_drlqn.emulator_lab import selected_emulator_matrix

import project_gui_owl_mrpo_train
import project_gui_owl_train
from project_gui_owl_train import (
    DEFAULT_MODEL_ID,
    build_gui_owl_messages,
    parse_gui_owl_output,
    teacher_gui_owl_output,
)


def task():
    return CurriculumTask(
        "task",
        "coverage",
        "android_settings_api",
        "android-aosp",
        "abc",
        "root",
        CurriculumTaskKind.GROUND_UI,
        "Open Private DNS",
        {
            "display_name": "Private DNS",
            "ui_path": ["Settings", "Network & internet", "Private DNS"],
        },
        {"snapshot": "abc"},
        setting_id="private_dns",
    )


def observation(step=0, screenshot="/tmp/screen.png"):
    return RolloutObservation(
        screenshot,
        ("Network & internet", "Connected devices", "Privacy"),
        {"node_count": 20},
        "com.android.settings",
        ".Settings",
        step,
    )


def flatten_text(message):
    return "\n".join(
        str(item.get("text") or "")
        for item in message["content"]
        if item.get("type") == "text"
    )


def test_default_model_is_gui_owl_15_2b_instruct():
    assert DEFAULT_MODEL_ID == "mPLUG/GUI-Owl-1.5-2B-Instruct"
    assert project_gui_owl_mrpo_train.DEFAULT_MODEL_ID == DEFAULT_MODEL_ID


def test_gui_owl_click_output_maps_1000_coordinates_to_normalized_box():
    text = (
        "Action: Tap Network & internet.\n"
        '<tool_call>{"name":"mobile.click","arguments":'
        '{"coordinate":[250,750],"target_label":"Network & internet"}}</tool_call>'
    )
    proposal = parse_gui_owl_output(text)
    assert proposal is not None
    assert proposal.kind is ProposalKind.TAP
    assert proposal.target_label == "Network & internet"
    x0, y0, x1, y1 = proposal.bbox_norm
    assert (x0 + x1) / 2 == pytest.approx(0.25)
    assert (y0 + y1) / 2 == pytest.approx(0.75)


def test_messages_keep_recent_images_and_compress_older_conclusions():
    history = [
        {
            "screenshot_reference": f"/tmp/{index}.png",
            "conclusion": f"conclusion {index}",
            "output": f"Action: conclusion {index}",
        }
        for index in range(6)
    ]
    messages = build_gui_owl_messages(
        task(), selected_emulator_matrix().profiles[0], observation(), history, history_n=2
    )
    current = flatten_text(messages[-1])
    assert "Compressed earlier action conclusions" in current
    assert "conclusion 0" in current and "conclusion 3" in current
    assert "Documented settings route" in current
    assert sum(message["role"] == "assistant" for message in messages) == 2


def test_teacher_output_round_trips_to_safe_action_mapping():
    target = TeacherTarget(
        ProposalKind.TAP,
        "Private DNS",
        (0.2, 0.4, 0.4, 0.6),
        "Tap Private DNS",
    )
    output = teacher_gui_owl_output(target)
    assert output.startswith("Action: Tap Private DNS")
    proposal = parse_gui_owl_output(output)
    assert proposal is not None
    assert proposal.kind is ProposalKind.TAP
    assert proposal.target_label == "Private DNS"


def test_mrpo_snapshot_is_content_addressed_and_immutable(tmp_path):
    source = tmp_path / "screen.png"
    source.write_bytes(b"first-frame")
    snap = project_gui_owl_mrpo_train.MRPOSafeGUIOwlStudent.snapshot_observation(
        observation(screenshot=str(source))
    )
    snapshot_path = Path(snap.screenshot_reference)
    assert snapshot_path != source
    assert snapshot_path.read_bytes() == b"first-frame"
    source.write_bytes(b"second-frame")
    assert snapshot_path.read_bytes() == b"first-frame"


def test_create_student_uses_default_and_allows_override(monkeypatch, tmp_path):
    captured = {}

    class FakeStudent:
        policy_version = 0

        def __init__(self, model_id, **kwargs):
            captured["model_id"] = model_id

    class FakeUpdater:
        def __init__(self, student, checkpoint_dir, **kwargs):
            captured["checkpoint_dir"] = checkpoint_dir

    monkeypatch.setattr(project_gui_owl_train, "GUIOwlStudent", FakeStudent)
    monkeypatch.setattr(project_gui_owl_train, "GUIOwlLoraUpdater", FakeUpdater)
    project_gui_owl_train.create_student({"resume": False}, tmp_path)
    assert captured["model_id"] == DEFAULT_MODEL_ID
    project_gui_owl_train.create_student(
        {"model_id": "org/custom-gui-owl", "resume": False}, tmp_path
    )
    assert captured["model_id"] == "org/custom-gui-owl"
    assert captured["checkpoint_dir"] == tmp_path / "checkpoints"


def test_mrpo_factory_uses_safe_student_and_updater(monkeypatch, tmp_path):
    captured = {}

    class FakeSafeStudent:
        policy_version = 0

        def __init__(self, model_id, **kwargs):
            captured["model_id"] = model_id

    class FakeSafeUpdater:
        def __init__(self, student, checkpoint_dir, **kwargs):
            captured["checkpoint_dir"] = checkpoint_dir

    monkeypatch.setattr(
        project_gui_owl_mrpo_train, "MRPOSafeGUIOwlStudent", FakeSafeStudent
    )
    monkeypatch.setattr(
        project_gui_owl_mrpo_train, "MRPOSafeGUIOwlLoraUpdater", FakeSafeUpdater
    )
    project_gui_owl_mrpo_train.create_student({"resume": False}, tmp_path)
    assert captured["model_id"] == DEFAULT_MODEL_ID
    assert captured["checkpoint_dir"] == tmp_path / "checkpoints"
