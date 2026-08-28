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


def test_mobile_type_parses_executes_and_round_trips():
    """mobile.type was declared by the contract but had no ProposalKind.

    parse_gui_owl_output returned None for it, so a student that obeyed its
    instructions was scored as a parse failure while a click-only student was
    not -- 9 of 256 held-out targets on the depth-24 bootstrap.
    """
    from android_drlqn.distillation import ProposalKind
    from project_gui_owl_train import (
        TeacherTarget,
        parse_gui_owl_output,
        teacher_gui_owl_output,
    )

    proposal = parse_gui_owl_output(
        'Action: I typed the network name into the field.\n'
        '<tool_call>{"name":"mobile.type","arguments":{"text":"AndroidWifi"}}</tool_call>'
    )
    assert proposal is not None
    assert proposal.kind is ProposalKind.TYPE
    # The typed string is the payload, not the Action conclusion.
    assert proposal.text == "AndroidWifi"
    assert proposal.message.startswith("I typed")

    # A type call with nothing to enter is not actionable.
    assert parse_gui_owl_output(
        'Action: x\n<tool_call>{"name":"mobile.type","arguments":{"text":"  "}}</tool_call>'
    ) is None

    serialized = teacher_gui_owl_output(
        TeacherTarget(
            preferred_kind=ProposalKind.TYPE,
            target_label=None,
            bbox_norm=None,
            response="AndroidWifi",
        )
    )
    assert '"name":"mobile.type"' in serialized
    again = parse_gui_owl_output(serialized)
    assert again is not None and again.kind is ProposalKind.TYPE
    assert again.text == "AndroidWifi"


def test_proposal_kinds_keep_distinct_leading_characters():
    """The closed action head selects a kind by its value's first subword token.

    A char-level tokenizer makes that the first character, so "type" would have
    collided with "tap"; the TYPE member's value is "input_text" for that
    reason. Appending it last also keeps every existing index stable.
    """
    from android_drlqn.distillation import ProposalKind

    values = [kind.value for kind in ProposalKind]
    assert len({value[0] for value in values}) == len(values)
    assert list(ProposalKind)[-1] is ProposalKind.TYPE
    assert ProposalKind.TYPE.value == "input_text"
