import json

import numpy as np
import pytest

from android_drlqn.distillation import (
    EnvironmentOutcome,
    NumpyLinearStudent,
    NumpyPolicyUpdater,
    OnPolicyConfig,
    ProposalKind,
    ReplayTeacher,
    RolloutObservation,
    StrictJsonTeacher,
    StudentProposal,
    TeacherEndpointConfig,
    TeacherJudgement,
    TeacherProfile,
    TeacherTarget,
)
from android_drlqn.document_curriculum import (
    CoverageLedger,
    CurriculumTask,
    CurriculumTaskKind,
    DocumentClass,
    DocumentSection,
    SourceDescriptor,
)
from android_drlqn.emulator_lab import selected_emulator_matrix
from android_drlqn.on_policy import OnPolicyDistiller


def source():
    return SourceDescriptor("android_settings_api", "android-aosp", "https://developer.android.com/reference/android/provider/Settings", 1, "CC-BY-2.5", "official_index", (DocumentClass.API_REFERENCE,), allowed_hosts=("developer.android.com",))


def task(coverage_id="coverage"):
    return CurriculumTask("task", coverage_id, "android_settings_api", "android-aosp", "abc", "root", CurriculumTaskKind.GROUND_UI, "Ground Bluetooth", {"display_name": "Bluetooth"}, {"snapshot": "abc"})


class OneStepEnvironment:
    def reset(self, task, profile):
        return RolloutObservation("fixture://screen", ("Bluetooth",), {}, "com.android.settings", ".Settings", 0)

    def step(self, proposal):
        return EnvironmentOutcome(1.0, True, True, None, "verified")


def judgement(kind=ProposalKind.TAP):
    return TeacherJudgement("teacher", 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, True, TeacherTarget(kind, "Bluetooth", (0.1, 0.1, 0.4, 0.2), "Tap Bluetooth"))


def test_selected_matrix_covers_required_axes():
    report = selected_emulator_matrix().require_complete()
    assert report.profile_count == 9
    assert report.expanded_variant_count == 66
    assert {10, 12, 14, 15, 16, 17}.issubset(set(report.android_versions))
    assert {"android-aosp", "grapheneos", "calyxos", "eos"}.issubset(set(report.os_families))


def test_coverage_is_snapshot_bounded_and_stale_revision_fails():
    section = DocumentSection("android_settings_api", "settings", "abc", "root", DocumentClass.API_REFERENCE, (), {}, True)
    ledger = CoverageLedger((source(),))
    ledger.register_source_revision("android_settings_api", "abc", "def")
    entry = ledger.register_section(section)
    ledger.link_task(entry.section.coverage_id, "task")
    ledger.mark_blocked(entry.section.coverage_id, "emulator artifact unavailable")
    report = ledger.report()
    assert report.terminal_sections == 1
    assert report.stale_sources == ("android_settings_api",)
    assert not report.complete


def test_flux3_profile_requires_explicit_contract_and_endpoint(monkeypatch):
    config = TeacherEndpointConfig(TeacherProfile.FLUX3_EARLY_ACCESS, "flux3", "https://api.bfl.ai/evaluate", "flux-3", "BFL_API_KEY", ("api.bfl.ai",), False)
    monkeypatch.setenv("BFL_API_KEY", "secret")
    with pytest.raises(RuntimeError, match="Early Access"):
        config.validate_for_network()


def test_strict_teacher_rejects_unknown_fields_and_executable_text(monkeypatch):
    monkeypatch.setenv("TEACHER_KEY", "secret")
    config = TeacherEndpointConfig(TeacherProfile.OPENAI_COMPATIBLE, "teacher", "https://teacher.example/evaluate", "model", "TEACHER_KEY", ("teacher.example",), True)
    def unsafe(endpoint, headers, payload):
        return {"confidence": 1, "scores": {"total": 1, "grounding": 1, "planning": 1, "language": 1, "safety": 1}, "should_update": True, "target": {"kind": "tap", "target_label": "Bluetooth", "bbox_norm": None, "response": "adb shell input tap 1 2"}, "tags": []}
    teacher = StrictJsonTeacher(config, unsafe)
    proposal = StudentProposal(ProposalKind.TAP, "Bluetooth", None, "Bluetooth", 1, (1, 0, 0, 0, 0, 0), 0, 0)
    with pytest.raises(ValueError, match="automation"):
        teacher.evaluate(task(), selected_emulator_matrix().profiles[0], OneStepEnvironment().reset(None, None), proposal, OneStepEnvironment().step(proposal))


def test_numpy_on_policy_update_changes_weights(tmp_path):
    matrix = selected_emulator_matrix()
    profile = matrix.profiles[0]
    one_profile_matrix = type(matrix)((profile,))
    student = NumpyLinearStudent(seed=7)
    before = student.weights.copy()
    updater = NumpyPolicyUpdater(student, learning_rate=0.1, checkpoint_dir=tmp_path)
    replay = ReplayTeacher({f"task:{profile.profile_id}:0": judgement()})
    metrics = OnPolicyDistiller(student, replay, updater, lambda _: OneStepEnvironment(), OnPolicyConfig(rollout_batch_size=1, max_steps_per_task=1)).train((task(),), one_profile_matrix)
    # Default MRPO uses a 2x oversampled two-trajectory group. All-success
    # collapse keeps SFT active while zeroing policy advantages.
    assert metrics.rollouts == 4
    assert metrics.updates == 1
    assert metrics.checkpoint_id
    assert student.policy_version == 1
    assert not np.allclose(before, student.weights)
