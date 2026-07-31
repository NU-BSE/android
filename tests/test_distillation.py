from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from android_drlqn.distillation import (
    EnvironmentOutcome,
    OnPolicyConfig,
    ProposalKind,
    ReplayTeacher,
    RolloutObservation,
    StudentProposal,
    TeacherProfile,
    load_distillation_config,
)


def make_proposal(
    kind: ProposalKind = ProposalKind.TAP,
) -> StudentProposal:
    distribution = [0.0] * len(ProposalKind)
    distribution[list(ProposalKind).index(kind)] = 1.0

    return StudentProposal(
        kind=kind,
        target_label="Bluetooth",
        bbox_norm=None,
        message="Open Bluetooth",
        confidence=1.0,
        action_distribution=tuple(distribution),
        log_probability=0.0,
        policy_version=0,
    )


def test_on_policy_config_defaults():
    config = OnPolicyConfig()

    assert config.rollout_batch_size == 16
    assert config.max_steps_per_task == 12
    assert config.gamma == pytest.approx(0.98)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rollout_batch_size": 0},
        {"max_steps_per_task": 0},
        {"updates_per_batch": 0},
        {"gamma": -0.1},
        {"gamma": 1.1},
        {"min_teacher_confidence": -0.1},
        {"min_teacher_confidence": 1.1},
    ],
)
def test_on_policy_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        OnPolicyConfig(**kwargs)


def test_student_proposal_validates_distribution():
    with pytest.raises(ValueError, match="sum to one"):
        StudentProposal(
            kind=ProposalKind.TAP,
            target_label="Bluetooth",
            bbox_norm=None,
            message="Open Bluetooth",
            confidence=1.0,
            action_distribution=(0.2, 0.2),
            log_probability=0.0,
            policy_version=0,
        )


def test_student_proposal_rejects_invalid_bbox():
    distribution = [0.0] * len(ProposalKind)
    distribution[0] = 1.0

    with pytest.raises(ValueError, match="normalized box"):
        StudentProposal(
            kind=ProposalKind.TAP,
            target_label="Bluetooth",
            bbox_norm=(0.8, 0.2, 0.4, 0.9),
            message="Open Bluetooth",
            confidence=1.0,
            action_distribution=tuple(distribution),
            log_probability=0.0,
            policy_version=0,
        )


def test_replay_teacher_returns_fallback_judgement():
    teacher = ReplayTeacher({})
    proposal = make_proposal()

    task = SimpleNamespace(task_id="task-1")
    profile = SimpleNamespace(profile_id="pixel-api-35")
    observation = RolloutObservation(
        screenshot_reference="screen.png",
        visible_labels=("Bluetooth",),
        accessibility_summary={},
        package="com.android.settings",
        activity=".Settings",
        step_index=0,
    )
    outcome = EnvironmentOutcome(
        reward=1.0,
        terminal=True,
        verified=True,
        next_observation=None,
    )

    judgement = teacher.evaluate(
        task,
        profile,
        observation,
        proposal,
        outcome,
    )

    assert judgement.teacher_id == "replay-teacher"
    assert judgement.should_update
    assert judgement.target.preferred_kind is ProposalKind.TAP
    assert "replay-fallback" in judgement.tags


def test_load_distillation_config(tmp_path):
    path = tmp_path / "distillation.json"
    path.write_text(
        json.dumps(
            {
                "teacher": {
                    "profile": "replay",
                    "teacher_id": "local-replay",
                },
                "on_policy": {
                    "rollout_batch_size": 4,
                    "seed": 42,
                },
                "student": {
                    "model": "test-model",
                },
            }
        ),
        encoding="utf-8",
    )

    teacher, policy, student = load_distillation_config(path)

    assert teacher.profile is TeacherProfile.REPLAY
    assert teacher.teacher_id == "local-replay"
    assert policy.rollout_batch_size == 4
    assert policy.seed == 42
    assert student == {"model": "test-model"}
