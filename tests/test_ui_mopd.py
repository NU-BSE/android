from __future__ import annotations

import numpy as np
import pytest

from android_drlqn.ui_mopd import (
    PlatformTeacherRouter,
    UIMOPDConfig,
    adaptive_kl_mask,
    clipped_policy_terms,
    group_relative_advantages,
    k3_reverse_kl,
    structured_action_reward,
    ui_mopd_loss,
)


def test_k3_estimator_is_nonnegative_and_zero_for_identical_policies() -> None:
    student = np.log(np.array([[0.7, 0.2], [0.4, 0.8]]))
    teacher = student.copy()
    assert np.allclose(k3_reverse_kl(student, teacher), 0.0)
    shifted = k3_reverse_kl(student, teacher + 0.5)
    assert np.all(shifted >= 0.0)


def test_group_advantage_and_adaptive_mask_follow_paper_equations() -> None:
    rewards = [1.0, -0.5, 1.0, 1.0]
    groups = ["a", "a", "b", "b"]
    advantages = group_relative_advantages(rewards, groups)
    assert np.allclose(advantages, [0.75, -0.75, 0.0, 0.0])
    mask = adaptive_kl_mask(rewards, groups, threshold=0.8)
    assert np.allclose(mask, [1.0, 1.0, 0.0, 0.0])


def test_structured_reward_matches_exact_partial_invalid_levels() -> None:
    assert structured_action_reward(1.0) == 1.0
    assert structured_action_reward(0.5) == -0.5
    assert structured_action_reward(0.0) == -0.5
    assert structured_action_reward(0.0, parsable=False) == -1.0
    with pytest.raises(ValueError):
        structured_action_reward(1.1)


def test_asymmetric_clipping_and_combined_loss() -> None:
    current = np.log(np.array([[1.5, 0.8], [0.5, 1.4]]))
    old = np.zeros_like(current)
    advantages = np.array([1.0, -1.0])
    terms = clipped_policy_terms(
        current,
        old,
        advantages,
        clip_low=0.2,
        clip_high=0.28,
    )
    assert terms[0, 0] == pytest.approx(1.28)
    assert terms[0, 1] == pytest.approx(0.8)
    assert terms[1, 0] == pytest.approx(-0.8)
    assert terms[1, 1] == pytest.approx(-1.4)

    metrics = ui_mopd_loss(
        current_log_probs=current,
        old_log_probs=old,
        teacher_log_probs=current + 0.1,
        advantages=advantages,
        response_mask=np.ones_like(current),
        kl_mask=np.array([1.0, 0.0]),
        config=UIMOPDConfig(beta=0.01),
    )
    assert metrics["kl_tokens"] == 2.0
    assert metrics["mopd_kl_loss"] > 0.0
    assert metrics["loss"] == pytest.approx(
        metrics["policy_loss"] + 0.01 * metrics["mopd_kl_loss"]
    )


def test_platform_router_partitions_rollouts_by_teacher() -> None:
    router = PlatformTeacherRouter(
        {"mobile": "mobile-teacher", "desktop": "desktop-teacher"}
    )
    assert router.partition(["mobile", "desktop", "mobile"]) == {
        "mobile-teacher": (0, 2),
        "desktop-teacher": (1,),
    }
    with pytest.raises(KeyError):
        router.teacher_for("web")
