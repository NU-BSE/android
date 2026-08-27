"""Reference UI-MOPD objectives for platform-routed GUI policy updates.

This module implements the equations from *UI-MOPD: Multi-Platform On-Policy
Distillation for Continual GUI Agent Learning*:

* student-to-teacher on-policy reverse KL with the K3 estimator;
* group-relative outcome advantages;
* adaptive group-level KL masking;
* asymmetric clipped policy optimization; and
* the combined policy-gradient plus MOPD loss.

NumPy helpers make the mathematics easy to unit-test. ``torch_ui_mopd_loss`` is
the differentiable entry point for a real rollout/training engine.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class UIMOPDConfig:
    """Defaults matching the paper's reported Stage-2 setup where specified."""

    beta: float = 0.01
    clip_low: float = 0.20
    clip_high: float = 0.28
    kl_reward_threshold: float = 0.80
    k3_delta_clip: float = 20.0
    rollouts_per_prompt: int = 8

    def __post_init__(self) -> None:
        if self.beta < 0:
            raise ValueError("beta must be non-negative")
        if min(self.clip_low, self.clip_high) < 0:
            raise ValueError("clip bounds must be non-negative")
        if not -1.0 <= self.kl_reward_threshold <= 1.0:
            raise ValueError("kl_reward_threshold must be in [-1, 1]")
        if self.k3_delta_clip <= 0:
            raise ValueError("k3_delta_clip must be positive")
        if self.rollouts_per_prompt < 2:
            raise ValueError("rollouts_per_prompt must be at least 2")


@dataclass(frozen=True)
class PlatformTeacherRouter:
    """Route each rollout to one stable teacher according to its platform."""

    teacher_by_platform: Mapping[str, str]

    def teacher_for(self, platform: str) -> str:
        key = str(platform).strip().casefold()
        normalized = {
            str(name).strip().casefold(): str(teacher)
            for name, teacher in self.teacher_by_platform.items()
        }
        if key not in normalized:
            raise KeyError(f"no UI-MOPD teacher configured for platform: {platform}")
        return normalized[key]

    def partition(self, platforms: Sequence[str]) -> dict[str, tuple[int, ...]]:
        partitions: dict[str, list[int]] = {}
        for index, platform in enumerate(platforms):
            teacher = self.teacher_for(platform)
            partitions.setdefault(teacher, []).append(index)
        return {teacher: tuple(indices) for teacher, indices in partitions.items()}


def _float_array(value: object, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def structured_action_reward(match_fraction: float, *, parsable: bool = True) -> float:
    """Equation 8: exact=1, partial=-0.5, invalid/unparsable=-1."""
    if not parsable:
        return -1.0
    fraction = float(match_fraction)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("match_fraction must be in [0, 1]")
    return 1.0 if fraction == 1.0 else -0.5


def group_relative_advantages(
    rewards: Sequence[float], group_ids: Sequence[object]
) -> np.ndarray:
    """Equation 9: subtract the mean reward of each prompt group."""
    reward_array = _float_array(rewards, name="rewards")
    if reward_array.ndim != 1 or len(reward_array) != len(group_ids):
        raise ValueError("rewards and group_ids must be aligned one-dimensional values")
    output = np.zeros_like(reward_array)
    groups: dict[object, list[int]] = {}
    for index, group_id in enumerate(group_ids):
        groups.setdefault(group_id, []).append(index)
    for indices in groups.values():
        group_values = reward_array[indices]
        output[indices] = group_values - float(group_values.mean())
    return output


def adaptive_kl_mask(
    rewards: Sequence[float],
    group_ids: Sequence[object],
    *,
    threshold: float,
) -> np.ndarray:
    """Equation 6: disable teacher KL when a prompt group is already strong."""
    reward_array = _float_array(rewards, name="rewards")
    if reward_array.ndim != 1 or len(reward_array) != len(group_ids):
        raise ValueError("rewards and group_ids must be aligned one-dimensional values")
    output = np.ones_like(reward_array)
    groups: dict[object, list[int]] = {}
    for index, group_id in enumerate(group_ids):
        groups.setdefault(group_id, []).append(index)
    for indices in groups.values():
        enabled = float(reward_array[indices].mean()) <= float(threshold)
        output[indices] = 1.0 if enabled else 0.0
    return output


def k3_reverse_kl(
    student_log_probs: object,
    teacher_log_probs: object,
    *,
    delta_clip: float = 20.0,
) -> np.ndarray:
    """Equations 4-5: ``exp(delta) - delta - 1`` on student samples."""
    student = _float_array(student_log_probs, name="student_log_probs")
    teacher = _float_array(teacher_log_probs, name="teacher_log_probs")
    if student.shape != teacher.shape:
        raise ValueError("student and teacher log probabilities must have the same shape")
    if delta_clip <= 0:
        raise ValueError("delta_clip must be positive")
    delta = np.clip(teacher - student, -float(delta_clip), float(delta_clip))
    return np.exp(delta) - delta - 1.0


def clipped_policy_terms(
    current_log_probs: object,
    old_log_probs: object,
    advantages: object,
    *,
    clip_low: float = 0.20,
    clip_high: float = 0.28,
) -> np.ndarray:
    """Equation 11 with asymmetric lower/upper policy-ratio clipping."""
    current = _float_array(current_log_probs, name="current_log_probs")
    old = _float_array(old_log_probs, name="old_log_probs")
    if current.shape != old.shape:
        raise ValueError("current and old log probabilities must have the same shape")
    advantage = _float_array(advantages, name="advantages")
    while advantage.ndim < current.ndim:
        advantage = np.expand_dims(advantage, axis=-1)
    try:
        advantage = np.broadcast_to(advantage, current.shape)
    except ValueError as exc:
        raise ValueError("advantages are not broadcastable to token log probabilities") from exc
    ratio = np.exp(np.clip(current - old, -40.0, 40.0))
    clipped = np.clip(ratio, 1.0 - float(clip_low), 1.0 + float(clip_high))
    return np.minimum(ratio * advantage, clipped * advantage)


def ui_mopd_loss(
    *,
    current_log_probs: object,
    old_log_probs: object,
    teacher_log_probs: object,
    advantages: object,
    response_mask: object,
    kl_mask: object,
    config: UIMOPDConfig | None = None,
) -> dict[str, float]:
    """Equations 3 and 10-12 over a batch of sampled response tokens."""
    cfg = config or UIMOPDConfig()
    current = _float_array(current_log_probs, name="current_log_probs")
    old = _float_array(old_log_probs, name="old_log_probs")
    teacher = _float_array(teacher_log_probs, name="teacher_log_probs")
    mask = _float_array(response_mask, name="response_mask")
    if current.shape != old.shape or current.shape != teacher.shape or current.shape != mask.shape:
        raise ValueError("log probabilities and response_mask must have identical shapes")
    if np.any((mask < 0) | (mask > 1)):
        raise ValueError("response_mask must be in [0, 1]")
    token_count = float(mask.sum())
    if token_count <= 0:
        raise ValueError("response_mask selects no response tokens")

    policy_terms = clipped_policy_terms(
        current,
        old,
        advantages,
        clip_low=cfg.clip_low,
        clip_high=cfg.clip_high,
    )
    policy_loss = -float((policy_terms * mask).sum() / token_count)

    sample_kl_mask = _float_array(kl_mask, name="kl_mask")
    while sample_kl_mask.ndim < current.ndim:
        sample_kl_mask = np.expand_dims(sample_kl_mask, axis=-1)
    try:
        sample_kl_mask = np.broadcast_to(sample_kl_mask, current.shape)
    except ValueError as exc:
        raise ValueError("kl_mask is not broadcastable to token log probabilities") from exc
    if np.any((sample_kl_mask < 0) | (sample_kl_mask > 1)):
        raise ValueError("kl_mask must be in [0, 1]")
    active_kl_tokens = mask * sample_kl_mask
    kl_denominator = float(active_kl_tokens.sum())
    if kl_denominator > 0:
        kl_loss = float(
            (
                k3_reverse_kl(current, teacher, delta_clip=cfg.k3_delta_clip)
                * active_kl_tokens
            ).sum()
            / kl_denominator
        )
    else:
        kl_loss = 0.0
    total = policy_loss + cfg.beta * kl_loss
    return {
        "loss": float(total),
        "policy_loss": policy_loss,
        "mopd_kl_loss": kl_loss,
        "beta": float(cfg.beta),
        "response_tokens": token_count,
        "kl_tokens": kl_denominator,
    }


def torch_ui_mopd_loss(
    *,
    current_log_probs: object,
    old_log_probs: object,
    teacher_log_probs: object,
    advantages: object,
    response_mask: object,
    kl_mask: object,
    config: UIMOPDConfig | None = None,
):
    """Differentiable PyTorch implementation of the same combined objective."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for torch_ui_mopd_loss") from exc
    cfg = config or UIMOPDConfig()
    current = current_log_probs
    old = old_log_probs
    teacher = teacher_log_probs
    mask = response_mask.to(dtype=current.dtype)
    advantage = advantages.to(dtype=current.dtype)
    while advantage.ndim < current.ndim:
        advantage = advantage.unsqueeze(-1)
    ratio = torch.exp(torch.clamp(current - old, min=-40.0, max=40.0))
    clipped = torch.clamp(
        ratio,
        min=1.0 - cfg.clip_low,
        max=1.0 + cfg.clip_high,
    )
    policy_terms = torch.minimum(ratio * advantage, clipped * advantage)
    token_count = mask.sum().clamp_min(1.0)
    policy_loss = -(policy_terms * mask).sum() / token_count

    sample_kl_mask = kl_mask.to(dtype=current.dtype)
    while sample_kl_mask.ndim < current.ndim:
        sample_kl_mask = sample_kl_mask.unsqueeze(-1)
    active = mask * sample_kl_mask
    delta = torch.clamp(
        teacher - current,
        min=-cfg.k3_delta_clip,
        max=cfg.k3_delta_clip,
    )
    k3 = torch.exp(delta) - delta - 1.0
    active_count = active.sum()
    kl_loss = torch.where(
        active_count > 0,
        (k3 * active).sum() / active_count.clamp_min(1.0),
        torch.zeros((), dtype=current.dtype, device=current.device),
    )
    total = policy_loss + cfg.beta * kl_loss
    return total, {
        "policy_loss": policy_loss.detach(),
        "mopd_kl_loss": kl_loss.detach(),
        "response_tokens": token_count.detach(),
        "kl_tokens": active_count.detach(),
    }


__all__ = [
    "PlatformTeacherRouter",
    "UIMOPDConfig",
    "adaptive_kl_mask",
    "clipped_policy_terms",
    "group_relative_advantages",
    "k3_reverse_kl",
    "structured_action_reward",
    "torch_ui_mopd_loss",
    "ui_mopd_loss",
]
