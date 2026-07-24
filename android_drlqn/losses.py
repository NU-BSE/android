"""Q(lambda) and multitask losses for Android trajectories."""
from __future__ import annotations

import numpy as np

from .autodiff import T


def peng_q_lambda_targets(rewards, discounts, bootstrap_q, trace_lambda=0.9, terminals=None):
    """Compute truncated Peng Q(lambda) returns for arrays shaped ``(B, T)``."""
    rewards = np.asarray(rewards, dtype=float)
    discounts = np.asarray(discounts, dtype=float)
    bootstrap_q = np.asarray(bootstrap_q, dtype=float)
    if rewards.shape != discounts.shape or rewards.shape != bootstrap_q.shape:
        raise ValueError("rewards, discounts and bootstrap_q must have the same shape")
    terminals = (
        np.zeros_like(rewards, dtype=bool)
        if terminals is None
        else np.asarray(terminals, dtype=bool)
    )
    if terminals.shape != rewards.shape:
        raise ValueError("terminals must have the same shape as rewards")

    output = np.zeros_like(rewards)
    carry = bootstrap_q[:, -1]
    lam = float(trace_lambda)
    for step in range(rewards.shape[1] - 1, -1, -1):
        gamma = discounts[:, step] * (~terminals[:, step])
        one_step = rewards[:, step] + gamma * bootstrap_q[:, step]
        carry = (1 - lam) * one_step + lam * (rewards[:, step] + gamma * carry)
        output[:, step] = carry
    return output


def huber(error: T, delta: float = 1.0) -> T:
    absolute = (error * error + 1e-12) ** 0.5
    quadratic = (absolute.data <= delta).astype(float)
    return T(quadratic) * 0.5 * error * error + T(1 - quadratic) * delta * (
        absolute - 0.5 * delta
    )


def residual_q_loss(
    q_selected: T,
    target_values,
    q_next_online: T | None = None,
    q_next_target: T | None = None,
    residual_mix: float = 0.1,
) -> T:
    """Stable target-network loss plus a small Bellman-residual correction."""
    target = T(np.asarray(target_values, dtype=float))
    semi_gradient = huber(q_selected - target).mean()
    if q_next_online is None or q_next_target is None or residual_mix <= 0:
        return semi_gradient
    correction = huber(q_next_online - q_next_target.detach()).mean()
    return (1 - residual_mix) * semi_gradient + residual_mix * correction


def binary_cross_entropy_with_logits(logits: T, labels) -> T:
    labels = T(np.asarray(labels, dtype=float))
    probabilities = logits.sigmoid()
    epsilon = 1e-7
    return -(
        labels * (probabilities + epsilon).log()
        + (1 - labels) * (1 - probabilities + epsilon).log()
    ).mean()


def android_multitask_loss(outputs, targets, weights=None):
    """Compose planner, grounding, gesture, verifier and world-model objectives."""
    weights = weights or {}
    losses = {}

    def cross_entropy(logits: T, labels) -> T:
        labels = np.asarray(labels, dtype=int)
        probabilities = logits.softmax(axis=-1)
        rows = np.arange(labels.shape[0])
        return -(probabilities[rows, labels] + 1e-9).log().mean()

    categorical = (
        ("planner", "planner_logits"),
        ("node", "ground_node_logits"),
        ("gesture", "gesture_type_logits"),
        ("fingers", "finger_count_logits"),
    )
    for target_name, output_name in categorical:
        if target_name in targets:
            losses[target_name] = cross_entropy(outputs[output_name], targets[target_name])

    regression = (
        ("box", "ground_box"),
        ("heatmap", "ground_heatmap"),
        ("control_points", "control_points"),
        ("next_state", "next_state_prediction"),
    )
    for target_name, output_name in regression:
        if target_name in targets:
            error = outputs[output_name] - T(targets[target_name])
            losses[target_name] = (error * error).mean()

    if "verifier" in targets and "verifier_logits" in outputs:
        losses["verifier"] = binary_cross_entropy_with_logits(
            outputs["verifier_logits"], targets["verifier"]
        )

    total = T(0.0)
    for name, value in losses.items():
        total = total + weights.get(name, 1.0) * value
    return total, losses
