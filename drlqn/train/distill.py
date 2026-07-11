"""Utilities for distilling tool-agent demonstrations into DRLQN actions."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..operators.library import N_KEYS, N_OFFSETS, click_lambda, to_env_action


@dataclass(frozen=True)
class TeacherActionMatch:
    lambda_id: int
    distance: float
    exact_action: bool


def match_teacher_action(action: int, anchors, x: int | None = None,
                         y: int | None = None) -> TeacherActionMatch:
    """Map a teacher primitive/coordinate action to the closest lambda action.

    Key actions map exactly. ACTION6 coordinates are matched against all five
    offsets of every active spatial anchor. The returned distance can be used as
    a sample weight or to reject demonstrations outside proposal coverage.
    """
    if 1 <= int(action) <= N_KEYS:
        return TeacherActionMatch(int(action) - 1, 0.0, True)
    if int(action) != 6 or x is None or y is None:
        raise ValueError("teacher action must be ACTION1..ACTION6; clicks need x and y")
    if not anchors:
        raise ValueError("cannot map a click without spatial anchors")

    best_lam = -1
    best_distance = float("inf")
    for slot in range(min(24, len(anchors))):
        for offset in range(N_OFFSETS):
            lam = click_lambda(slot, offset)
            _, px, py = to_env_action(lam, anchors)
            distance = float(np.hypot(float(px - x), float(py - y)))
            if distance < best_distance:
                best_lam = lam
                best_distance = distance
    return TeacherActionMatch(best_lam, best_distance, best_distance < 0.5)


def behavior_cloning_targets(matches: list[TeacherActionMatch],
                             max_distance: float = 12.0) -> tuple[np.ndarray, np.ndarray]:
    """Return lambda labels and smooth coordinate-coverage sample weights."""
    labels = np.array([match.lambda_id for match in matches], dtype=np.int64)
    distances = np.array([match.distance for match in matches], dtype=np.float64)
    weights = np.clip(1.0 - distances / max(max_distance, 1e-6), 0.05, 1.0)
    return labels, weights
