from dataclasses import dataclass

from drlqn.train.distill import behavior_cloning_targets, match_teacher_action


@dataclass
class Anchor:
    cx: float
    cy: float
    x0: int
    x1: int
    y0: int
    y1: int


def test_key_teacher_action_maps_exactly():
    match = match_teacher_action(4, [])
    assert match.lambda_id == 3
    assert match.exact_action


def test_click_teacher_action_maps_to_nearest_anchor_offset():
    anchors = [Anchor(cx=20, cy=20, x0=16, x1=23, y0=16, y1=23)]
    match = match_teacher_action(6, anchors, x=20, y=20)
    assert match.lambda_id == 5
    assert match.distance == 0.0
    labels, weights = behavior_cloning_targets([match])
    assert labels.tolist() == [5]
    assert weights.tolist() == [1.0]
