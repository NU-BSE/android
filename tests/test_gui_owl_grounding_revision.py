from __future__ import annotations

from dataclasses import dataclass

import pytest

from android_drlqn.gui_owl_metrics import parse_tool_call
from scripts.gui_owl_grounding_revision import balance_gui_action_records
from scripts.gui_owl_grounding_structural import (
    select_salient_layer_ids,
    text_layer_gap_limit,
    text_layer_selection_is_safe,
)


@dataclass(frozen=True)
class _Record:
    mode: str
    messages: list[object]
    loss_weight: float = 1.0


def _gui_record(action: str, weight: float = 1.0) -> _Record:
    text = (
        'Action: test\n<tool_call>{"name":"mobile.'
        + action
        + '","arguments":{}}</tool_call>'
    )
    return _Record(
        mode="gui_policy",
        messages=[{"role": "assistant", "content": text}],
        loss_weight=weight,
    )


def test_historical_depth24_gap_is_rejected():
    source = tuple(range(28))
    historical = (
        0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 15,
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
    )
    assert text_layer_gap_limit(source, 24) == 3
    assert not text_layer_selection_is_safe(source, historical)


def test_historical_depth20_gap_is_rejected():
    source = (
        0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 15,
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
    )
    historical = (
        0, 1, 2, 3, 4, 5, 9, 10, 16, 17,
        18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
    )
    assert not text_layer_selection_is_safe(source, historical)


def test_selection_preserves_prefix_and_coverage_at_24_and_20():
    scores = [1.0] * 28
    for index in (11, 12, 13, 14, 15, 23, 24, 25, 26, 27):
        scores[index] = 1000.0 + index

    depth24 = select_salient_layer_ids(scores, tuple(range(28)), 24, protect_tail=6)
    assert depth24[:3] == (0, 1, 2)
    assert text_layer_selection_is_safe(tuple(range(28)), depth24)
    assert max(b - a for a, b in zip(depth24, depth24[1:])) <= 3

    depth20 = select_salient_layer_ids(scores, depth24, 20, protect_tail=6)
    assert depth20[:3] == (0, 1, 2)
    assert text_layer_selection_is_safe(depth24, depth20)


def test_depth6_gap_expands_only_to_anchor_feasible_minimum():
    scores = [float(index + 1) for index in range(28)]
    depth12 = select_salient_layer_ids(scores, tuple(range(28)), 12, protect_tail=4)
    depth9 = select_salient_layer_ids(scores, depth12, 9, protect_tail=3)
    depth6 = select_salient_layer_ids(scores, depth9, 6, protect_tail=2)
    assert depth6[:3] == (0, 1, 2)
    assert depth6[-2:] == (26, 27)
    assert text_layer_gap_limit(depth9, 6) >= 12
    assert text_layer_selection_is_safe(depth9, depth6)


def test_missing_visual_integration_prefix_is_fail_closed():
    source = (0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21)
    selected = source[:19] + source[-1:]
    assert not text_layer_selection_is_safe(source, selected)


def test_gui_action_balancing_damps_click_majority_without_changing_total_mass():
    records = [
        _gui_record("click"),
        _gui_record("click"),
        _gui_record("click"),
        _gui_record("click"),
        _gui_record("swipe"),
        _gui_record("type"),
    ]
    balanced = balance_gui_action_records(records)
    assert sum(record.loss_weight for record in balanced) == pytest.approx(6.0)
    assert balanced[0].loss_weight < 1.0
    assert balanced[4].loss_weight > 1.0
    assert balanced[5].loss_weight > 1.0


def test_metrics_reject_unknown_mobile_namespace_fragment_but_accept_dataset_actions():
    unknown = 'Action: test\n<tool_call>{"name":"mobile","arguments":{}}</tool_call>'
    long_press = 'Action: test\n<tool_call>{"name":"mobile.long_press","arguments":{}}</tool_call>'
    type_call = 'Action: test\n<tool_call>{"name":"mobile.type","arguments":{}}</tool_call>'
    assert parse_tool_call(unknown) is None
    assert parse_tool_call(long_press).name == "long_press"
    assert parse_tool_call(type_call).name == "type"
