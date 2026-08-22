from __future__ import annotations

import math

import pytest

from android_drlqn.gui_owl_metrics import (
    assistant_content_text,
    compare_gui_outputs,
    parse_tool_call,
)


def _call(name: str, arguments: str) -> str:
    return f'Action: test\n<tool_call>{{"name":"{name}","arguments":{arguments}}}</tool_call>'


def test_action_name_match_keeps_distinct_gui_actions():
    target = _call("mobile.swipe", '{"coordinate":[100,200],"coordinate2":[100,800]}')
    generated = _call("mobile.click", '{"coordinate":[100,200]}')
    result = compare_gui_outputs(generated, target)
    assert result.target_action == "swipe"
    assert result.generated_action == "click"
    assert result.action_match is False


def test_coordinate_error_is_euclidean_in_normalized_gui_space():
    target = _call("mobile.click", '{"coordinate":[100,200]}')
    generated = _call("mobile.click", '{"coordinate":[103,204]}')
    result = compare_gui_outputs(generated, target)
    assert result.action_match is True
    assert result.coordinate_error == pytest.approx(5.0)


def test_missing_generated_coordinate_is_penalized_not_dropped():
    target = _call("mobile.click", '{"coordinate":[100,200]}')
    generated = _call("mobile.click", '{}')
    result = compare_gui_outputs(generated, target)
    assert result.action_match is True
    assert result.target_coordinate == (100.0, 200.0)
    assert result.generated_coordinate is None
    assert result.coordinate_error == pytest.approx(math.sqrt(2.0) * 1000.0)


def test_action_aliases_match_dataset_and_runtime_names():
    target = _call("mobile.tap", '{"coordinate":[100,200]}')
    generated = _call("mobile.click", '{"coordinate":[100,200]}')
    result = compare_gui_outputs(generated, target)
    assert result.target_action == "click"
    assert result.generated_action == "click"
    assert result.action_match is True


def test_assistant_content_text_handles_qwen_content_blocks():
    content = [
        {"type": "text", "text": "Action: tap"},
        {"type": "text", "text": '<tool_call>{"name":"mobile.click","arguments":{}}</tool_call>'},
    ]
    text = assistant_content_text(content)
    assert text.startswith("Action: tap")
    assert parse_tool_call(text) is not None
