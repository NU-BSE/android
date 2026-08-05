from __future__ import annotations

import json

import pytest

from android_drlqn.qwen_screen_qa import (
    DEFAULT_MODEL,
    QwenScreenQAClient,
    ScreenContext,
    chat_sft_records,
)


def context():
    return ScreenContext(
        screen_id="private-dns",
        package="com.android.settings",
        screen_name="Private DNS",
        goal="Find the Private DNS screen",
        visible_labels=("Network & internet", "Private DNS"),
        ui_path=("Settings", "Network & internet", "Private DNS"),
        os_family="android-aosp",
    )


def valid_response():
    return {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "items": [
                                {
                                    "question": "Where can I find Private DNS?",
                                    "answer": (
                                        "Open Settings, choose Network & internet, "
                                        "then select Private DNS."
                                    ),
                                    "intent": "find",
                                },
                                {
                                    "question": "What is the Private DNS screen for?",
                                    "answer": (
                                        "It lets you review or choose Android's "
                                        "Private DNS mode and provider."
                                    ),
                                    "intent": "understand",
                                },
                            ]
                        }
                    )
                }
            }
        ]
    }


def test_client_uses_qwen37_json_mode_without_thinking(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "secret-value")
    captured = {}

    def transport(endpoint, headers, payload, timeout_s):
        captured.update(
            endpoint=endpoint,
            headers=dict(headers),
            payload=dict(payload),
            timeout_s=timeout_s,
        )
        return valid_response()

    client = QwenScreenQAClient(transport=transport, max_retries=0)
    items = client.generate(context(), count=2)
    assert client.model == DEFAULT_MODEL == "qwen3.7-plus"
    assert captured["endpoint"].endswith("/chat/completions")
    assert captured["headers"]["Authorization"] == "Bearer secret-value"
    assert captured["payload"]["response_format"] == {"type": "json_object"}
    assert captured["payload"]["enable_thinking"] is False
    assert "max_tokens" not in captured["payload"]
    assert captured["payload"]["temperature"] == 0.2
    prompt = captured["payload"]["messages"][1]["content"]
    assert "Private DNS" in prompt and "Settings > Network & internet" in prompt
    assert "non-empty JSON string" in prompt
    assert len(items) == 2
    records = list(chat_sft_records(context(), items))
    assert records[0]["messages"][-1]["role"] == "assistant"
    assert "secret-value" not in json.dumps(records)


def test_client_rejects_non_alibaba_endpoint():
    with pytest.raises(ValueError, match="Alibaba Cloud"):
        QwenScreenQAClient(base_url="https://example.com/v1")


def test_generated_executable_guidance_is_rejected(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "secret")
    raw = valid_response()
    payload = json.loads(raw["choices"][0]["message"]["content"])
    payload["items"][0]["answer"] = (
        "Run adb shell settings put secure private_dns_mode hostname"
    )
    raw["choices"][0]["message"]["content"] = json.dumps(payload)
    client = QwenScreenQAClient(transport=lambda *args: raw, max_retries=0)
    with pytest.raises(RuntimeError, match="executable"):
        client.generate(context(), count=2)


def test_wrong_item_count_is_rejected(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "secret")
    raw = valid_response()
    payload = json.loads(raw["choices"][0]["message"]["content"])
    payload["items"] = payload["items"][:1]
    raw["choices"][0]["message"]["content"] = json.dumps(payload)
    client = QwenScreenQAClient(transport=lambda *args: raw, max_retries=0)
    with pytest.raises(RuntimeError, match="expected 2, got 1"):
        client.generate(context(), count=2)


def test_schema_failure_retries_with_corrective_prompt(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "secret")
    invalid = valid_response()
    value = json.loads(invalid["choices"][0]["message"]["content"])
    value["items"][0]["intent"] = "navigate"
    invalid["choices"][0]["message"]["content"] = json.dumps(value)
    calls = []

    def transport(endpoint, headers, payload, timeout_s):
        calls.append(dict(payload))
        return invalid if len(calls) == 1 else valid_response()

    monkeypatch.setattr("android_drlqn.qwen_screen_qa.time.sleep", lambda _: None)
    client = QwenScreenQAClient(transport=transport, max_retries=1)
    items = client.generate(context(), count=2)

    assert len(items) == 2
    assert len(calls) == 2
    assert len(calls[0]["messages"]) == 2
    assert len(calls[1]["messages"]) == 3
    correction = calls[1]["messages"][-1]["content"]
    assert "intent 'navigate' is unsupported" in correction
    assert "find, open, understand, confirm" in correction


def test_empty_item_error_identifies_field(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "secret")
    raw = valid_response()
    value = json.loads(raw["choices"][0]["message"]["content"])
    value["items"][1]["answer"] = ""
    raw["choices"][0]["message"]["content"] = json.dumps(value)
    client = QwenScreenQAClient(transport=lambda *args: raw, max_retries=0)
    with pytest.raises(RuntimeError, match="item 1 answer is empty"):
        client.generate(context(), count=2)
