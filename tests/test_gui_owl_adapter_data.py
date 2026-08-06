from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from scripts.train_gui_owl_adapter import (
    _default_device,
    iter_messages,
    load_adapter_config,
)


class _Availability:
    def __init__(self, available: bool):
        self.available = available

    def is_available(self) -> bool:
        return self.available


def _fake_torch(*, cuda: bool, mps: bool):
    return SimpleNamespace(
        cuda=_Availability(cuda),
        backends=SimpleNamespace(mps=_Availability(mps)),
    )


def test_default_device_prefers_cuda_then_mps_then_cpu():
    assert _default_device(_fake_torch(cuda=True, mps=True)) == "cuda"
    assert _default_device(_fake_torch(cuda=False, mps=True)) == "mps"
    assert _default_device(_fake_torch(cuda=False, mps=False)) == "cpu"


def test_load_adapter_config_selects_named_adapter(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "base_model": "mPLUG/GUI-Owl-1.5-2B-Instruct",
                "adapters": {
                    "chat": {
                        "prepared_jsonl": "chat.jsonl",
                        "lora": {"rank": 16, "target_modules": ["q_proj"]},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    model, adapter = load_adapter_config(path, "chat")
    assert model == "mPLUG/GUI-Owl-1.5-2B-Instruct"
    assert adapter["prepared_jsonl"] == "chat.jsonl"
    with pytest.raises(ValueError, match="unknown adapter"):
        load_adapter_config(path, "gui_policy")


def test_iter_messages_requires_final_assistant_completion(tmp_path):
    valid = tmp_path / "valid.jsonl"
    valid.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "Where is Private DNS?"},
                    {"role": "assistant", "content": "Open Network & internet."},
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )
    messages = list(iter_messages(valid))
    assert messages[0][-1]["role"] == "assistant"

    invalid = tmp_path / "invalid.jsonl"
    invalid.write_text(
        json.dumps({"messages": [{"role": "user", "content": "hello"}]}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="assistant completion"):
        list(iter_messages(invalid))
