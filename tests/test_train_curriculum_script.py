import importlib.util
import json
from pathlib import Path

import numpy as np


def _load_train_curriculum_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "train_curriculum.py"
    spec = importlib.util.spec_from_file_location("train_curriculum_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _DummyTrainer:
    def train_batch(self, batch):
        return {"loss": float(len(batch))}


def test_stage_emits_batch_progress_for_long_epochs(capsys):
    module = _load_train_curriculum_module()
    rng = np.random.default_rng(0)

    history = module._stage(
        "demo",
        epochs=1,
        items=[object(), object(), object(), object()],
        trainer=_DummyTrainer(),
        batch_size=2,
        rng=rng,
        heartbeat_batches=2,
    )

    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    records = [json.loads(line) for line in lines]
    assert any(record.get("event") == "batch_progress" for record in records)
    assert records[-1]["stage"] == "demo"
    assert history[0]["train"]["loss"] == 2.0
