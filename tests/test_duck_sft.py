from __future__ import annotations

import json

import numpy as np

from drlqn.config import drlqn_xs
from drlqn.train.checkpoint import load_solver_checkpoint, save_model_checkpoint
from drlqn.train.duck_dataset import DuckExample, actions_from_python, load_duck_trace
from drlqn.train.sft import DuckSFTTrainer, prepare_example
from drlqn.train.submission_solver import SubmissionDRLQNSolver


def test_duck_python_action_parser_preserves_rows_cols_and_extra_keys():
    code = "action(['UP', {'action': 'MOUSE', 'row': 4, 'col': 7}, 'ACTION7'])"
    assert actions_from_python(code) == [
        (1, None, None),
        (6, 7, 4),
        (7, None, None),
    ]


def test_trace_loader_keeps_only_first_action_after_each_fresh_grid(tmp_path):
    grid = np.zeros((64, 64), dtype=np.uint8).tolist()
    payload = {
        "id": "run/gp01-demo",
        "messages": [
            {"role": "user", "content": "state", "grid": grid},
            {
                "role": "assistant",
                "reasoning": "test",
                "tool_calls": [{
                    "function": {
                        "name": "python",
                        "arguments": json.dumps({"code": "action(['LEFT', 'RIGHT'])"}),
                    }
                }],
            },
            {
                "role": "assistant",
                "tool_calls": [{
                    "function": {
                        "name": "python",
                        "arguments": json.dumps({"code": "action(['DOWN'])"}),
                    }
                }],
            },
            {"role": "user", "content": "next", "grid": grid},
            {
                "role": "assistant",
                "tool_calls": [{
                    "function": {
                        "name": "python",
                        "arguments": json.dumps({"code": "action(['ACTION7'])"}),
                    }
                }],
            },
        ],
    }
    path = tmp_path / "trace.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    examples = load_duck_trace(path)
    assert [example.action for example in examples] == [3, 7]
    assert examples[0].reasoning == "test"


def test_sft_step_and_checkpoint_roundtrip(tmp_path):
    grid = np.zeros((64, 64), dtype=np.uint8)
    grid[20:24, 20:24] = 8
    prepared = prepare_example(DuckExample("demo", 1, grid, action=7))
    assert prepared is not None

    solver = SubmissionDRLQNSolver(drlqn_xs(), seed=3)
    metrics = DuckSFTTrainer(solver).train_batch([prepared])
    assert np.isfinite(metrics["loss"])
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert solver.updates == 1

    path = tmp_path / "student.npz"
    save_model_checkpoint(solver.model, path, {"source": "unit-test"})
    restored = SubmissionDRLQNSolver(drlqn_xs(), seed=9)
    metadata = load_solver_checkpoint(restored, path)
    assert metadata["source"] == "unit-test"
    assert metadata["format"] == "drlqn-ordered-parameters-v1"
    left = solver.model.state_dict()
    right = restored.model.state_dict()
    assert len(left) == len(right)
    for actual, expected in zip(left, right):
        assert np.array_equal(actual, expected)
