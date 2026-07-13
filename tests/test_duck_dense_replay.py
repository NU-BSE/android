from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from drlqn.config import drlqn_xs
from drlqn.train.dense_sft import (
    DenseDuckSFTTrainer,
    prepare_dense_examples,
    split_by_game_and_seed,
)
from drlqn.train.duck_dataset import (
    DuckExample,
    load_examples_npz,
    save_examples_npz,
)
from drlqn.train.duck_replay import parse_recording_events, replay_recording
from drlqn.train.submission_solver import SubmissionDRLQNSolver


class _ReplayEnv:
    def __init__(self, seed: int):
        self.seed = seed
        self.env_id = "aa00"
        self.level = 1
        self.state = "NOT_FINISHED"
        self._grid = np.zeros((64, 64), dtype=np.uint8)
        if seed != 3:
            self._grid[63, 63] = seed + 1
        self._step = 0

    def reset(self):
        return SimpleNamespace(grid=self._grid.copy())

    def step(self, action: int, x=None, y=None):
        assert action in (2, 4)
        self._step += 1
        self._grid[0, self._step - 1] = action
        level_won = self._step == 2
        if level_won:
            self.level = 2
            self.state = "WIN"
        return SimpleNamespace(
            frame=SimpleNamespace(grid=self._grid.copy()),
            level_won=level_won,
            env_won=level_won,
        )


def test_batched_duck_actions_replay_to_dense_transitions():
    initial = np.zeros((64, 64), dtype=np.uint8)
    final = initial.copy()
    final[0, 0] = 4
    final[0, 1] = 2
    events = [
        {"role": "user", "grid": initial.tolist()},
        {
            "role": "assistant",
            "reasoning": "two actions in one Python call",
            "tool_calls": [
                {
                    "function": {
                        "name": "python",
                        "arguments": json.dumps({"code": "action(['RIGHT', 'DOWN'])"}),
                    }
                }
            ],
        },
        {"role": "user", "grid": final.tolist()},
        {"data": {"won": 1, "total_actions": 2}},
    ]
    recording = parse_recording_events(
        events,
        {"id": "recording-1", "game_code": "aa00", "game_id": "aa00-instance"},
    )
    examples, report = replay_recording(
        recording,
        lambda seed: _ReplayEnv(seed),
        candidate_seeds=(0, 1, 2, 3, 4),
        environment_name="fake-aa00",
    )

    assert report.seed == 3
    assert report.actions == 2
    assert report.verified_anchors == 2
    assert not report.mismatches
    assert report.success
    assert [example.action for example in examples] == [4, 2]
    assert [example.remaining_path_length for example in examples] == [1, 0]
    assert [example.level_advanced for example in examples] == [False, True]
    assert np.array_equal(examples[0].next_grid, examples[1].grid)


def test_dense_npz_roundtrip_and_grouped_split(tmp_path):
    base = np.zeros((64, 64), dtype=np.uint8)
    base[10:12, 10:12] = 8
    examples = []
    for game, seed in (("aa00", 0), ("aa00", 1), ("bb01", 0), ("cc02", 0)):
        next_grid = base.copy()
        next_grid[20 + seed, 20] = 9
        examples.append(
            DuckExample(
                game_id=game,
                message_index=seed,
                grid=base,
                action=7,
                next_grid=next_grid,
                visual_changed=True,
                level_advanced=False,
                remaining_path_length=2,
                seed=seed,
                success=game != "cc02",
                trajectory_id=f"{game}-{seed}",
            )
        )

    path = save_examples_npz(examples, tmp_path / "dense.npz")
    restored = load_examples_npz(path)
    assert len(restored) == len(examples)
    assert restored[0].next_grid is not None
    assert restored[0].remaining_path_length == 2
    assert restored[0].trajectory_id == "aa00-0"

    prepared = prepare_dense_examples(restored)
    split = split_by_game_and_seed(
        prepared,
        held_out_games=("cc02",),
        validation_fraction=0.5,
        seed=7,
    )
    train_groups = {(item.game_id, item.seed) for item in split.train}
    validation_groups = {(item.game_id, item.seed) for item in split.validation}
    assert train_groups.isdisjoint(validation_groups)
    assert {item.game_id for item in split.held_out} == {"cc02"}
    assert all(item.game_id != "cc02" for item in (*split.train, *split.validation))


def test_dense_sft_trains_router_and_auxiliary_heads():
    grid = np.zeros((64, 64), dtype=np.uint8)
    grid[20:24, 20:24] = 8
    next_grid = grid.copy()
    next_grid[30:32, 30:32] = 9
    examples = prepare_dense_examples(
        [
            DuckExample(
                "demo",
                1,
                grid,
                action=7,
                next_grid=next_grid,
                visual_changed=True,
                level_advanced=True,
                seed=0,
                success=True,
                trajectory_id="demo-0",
            )
        ]
    )
    assert len(examples) == 1
    solver = SubmissionDRLQNSolver(drlqn_xs(), seed=3)
    metrics = DenseDuckSFTTrainer(solver, seed=3).train_batch(examples)
    assert np.isfinite(metrics["loss"])
    assert 0.0 <= metrics["key_action_accuracy"] <= 1.0
    assert 0.0 <= metrics["level_advance_recall"] <= 1.0
    assert solver.updates == 1
