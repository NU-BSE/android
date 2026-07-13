"""Run Duck SFT -> environment model -> planner/value distillation curriculum."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from drlqn.config import drlqn_xs
from drlqn.train.checkpoint import load_solver_checkpoint, save_model_checkpoint
from drlqn.train.curriculum import EnvironmentTransitionTrainer, PlannerValueDistiller
from drlqn.train.dense_sft import (
    DenseDuckSFTTrainer,
    PreparedDenseSFT,
    prepare_dense_examples,
    split_by_game_and_seed,
)
from drlqn.train.duck_dataset import load_duck_traces
from drlqn.train.submission_solver import SubmissionDRLQNSolver


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duck", nargs="+", type=Path, required=True)
    parser.add_argument("--generated", nargs="+", type=Path, required=True)
    parser.add_argument("--duck-epochs", type=int, default=10)
    parser.add_argument("--world-epochs", type=int, default=8)
    parser.add_argument("--planner-epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--held-out-game", action="append", default=[])
    parser.add_argument("--held-out-game-count", type=int, default=2)
    parser.add_argument("--input-checkpoint", type=Path)
    parser.add_argument(
        "--output-checkpoint",
        type=Path,
        default=Path("checkpoints/duck_env_planner_drlqn_xs.npz"),
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("results/curriculum_metrics.json"),
    )
    return parser.parse_args()


def _batches(items: Sequence[PreparedDenseSFT], size: int, rng: np.random.Generator):
    order = rng.permutation(len(items))
    for start in range(0, len(order), size):
        yield [items[int(index)] for index in order[start : start + size]]


def _stage(
    name: str,
    epochs: int,
    items: Sequence[PreparedDenseSFT],
    trainer: object,
    batch_size: int,
    rng: np.random.Generator,
    evaluate: Callable[[Sequence[PreparedDenseSFT]], dict] | None = None,
) -> list[dict]:
    history = []
    for epoch in range(1, epochs + 1):
        rows = [trainer.train_batch(batch) for batch in _batches(items, batch_size, rng)]
        numeric = sorted(set.intersection(*(set(row) for row in rows))) if rows else []
        result = {
            "stage": name,
            "epoch": epoch,
            "train": {
                key: float(np.mean([row[key] for row in rows]))
                for key in numeric
            },
        }
        if evaluate is not None:
            result["evaluation"] = evaluate(items)
        history.append(result)
        print(json.dumps(result, sort_keys=True))
    return history


def main() -> None:
    args = _parse_args()
    if min(args.duck_epochs, args.world_epochs, args.planner_epochs, args.batch_size) <= 0:
        raise SystemExit("epochs and batch size must be positive")
    duck_examples = load_duck_traces(args.duck)
    generated_examples = load_duck_traces(args.generated)
    duck = prepare_dense_examples(duck_examples)
    generated = prepare_dense_examples(
        generated_examples,
        success_weight=1.4,
        failure_weight=1.0,
        informative_failure_weight=1.2,
    )
    if not duck or not generated:
        raise SystemExit("both Duck and generated datasets must contain usable transitions")
    combined = [*duck, *generated]
    split = split_by_game_and_seed(
        combined,
        held_out_games=args.held_out_game,
        held_out_game_count=args.held_out_game_count,
        validation_fraction=0.2,
        seed=args.seed,
    )
    train_groups = set(split.train_groups)
    duck_train = [item for item in duck if (item.game_id, item.seed) in train_groups]
    generated_train = [item for item in generated if (item.game_id, item.seed) in train_groups]
    if not duck_train or not generated_train:
        raise SystemExit("grouped split removed all examples from a curriculum stage")

    # Keep a small Duck replay stream during environment-model training to avoid
    # catastrophic forgetting, then mix Duck and planner trajectories for the
    # final action/value distillation stage. Random actions remain value/effect
    # evidence but are masked from planner imitation by PlannerValueDistiller.
    duck_replay_stride = max(1, len(duck_train) // max(1, len(generated_train) // 4))
    world_train = [*generated_train, *duck_train[::duck_replay_stride]]
    planner_train = [*generated_train, *duck_train]

    solver = SubmissionDRLQNSolver(drlqn_xs(), seed=args.seed)
    if args.input_checkpoint:
        load_solver_checkpoint(solver, args.input_checkpoint)
    rng = np.random.default_rng(args.seed)
    history = []

    duck_trainer = DenseDuckSFTTrainer(solver, seed=args.seed)
    history.extend(
        _stage(
            "duck_sft",
            args.duck_epochs,
            duck_train,
            duck_trainer,
            args.batch_size,
            rng,
        )
    )

    world_trainer = EnvironmentTransitionTrainer(
        solver,
        seed=args.seed,
        heads=duck_trainer.heads,
    )
    history.extend(
        _stage(
            "environment_world_model",
            args.world_epochs,
            world_train,
            world_trainer,
            args.batch_size,
            rng,
        )
    )

    planner_trainer = PlannerValueDistiller(solver)
    history.extend(
        _stage(
            "planner_value_distillation",
            args.planner_epochs,
            planner_train,
            planner_trainer,
            args.batch_size,
            rng,
        )
    )

    evaluation = {
        "validation": duck_trainer.evaluate(split.validation),
        "held_out_games": duck_trainer.evaluate(split.held_out),
    }
    metadata = {
        "config": solver.cfg.name,
        "source": "duck-environment-planner-curriculum",
        "curriculum": [
            "duck_sft",
            "environment_world_model_with_duck_replay",
            "planner_value_distillation_with_duck_mix",
            "online_adaptation_at_evaluation",
        ],
        "duck_examples": len(duck),
        "generated_examples": len(generated),
        "stage_examples": {
            "duck_sft": len(duck_train),
            "environment_world_model": len(world_train),
            "planner_value_distillation": len(planner_train),
        },
        "held_out_games": list(split.held_out_games),
        "updates": int(solver.updates),
        "evaluation": evaluation,
    }
    save_model_checkpoint(solver.model, args.output_checkpoint, metadata)
    validator = SubmissionDRLQNSolver(drlqn_xs(), seed=args.seed + 2)
    validated = load_solver_checkpoint(validator, args.output_checkpoint)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(
        json.dumps({"metadata": validated, "history": history}, indent=2),
        encoding="utf-8",
    )
    print(f"checkpoint: {args.output_checkpoint}")
    print(f"metrics: {args.metrics}")


if __name__ == "__main__":
    main()
