"""Distill dense Duck trajectories into a validated DRLQN checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from drlqn.config import drlqn_xs
from drlqn.train.checkpoint import load_solver_checkpoint, save_model_checkpoint
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
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-click-distance", type=float, default=12.0)
    parser.add_argument("--drop-uncovered-clicks", action="store_true")
    parser.add_argument("--held-out-game", action="append", default=[])
    parser.add_argument("--held-out-game-count", type=int, default=1)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--success-weight", type=float, default=2.5)
    parser.add_argument("--failure-weight", type=float, default=0.6)
    parser.add_argument("--informative-failure-weight", type=float, default=1.5)
    parser.add_argument("--frame-change-coef", type=float, default=0.3)
    parser.add_argument("--level-advance-coef", type=float, default=0.5)
    parser.add_argument("--next-embedding-coef", type=float, default=0.2)
    parser.add_argument("--input-checkpoint", type=Path)
    parser.add_argument(
        "--output-checkpoint",
        type=Path,
        default=Path("checkpoints/duck_sft_drlqn_xs.npz"),
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("results/duck_sft_metrics.json"),
    )
    return parser.parse_args()


def _batches(
    items: tuple[PreparedDenseSFT, ...],
    batch_size: int,
    rng: np.random.Generator,
):
    order = rng.permutation(len(items))
    for start in range(0, len(order), batch_size):
        yield [items[int(index)] for index in order[start : start + batch_size]]


def _mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted(set.intersection(*(set(row) for row in rows)))
    return {
        key: float(np.mean([row[key] for row in rows]))
        for key in keys
        if isinstance(rows[0][key], (int, float))
    }


def main() -> None:
    args = _parse_args()
    if args.epochs <= 0 or args.batch_size <= 0:
        raise SystemExit("--epochs and --batch-size must be positive")
    if not 0.0 <= args.validation_fraction < 1.0:
        raise SystemExit("--validation-fraction must be in [0, 1)")

    examples = load_duck_traces(args.traces)
    prepared = prepare_dense_examples(
        examples,
        max_click_distance=float(args.max_click_distance),
        drop_uncovered_clicks=bool(args.drop_uncovered_clicks),
        success_weight=float(args.success_weight),
        failure_weight=float(args.failure_weight),
        informative_failure_weight=float(args.informative_failure_weight),
    )
    if not prepared:
        raise SystemExit("No usable Duck state/action pairs found")
    split = split_by_game_and_seed(
        prepared,
        held_out_games=args.held_out_game,
        held_out_game_count=args.held_out_game_count,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    if not split.train:
        raise SystemExit("Grouped split produced no training groups")

    solver = SubmissionDRLQNSolver(drlqn_xs(), seed=args.seed)
    if args.input_checkpoint:
        load_solver_checkpoint(solver, args.input_checkpoint)
    trainer = DenseDuckSFTTrainer(
        solver,
        seed=args.seed,
        frame_change_coef=args.frame_change_coef,
        level_advance_coef=args.level_advance_coef,
        next_embedding_coef=args.next_embedding_coef,
        click_coverage_distance=args.max_click_distance,
    )
    rng = np.random.default_rng(args.seed)

    history: list[dict] = []
    for epoch in range(1, args.epochs + 1):
        batch_metrics = [
            trainer.train_batch(batch)
            for batch in _batches(split.train, args.batch_size, rng)
        ]
        row = {
            "epoch": epoch,
            "train": _mean_metrics(batch_metrics),
            "validation": trainer.evaluate(split.validation),
            "held_out_games": trainer.evaluate(split.held_out),
            "updates": solver.updates,
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True))

    games = sorted({item.game_id for item in prepared})
    metadata = {
        "config": solver.cfg.name,
        "source": "duck-harness-dense-replay-sft",
        "raw_examples": len(examples),
        "prepared_examples": len(prepared),
        "dense_next_state_examples": sum(int(item.has_next) for item in prepared),
        "successful_examples": sum(int(item.success == 1) for item in prepared),
        "informative_failure_examples": sum(
            int(item.success == 0 and (item.visual_changed == 1 or item.level_advanced == 1))
            for item in prepared
        ),
        "games": games,
        "held_out_games": list(split.held_out_games),
        "train_groups": [list(group) for group in split.train_groups],
        "validation_groups": [list(group) for group in split.validation_groups],
        "split_counts": {
            "train": len(split.train),
            "validation": len(split.validation),
            "held_out": len(split.held_out),
        },
        "max_click_distance": float(args.max_click_distance),
        "epochs": int(args.epochs),
        "updates": int(solver.updates),
        "final_metrics": history[-1],
        "auxiliary_heads": {
            "frame_change": float(args.frame_change_coef),
            "level_advance": float(args.level_advance_coef),
            "next_state_embedding": float(args.next_embedding_coef),
            "persisted_in_checkpoint": False,
        },
    }
    save_model_checkpoint(solver.model, args.output_checkpoint, metadata)
    # Validate the exact artifact that downstream benchmarks will load.
    validator = SubmissionDRLQNSolver(drlqn_xs(), seed=args.seed + 1)
    validated_metadata = load_solver_checkpoint(validator, args.output_checkpoint)
    if validated_metadata.get("source") != metadata["source"]:
        raise RuntimeError("checkpoint metadata failed round-trip validation")

    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(
        json.dumps({"metadata": validated_metadata, "history": history}, indent=2),
        encoding="utf-8",
    )
    print(f"checkpoint: {args.output_checkpoint}")
    print(f"metrics: {args.metrics}")


if __name__ == "__main__":
    main()
