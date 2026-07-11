"""Distill Duck Harness trace exports into a submission DRLQN checkpoint.

Duck exports traces with ``inference-traces``. This script consumes either JSON
files or directories containing those exports, keeps conservative state/action
pairs with an observed 64x64 grid, and behavior-clones the DRLQN encoder/router.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from drlqn.config import drlqn_xs
from drlqn.train.checkpoint import load_solver_checkpoint, save_model_checkpoint
from drlqn.train.duck_dataset import load_duck_traces
from drlqn.train.sft import DuckSFTTrainer, PreparedSFT, prepare_examples
from drlqn.train.submission_solver import SubmissionDRLQNSolver


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-click-distance", type=float, default=12.0)
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


def _batches(items: list[PreparedSFT], batch_size: int, rng: np.random.Generator):
    order = rng.permutation(len(items))
    for start in range(0, len(order), batch_size):
        yield [items[int(index)] for index in order[start : start + batch_size]]


def main() -> None:
    args = _parse_args()
    if args.epochs <= 0 or args.batch_size <= 0:
        raise SystemExit("--epochs and --batch-size must be positive")

    examples = load_duck_traces(args.traces)
    prepared = prepare_examples(
        examples,
        max_click_distance=float(args.max_click_distance),
    )
    if not prepared:
        raise SystemExit(
            "No usable state/action pairs found. Export Duck traces with grids "
            "and Python action tool calls before running SFT."
        )

    solver = SubmissionDRLQNSolver(drlqn_xs(), seed=args.seed)
    if args.input_checkpoint:
        load_solver_checkpoint(solver, args.input_checkpoint)
    trainer = DuckSFTTrainer(solver)
    rng = np.random.default_rng(args.seed)

    history: list[dict] = []
    for epoch in range(1, args.epochs + 1):
        batch_metrics = [
            trainer.train_batch(batch)
            for batch in _batches(prepared, args.batch_size, rng)
        ]
        row = {
            "epoch": epoch,
            "loss": float(np.mean([item["loss"] for item in batch_metrics])),
            "accuracy": float(np.mean([item["accuracy"] for item in batch_metrics])),
            "mean_weight": float(np.mean([item["mean_weight"] for item in batch_metrics])),
            "updates": solver.updates,
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True))

    games = sorted({item.game_id for item in prepared})
    metadata = {
        "format": "drlqn-numpy-state-dict-v1",
        "config": solver.cfg.name,
        "source": "duck-harness-trace-sft",
        "raw_examples": len(examples),
        "prepared_examples": len(prepared),
        "games": games,
        "max_click_distance": float(args.max_click_distance),
        "epochs": int(args.epochs),
        "updates": int(solver.updates),
        "final_metrics": history[-1],
    }
    save_model_checkpoint(solver.model, args.output_checkpoint, metadata)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(
        json.dumps({"metadata": metadata, "history": history}, indent=2),
        encoding="utf-8",
    )
    print(f"checkpoint: {args.output_checkpoint}")
    print(f"metrics: {args.metrics}")


if __name__ == "__main__":
    main()
