"""Benchmark SubmissionDRLQNSolver on the checked-in official ARCEngine games.

This runner does not require the competition gateway or ``arc_agi`` package. It
loads each provided environment implementation directly, exposes the same
``step(GameAction, data)`` boundary through ``ArcadeEnvAdapter``, and writes a
machine-readable summary. It is useful for smoke budgets and for evaluating a
Duck-distilled checkpoint before packaging a competition notebook.
"""
from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

from drlqn.config import drlqn_xs
from drlqn.env.arcade_adapter import ArcadeEnvAdapter
from drlqn.train.checkpoint import load_solver_checkpoint
from drlqn.train.submission_solver import SubmissionDRLQNSolver


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("arc-prize-2026-arc-agi-3/environment_files"),
    )
    parser.add_argument("--game", action="append", default=[])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--history-len", type=int, default=4)
    parser.add_argument("--max-actions-per-level", type=int, default=8)
    parser.add_argument("--max-levels", type=int, default=128)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/official_benchmark.json"),
    )
    return parser.parse_args()


def _load_arcengine():
    try:
        import arcengine
    except ImportError as exc:
        raise SystemExit(
            "Install arcengine before running the local official benchmark"
        ) from exc
    return arcengine


def _game_files(root: Path, selected: set[str]) -> list[Path]:
    files = sorted(
        path
        for path in root.glob("*/*/*.py")
        if not path.name.startswith("_")
    )
    if selected:
        files = [path for path in files if path.stem in selected]
    return files


def _load_game(path: Path, base_game_cls):
    module_name = f"official_benchmark_{path.parent.parent.name}_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load environment module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    candidates = [
        value
        for value in vars(module).values()
        if inspect.isclass(value)
        and value.__module__ == module.__name__
        and issubclass(value, base_game_cls)
        and value is not base_game_cls
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"{path}: expected one ARCEngine game subclass, found {len(candidates)}"
        )
    return candidates[0]()


class DirectArcadeEnvironment:
    """Local implementation of the Arcade environment step contract."""

    def __init__(self, game: Any, action_input_cls: Any):
        self.game = game
        self.ActionInput = action_input_cls
        self._last_response = None

    def step(self, action, data):
        response = self.game.perform_action(
            self.ActionInput(id=action, data=dict(data)),
            raw=True,
        )
        self._last_response = response
        return response


def _solve(path: Path, arcengine, args: argparse.Namespace, index: int) -> dict:
    game = _load_game(path, arcengine.ARCBaseGame)
    raw_env = DirectArcadeEnvironment(game, arcengine.ActionInput)
    env = ArcadeEnvAdapter(raw_env, path.stem, arcengine.GameAction)
    solver = SubmissionDRLQNSolver(
        drlqn_xs(),
        seed=args.seed + index,
        history_len=args.history_len,
    )
    checkpoint_metadata = {}
    if args.checkpoint is not None:
        checkpoint_metadata = load_solver_checkpoint(solver, args.checkpoint)
    budgets = {
        level: int(args.max_actions_per_level)
        for level in range(1, int(args.max_levels) + 1)
    }
    result = solver.run_env(env, budgets)
    return {
        "env_id": result.env_id,
        "completed": int(result.completed),
        "won": bool(result.won),
        "actions_per_level": result.actions_per_level,
        "total_actions": int(result.total_actions),
        "updates": int(result.updates),
        "wall_clock_s": float(result.wall_clock_s),
        "q_margin_mean": float(result.q_margin_mean),
        "delta_support_mean": float(result.delta_support_mean),
        "telemetry": result.telemetry,
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "checkpoint_metadata": checkpoint_metadata,
    }


def main() -> int:
    args = _parse_args()
    if args.max_actions_per_level <= 0 or args.max_levels <= 0:
        raise SystemExit("action and level limits must be positive")
    if args.checkpoint is not None and not args.checkpoint.exists():
        raise SystemExit(f"checkpoint does not exist: {args.checkpoint}")

    arcengine = _load_arcengine()
    files = _game_files(args.root, set(args.game))
    if not files:
        raise SystemExit(f"no environment files found under {args.root}")

    rows = []
    errors = []
    for index, path in enumerate(files):
        try:
            row = _solve(path, arcengine, args, index)
        except Exception as exc:  # retain all game failures in one benchmark run
            row = {
                "env_id": path.stem,
                "error": f"{type(exc).__name__}: {exc}",
            }
            errors.append(row)
        rows.append(row)
        print(json.dumps(row, sort_keys=True))

    aggregate = {
        "games": len(rows),
        "errors": len(errors),
        "wins": sum(int(row.get("won", False)) for row in rows),
        "levels_completed": sum(int(row.get("completed", 0)) for row in rows),
        "total_actions": sum(int(row.get("total_actions", 0)) for row in rows),
        "max_actions_per_level": int(args.max_actions_per_level),
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
    }
    payload = {"aggregate": aggregate, "games": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"aggregate": aggregate}, sort_keys=True))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
