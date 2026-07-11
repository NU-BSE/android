"""Run SubmissionDRLQNSolver through the ARC Prize ``arc_agi.Arcade`` API.

The same entry point works against the local 25-game environment directory and
inside a competition rerun connected to the gateway. Competition wheels must be
installed by the notebook before importing this script.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from drlqn.config import drlqn_xs
from drlqn.env.arcade_adapter import ArcadeEnvAdapter
from drlqn.train.checkpoint import load_solver_checkpoint
from drlqn.train.submission_solver import SubmissionDRLQNSolver


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", action="append", default=[])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-actions-per-level", type=int, default=600)
    parser.add_argument("--history-len", type=int, default=4)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/arcade_results.json"))
    return parser.parse_args()


def solve_game(raw_env, game_id: str, game_action_cls, *, seed: int,
               max_actions_per_level: int, history_len: int,
               checkpoint: Path | None = None) -> dict:
    env = ArcadeEnvAdapter(raw_env, game_id, game_action_cls)
    solver = SubmissionDRLQNSolver(
        drlqn_xs(),
        seed=seed,
        history_len=history_len,
    )
    checkpoint_metadata = {}
    if checkpoint is not None:
        checkpoint_metadata = load_solver_checkpoint(solver, checkpoint)
    budgets = {level: int(max_actions_per_level) for level in range(1, 129)}
    result = solver.run_env(env, budgets)
    return {
        "env_id": result.env_id,
        "completed": result.completed,
        "won": result.won,
        "actions_per_level": result.actions_per_level,
        "total_actions": result.total_actions,
        "updates": result.updates,
        "wall_clock_s": result.wall_clock_s,
        "telemetry": result.telemetry,
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
        "checkpoint_metadata": checkpoint_metadata,
    }


def main() -> None:
    args = _parse_args()
    try:
        import arc_agi
        from arcengine import GameAction
    except ImportError as exc:  # pragma: no cover - competition dependency
        raise SystemExit(
            "Install the ARC-AGI-3 competition wheels before running this entry point"
        ) from exc

    if args.checkpoint is not None and not args.checkpoint.exists():
        raise SystemExit(f"checkpoint does not exist: {args.checkpoint}")

    arcade = arc_agi.Arcade()
    selected = set(args.game)
    environments = [
        info
        for info in arcade.available_environments
        if not selected or info.game_id in selected
    ]
    rows = []
    for index, info in enumerate(environments):
        row = solve_game(
            arcade.make(info.game_id),
            info.game_id,
            GameAction,
            seed=args.seed + index,
            max_actions_per_level=args.max_actions_per_level,
            history_len=args.history_len,
            checkpoint=args.checkpoint,
        )
        rows.append(row)
        print(json.dumps(row, sort_keys=True))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    if hasattr(arcade, "get_scorecard"):
        scorecard = arcade.get_scorecard()
        print(
            json.dumps(
                {
                    "score": float(scorecard.score),
                    "environments_completed": int(scorecard.total_environments_completed),
                    "total_environments": int(scorecard.total_environments),
                    "levels_completed": int(scorecard.total_levels_completed),
                    "total_levels": int(scorecard.total_levels),
                    "total_actions": int(scorecard.total_actions),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
