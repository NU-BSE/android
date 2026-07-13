"""Hardened competition scorecard runner with deterministic passes and traces."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
from typing import Any

import numpy as np

from drlqn.config import drlqn_xs
from drlqn.env.arcade_adapter import ArcadeEnvAdapter
from drlqn.train.ablation_solver import AblationSubmissionDRLQNSolver
from drlqn.train.checkpoint import load_solver_checkpoint


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--game", action="append", default=[])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument("--max-actions-per-level", type=int, default=600)
    parser.add_argument("--max-levels", type=int, default=128)
    parser.add_argument("--trace-dir", type=Path, default=Path("results/submission_traces"))
    parser.add_argument("--output", type=Path, default=Path("results/submission_scorecard.json"))
    return parser.parse_args()


def _digest(grid) -> str:
    return hashlib.sha1(np.asarray(grid, dtype=np.uint8).tobytes()).hexdigest()[:16]


class TraceEnvironment:
    def __init__(self, env: ArcadeEnvAdapter):
        self.env = env
        self.events: list[dict[str, Any]] = []

    def __getattr__(self, name: str):
        return getattr(self.env, name)

    def reset(self):
        frame = self.env.reset()
        self.events.append({"kind": "reset", "level": int(self.env.level), "frame": _digest(frame.grid)})
        return frame

    def reset_level(self):
        frame = self.env.reset_level()
        self.events.append({"kind": "reset_level", "level": int(self.env.level), "frame": _digest(frame.grid)})
        return frame

    def step(self, action: int, x: int | None = None, y: int | None = None):
        level_before = int(self.env.level)
        before = _digest(self.env._last_grid(self.env._response)) if self.env._response is not None else None
        result = self.env.step(action, x, y)
        self.events.append({
            "kind": "action",
            "index": int(self.env.action_count),
            "action": int(action),
            "x": x,
            "y": y,
            "level_before": level_before,
            "level_after": int(self.env.level),
            "frame_before": before,
            "frame_after": _digest(result.frame.grid),
            "level_won": bool(result.level_won),
            "env_won": bool(result.env_won),
            "state": str(self.env.state),
        })
        return result


def _pass_config(index: int) -> dict[str, bool]:
    configs = (
        {"planner_portfolio": True, "graph_planning": True, "differential_complexity": True},
        {"planner_portfolio": False, "graph_planning": True, "differential_complexity": False},
        {"planner_portfolio": True, "graph_planning": False, "differential_complexity": False},
    )
    return configs[index % len(configs)]


def _scorecard(arcade) -> dict[str, Any] | None:
    if not hasattr(arcade, "get_scorecard"):
        return None
    score = arcade.get_scorecard()
    return {
        "score": float(score.score),
        "environments_completed": int(score.total_environments_completed),
        "total_environments": int(score.total_environments),
        "levels_completed": int(score.total_levels_completed),
        "total_levels": int(score.total_levels),
        "total_actions": int(score.total_actions),
    }


def main() -> None:
    args = _parse_args()
    if not args.checkpoint.exists():
        raise SystemExit(f"checkpoint does not exist: {args.checkpoint}")
    if args.passes <= 0 or args.max_actions_per_level <= 0 or args.max_levels <= 0:
        raise SystemExit("passes, budgets and level limits must be positive")
    try:
        import arc_agi
        from arcengine import GameAction
    except ImportError as exc:
        raise SystemExit("Install competition wheels before running the scorecard") from exc

    selected = set(args.game)
    payload = {"checkpoint": str(args.checkpoint), "passes": []}
    args.trace_dir.mkdir(parents=True, exist_ok=True)
    for pass_index in range(args.passes):
        pass_seed = int(args.seed + pass_index * 10_000)
        random.seed(pass_seed)
        np.random.seed(pass_seed)
        arcade = arc_agi.Arcade()
        environments = [
            info for info in arcade.available_environments
            if not selected or info.game_id in selected
        ]
        config = _pass_config(pass_index)
        rows = []
        for game_index, info in enumerate(environments):
            game_seed = pass_seed + game_index
            random.seed(game_seed)
            np.random.seed(game_seed)
            raw = arcade.make(info.game_id)
            traced = TraceEnvironment(ArcadeEnvAdapter(raw, info.game_id, GameAction))
            solver = AblationSubmissionDRLQNSolver(
                drlqn_xs(),
                seed=game_seed,
                history_enabled=True,
                history_len=4,
                **config,
            )
            metadata = load_solver_checkpoint(solver, args.checkpoint)
            budgets = {level: int(args.max_actions_per_level) for level in range(1, args.max_levels + 1)}
            result = solver.run_env(traced, budgets)
            excess = {level: count for level, count in result.actions_per_level.items() if count > args.max_actions_per_level}
            if excess:
                raise RuntimeError(f"action budget exceeded for {info.game_id}: {excess}")
            trace_path = args.trace_dir / f"pass-{pass_index:02d}-{info.game_id}.json"
            trace_path.write_text(json.dumps({
                "game": info.game_id,
                "seed": game_seed,
                "config": config,
                "checkpoint_metadata": metadata,
                "result": {
                    "won": bool(result.won),
                    "completed": int(result.completed),
                    "actions_per_level": result.actions_per_level,
                    "total_actions": int(result.total_actions),
                    "updates": int(result.updates),
                    "telemetry": result.telemetry,
                },
                "events": traced.events,
            }, indent=2), encoding="utf-8")
            row = {
                "game": info.game_id,
                "seed": game_seed,
                "won": bool(result.won),
                "levels_completed": int(result.completed),
                "total_actions": int(result.total_actions),
                "trace": str(trace_path),
            }
            rows.append(row)
            print(json.dumps({"pass": pass_index, **row}, sort_keys=True))
        payload["passes"].append({
            "index": pass_index,
            "seed": pass_seed,
            "config": config,
            "games": rows,
            "scorecard": _scorecard(arcade),
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"scorecard: {args.output}")


if __name__ == "__main__":
    main()
