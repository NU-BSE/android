"""Run cold-versus-Duck-SFT DRLQN ablations on checked-in official games.

The default matrix covers budgets 32/100/300/600, five seeds, cold and distilled
initialization, history off/on, graph planning off/on and differential
JVP/executor complexity off/on. Rows are appended to JSONL as they finish so a
long run can be resumed safely.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import product
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from drlqn.config import drlqn_xs
from drlqn.env.official_loader import environment_files, load_arcengine, make_arcade_env
from drlqn.train.ablation_solver import AblationSubmissionDRLQNSolver
from drlqn.train.checkpoint import load_solver_checkpoint
from drlqn.train.trainer import PhaseSchedule


DEFAULT_BUDGETS = (32, 100, 300, 600)
DEFAULT_SEEDS = (0, 1, 2, 3, 4)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("arc-prize-2026-arc-agi-3/environment_files"),
    )
    parser.add_argument("--game", action="append", default=[])
    parser.add_argument("--budget", action="append", type=int, default=[])
    parser.add_argument("--seed", action="append", type=int, default=[])
    parser.add_argument("--mode", action="append", choices=("cold", "duck-sft"), default=[])
    parser.add_argument("--history", action="append", choices=("off", "on"), default=[])
    parser.add_argument("--graph", action="append", choices=("off", "on"), default=[])
    parser.add_argument(
        "--differential",
        action="append",
        choices=("off", "on"),
        default=[],
        help="Toggle executor/world-model losses and the JVP approximate runtime.",
    )
    parser.add_argument("--history-len", type=int, default=4)
    parser.add_argument("--max-levels", type=int, default=128)
    parser.add_argument(
        "--duck-checkpoint",
        type=Path,
        default=Path("checkpoints/duck_sft_drlqn_xs.npz"),
    )
    parser.add_argument(
        "--rows",
        type=Path,
        default=Path("results/drlqn_baseline_matrix.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/drlqn_baseline_matrix.json"),
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


class AdvancementProbe:
    """Proxy that records when rewards arrive relative to online learning."""

    def __init__(self, env: Any, warmup: int):
        self.env = env
        self.warmup = int(warmup)
        self.first_level_advance_action: int | None = None
        self.level_advance_actions: list[int] = []

    def __getattr__(self, name: str):
        return getattr(self.env, name)

    def reset(self):
        return self.env.reset()

    def reset_level(self):
        return self.env.reset_level()

    def step(self, action: int, x: int | None = None, y: int | None = None):
        result = self.env.step(action, x, y)
        if result.level_won:
            action_count = int(self.env.action_count)
            self.level_advance_actions.append(action_count)
            if self.first_level_advance_action is None:
                self.first_level_advance_action = action_count
        return result

    @property
    def advanced_before_online_learning(self) -> bool:
        return (
            self.first_level_advance_action is not None
            and self.first_level_advance_action <= self.warmup
        )


def _values(raw: list[Any], defaults: Iterable[Any]) -> tuple[Any, ...]:
    return tuple(raw) if raw else tuple(defaults)


def _bool_values(raw: list[str]) -> tuple[bool, ...]:
    values = raw or ["off", "on"]
    return tuple(value == "on" for value in values)


def _run_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["game"],
        row["seed"],
        row["budget"],
        row["mode"],
        row["history"],
        row["graph_planning"],
        row["differential_complexity"],
    )


def _load_finished(path: Path) -> tuple[set[tuple[Any, ...]], list[dict[str, Any]]]:
    keys: set[tuple[Any, ...]] = set()
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return keys, rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and all(
            name in row
            for name in (
                "game",
                "seed",
                "budget",
                "mode",
                "history",
                "graph_planning",
                "differential_complexity",
            )
        ):
            keys.add(_run_key(row))
            rows.append(row)
    return keys, rows


def _solve(
    path: Path,
    arcengine: Any,
    *,
    seed: int,
    budget: int,
    mode: str,
    checkpoint: Path,
    history: bool,
    history_len: int,
    graph_planning: bool,
    differential_complexity: bool,
    max_levels: int,
) -> dict[str, Any]:
    solver = AblationSubmissionDRLQNSolver(
        drlqn_xs(),
        seed=seed,
        history_enabled=history,
        history_len=history_len,
        graph_planning=graph_planning,
        differential_complexity=differential_complexity,
    )
    checkpoint_metadata: dict[str, Any] = {}
    if mode == "duck-sft":
        checkpoint_metadata = load_solver_checkpoint(solver, checkpoint)

    env = AdvancementProbe(make_arcade_env(path, arcengine, seed=seed), solver.sched.warmup)
    budgets = {level: int(budget) for level in range(1, int(max_levels) + 1)}
    result = solver.run_env(env, budgets)
    return {
        "game": path.stem,
        "environment_path": str(path),
        "seed": int(seed),
        "budget": int(budget),
        "mode": mode,
        "history": bool(history),
        "history_len": int(history_len if history else 0),
        "graph_planning": bool(graph_planning),
        "differential_complexity": bool(differential_complexity),
        "won": bool(result.won),
        "levels_completed": int(result.completed),
        "actions_per_level": result.actions_per_level,
        "total_actions": int(result.total_actions),
        "updates": int(result.updates),
        "online_learning_started": bool(result.updates > 0),
        "first_level_advance_action": env.first_level_advance_action,
        "level_advance_actions": env.level_advance_actions,
        "advanced_before_online_learning": env.advanced_before_online_learning,
        "wall_clock_s": float(result.wall_clock_s),
        "q_margin_mean": float(result.q_margin_mean),
        "delta_support_mean": float(result.delta_support_mean),
        "telemetry": result.telemetry,
        "checkpoint": str(checkpoint) if mode == "duck-sft" else None,
        "checkpoint_source": checkpoint_metadata.get("source"),
    }


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("error"):
            continue
        key = (
            row["budget"],
            row["mode"],
            row["history"],
            row["graph_planning"],
            row["differential_complexity"],
        )
        groups[key].append(row)

    result = []
    for key, group in sorted(groups.items(), key=lambda item: tuple(str(v) for v in item[0])):
        budget, mode, history, graph, differential = key
        games = {row["game"] for row in group}
        seeds = {int(row["seed"]) for row in group}
        result.append(
            {
                "budget": int(budget),
                "mode": mode,
                "history": bool(history),
                "graph_planning": bool(graph),
                "differential_complexity": bool(differential),
                "runs": len(group),
                "games": len(games),
                "seeds": len(seeds),
                "win_rate": float(np.mean([row["won"] for row in group])),
                "mean_levels_completed": float(
                    np.mean([row["levels_completed"] for row in group])
                ),
                "mean_total_actions": float(np.mean([row["total_actions"] for row in group])),
                "online_learning_activation_rate": float(
                    np.mean([row["online_learning_started"] for row in group])
                ),
                "early_level_advance_rate": float(
                    np.mean([row["advanced_before_online_learning"] for row in group])
                ),
                "level_advance_rate": float(
                    np.mean([row["first_level_advance_action"] is not None for row in group])
                ),
                "median_first_level_advance_action": (
                    float(
                        np.median(
                            [
                                row["first_level_advance_action"]
                                for row in group
                                if row["first_level_advance_action"] is not None
                            ]
                        )
                    )
                    if any(row["first_level_advance_action"] is not None for row in group)
                    else None
                ),
            }
        )
    return result


def main() -> int:
    args = _parse_args()
    budgets = tuple(int(value) for value in _values(args.budget, DEFAULT_BUDGETS))
    seeds = tuple(int(value) for value in _values(args.seed, DEFAULT_SEEDS))
    modes = tuple(str(value) for value in _values(args.mode, ("cold", "duck-sft")))
    histories = _bool_values(args.history)
    graphs = _bool_values(args.graph)
    differentials = _bool_values(args.differential)
    if any(value <= 0 for value in budgets) or args.max_levels <= 0:
        raise SystemExit("budgets and --max-levels must be positive")
    if args.history_len <= 0:
        raise SystemExit("--history-len must be positive")
    if "duck-sft" in modes and not args.duck_checkpoint.exists():
        raise SystemExit(f"Duck-SFT checkpoint does not exist: {args.duck_checkpoint}")

    arcengine = load_arcengine()
    selected = set(args.game)
    files = environment_files(args.root)
    if selected:
        files = [path for path in files if path.stem in selected]
    if not files:
        raise SystemExit(f"no environment files found under {args.root}")

    args.rows.parent.mkdir(parents=True, exist_ok=True)
    finished, rows = _load_finished(args.rows) if args.resume else (set(), [])
    mode = "a" if args.resume else "w"
    errors = 0
    with args.rows.open(mode, encoding="utf-8") as output:
        for path, seed, budget, init_mode, history, graph, differential in product(
            files,
            seeds,
            budgets,
            modes,
            histories,
            graphs,
            differentials,
        ):
            identity = (
                path.stem,
                seed,
                budget,
                init_mode,
                history,
                graph,
                differential,
            )
            if identity in finished:
                continue
            try:
                row = _solve(
                    path,
                    arcengine,
                    seed=seed,
                    budget=budget,
                    mode=init_mode,
                    checkpoint=args.duck_checkpoint,
                    history=history,
                    history_len=args.history_len,
                    graph_planning=graph,
                    differential_complexity=differential,
                    max_levels=args.max_levels,
                )
            except Exception as exc:  # retain all failures in one matrix run
                errors += 1
                row = {
                    "game": path.stem,
                    "seed": int(seed),
                    "budget": int(budget),
                    "mode": init_mode,
                    "history": bool(history),
                    "graph_planning": bool(graph),
                    "differential_complexity": bool(differential),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            output.write(json.dumps(row, sort_keys=True) + "\n")
            output.flush()
            rows.append(row)
            print(json.dumps(row, sort_keys=True))

    payload = {
        "matrix": {
            "budgets": list(budgets),
            "seeds": list(seeds),
            "modes": list(modes),
            "history": list(histories),
            "graph_planning": list(graphs),
            "differential_complexity": list(differentials),
            "games": [path.stem for path in files],
            "warmup_actions": int(PhaseSchedule().warmup),
        },
        "aggregate": _aggregate(rows),
        "errors": [row for row in rows if row.get("error")],
        "rows_file": str(args.rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"aggregate_groups": len(payload["aggregate"]), "errors": errors}))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
