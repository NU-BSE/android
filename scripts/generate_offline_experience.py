"""Generate random, frontier and specialist-planner experience on all official games."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from drlqn.config import drlqn_xs
from drlqn.env.official_loader import environment_files, load_arcengine, make_arcade_env
from drlqn.train.ablation_solver import AblationSubmissionDRLQNSolver
from drlqn.train.duck_dataset import save_examples_npz
from drlqn.train.offline_experience import collect_trajectory, experience_summary
from drlqn.train.trainer import PhaseSchedule


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("arc-prize-2026-arc-agi-3/environment_files"))
    parser.add_argument("--game", action="append", default=[])
    parser.add_argument("--seed", action="append", type=int, default=[])
    parser.add_argument("--policy", action="append", choices=("random", "frontier", "specialist"), default=[])
    parser.add_argument("--max-actions-per-level", type=int, default=300)
    parser.add_argument("--max-levels", type=int, default=128)
    parser.add_argument("--output", type=Path, default=Path("data/environment_generated_transitions.npz"))
    parser.add_argument("--manifest", type=Path, default=Path("results/environment_generated_manifest.json"))
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    seeds = tuple(args.seed or range(5))
    policies = tuple(args.policy or ("random", "frontier", "specialist"))
    if args.max_actions_per_level <= 0 or args.max_levels <= 0:
        raise SystemExit("action and level limits must be positive")
    arcengine = load_arcengine()
    files = environment_files(args.root)
    selected = set(args.game)
    if selected:
        files = [path for path in files if path.stem in selected]
    if not files:
        raise SystemExit(f"no official environment files found under {args.root}")

    examples = []
    runs = []
    errors = []
    for path in files:
        for seed in seeds:
            for policy in policies:
                try:
                    schedule = PhaseSchedule(warmup=10**9, diff_losses_at=10**9, approx_at=10**9)
                    solver = AblationSubmissionDRLQNSolver(
                        drlqn_xs(),
                        seed=int(seed),
                        schedule=schedule,
                        history_enabled=True,
                        graph_planning=policy != "random",
                        planner_portfolio=policy == "specialist",
                        differential_complexity=False,
                    )
                    env = make_arcade_env(path, arcengine, seed=int(seed))
                    trajectory = collect_trajectory(
                        env,
                        solver,
                        policy=policy,
                        seed=int(seed),
                        max_actions_per_level=int(args.max_actions_per_level),
                        max_levels=int(args.max_levels),
                    )
                    examples.extend(trajectory)
                    row = {
                        "game": path.stem,
                        "seed": int(seed),
                        "policy": policy,
                        "transitions": len(trajectory),
                        "won": bool(trajectory and trajectory[-1].success),
                        "levels_completed": int(getattr(env, "_levels_completed", 0)),
                        "total_actions": int(env.action_count),
                    }
                except Exception as exc:
                    row = {
                        "game": path.stem,
                        "seed": int(seed),
                        "policy": policy,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    errors.append(row)
                runs.append(row)
                print(json.dumps(row, sort_keys=True))

    if not examples:
        raise SystemExit("offline generation produced no transitions")
    save_examples_npz(examples, args.output)
    payload = {
        "config": {
            "games": [path.stem for path in files],
            "seeds": list(seeds),
            "policies": list(policies),
            "max_actions_per_level": int(args.max_actions_per_level),
            "max_levels": int(args.max_levels),
        },
        "summary": experience_summary(examples),
        "runs": runs,
        "errors": errors,
        "dataset": str(args.output),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"dataset: {args.output}")
    print(f"manifest: {args.manifest}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
