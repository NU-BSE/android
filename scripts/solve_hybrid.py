"""Run the Duck-inspired HybridDRLQNSolver on all local environments."""
from __future__ import annotations

import json
from pathlib import Path

from drlqn.config import drlqn_xs
from drlqn.env.games import make, REGISTRY
from drlqn.eval.baselines import optimal_actions
from drlqn.train.hybrid_solver import HybridDRLQNSolver


def main() -> None:
    rows = []
    for index, env_id in enumerate(REGISTRY):
        optimal = optimal_actions(env_id)
        budgets = {level: max(24, 12 * count) for level, count in enumerate(optimal, 1)}
        result = HybridDRLQNSolver(drlqn_xs(), seed=index).run_env(make(env_id), budgets)
        rows.append({
            "env_id": result.env_id,
            "completed": result.completed,
            "won": result.won,
            "actions_per_level": result.actions_per_level,
            "total_actions": result.total_actions,
            "updates": result.updates,
            "wall_clock_s": result.wall_clock_s,
            "telemetry": result.telemetry,
        })
        print(env_id, rows[-1])
    out = Path("results/hybrid_results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
