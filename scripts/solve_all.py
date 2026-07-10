#!/usr/bin/env python
"""Solve all local ARC-AGI-3-style environments with DRLQN and score RHAE.

Budget note: the official leaderboard terminates at 5x the human baseline per
level; we run locally with 12x (no API cost constraint) and report both the
raw RHAE and the 5x-capped variant (levels solved beyond 5x count as
uncompleted there).
"""
import json
import sys
import time

sys.path.insert(0, ".")

import numpy as np

from drlqn.config import drlqn_xs
from drlqn.env.games import make, REGISTRY
from drlqn.eval.baselines import optimal_actions, human_baseline
from drlqn.eval.metrics import level_score, environment_score, total_score
from drlqn.train.trainer import DRLQNSolver, PhaseSchedule

BUDGET_MULT = 12
SEED = 0


def main():
    results = {}
    sched = PhaseSchedule(diff_losses_at=60, approx_at=120, train_every=1,
                          warmup=8, batch=12)
    for eid in REGISTRY:
        t0 = time.time()
        env = make(eid)
        opt = optimal_actions(eid)
        hb = human_baseline(eid)
        solver = DRLQNSolver(drlqn_xs(), seed=SEED, schedule=sched)
        res = solver.run_env(env, {l: BUDGET_MULT * h for l, h in hb.items()})
        acts = {int(l): int(a) for l, a in res.actions_per_level.items()}
        scores = {l: level_score(hb[l], acts.get(l)) if l <= res.completed else 0.0
                  for l in hb}
        e_score = environment_score(hb, acts, res.completed)
        # 5x-capped variant (official leaderboard budget)
        acts5 = {l: a for l, a in acts.items() if l in hb and a <= 5 * hb[l]}
        comp5 = 0
        for l in sorted(hb):
            if l in acts5 and l == comp5 + 1 and l <= res.completed:
                comp5 = l
            else:
                break
        e_score5 = environment_score(hb, acts5, comp5)
        results[eid] = {
            "completed_levels": res.completed,
            "won": res.won,
            "optimal": opt,
            "human_baseline": hb,
            "actions_per_level": acts,
            "level_scores": {l: round(s, 4) for l, s in scores.items()},
            "environment_score": round(e_score, 4),
            "environment_score_5x_budget": round(e_score5, 4),
            "total_actions": res.total_actions,
            "updates": res.updates,
            "wall_clock_s": round(res.wall_clock_s, 1),
            "q_margin_mean": round(res.q_margin_mean, 4),
            "delta_support_mean": round(res.delta_support_mean, 4),
            "runtime_telemetry": res.telemetry,
            "column_usage_nonzero": int(np.count_nonzero(res.column_usage)),
        }
        print(f"{eid}: {res.completed}/6 levels, env score "
              f"{e_score:.3f} (5x-budget {e_score5:.3f}), "
              f"{res.total_actions} actions, {time.time()-t0:.0f}s", flush=True)

    results["TOTAL"] = {
        "RHAE": round(total_score(
            [results[e]["environment_score"] for e in REGISTRY]), 4),
        "RHAE_5x_budget": round(total_score(
            [results[e]["environment_score_5x_budget"] for e in REGISTRY]), 4),
        "environments_won": sum(results[e]["won"] for e in REGISTRY),
        "levels_completed": sum(results[e]["completed_levels"] for e in REGISTRY),
        "levels_total": 6 * len(REGISTRY),
    }
    with open("results/results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(json.dumps(results["TOTAL"], indent=2))


if __name__ == "__main__":
    main()
