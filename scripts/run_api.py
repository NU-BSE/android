#!/usr/bin/env python
"""Run the DRLQN solver against the official ARC-AGI-3 API.

Usage: ARC_API_KEY=... python scripts/run_api.py [game_id ...]
"""
import sys

sys.path.insert(0, ".")

from drlqn.api.arc_client import ArcClient, RemoteEnv
from drlqn.config import drlqn_xs
from drlqn.train.trainer import DRLQNSolver, PhaseSchedule


def main():
    client = ArcClient()
    games = sys.argv[1:] or [g["game_id"] for g in client.list_games()]
    card = client.open_scorecard()
    try:
        for gid in games:
            env = RemoteEnv(client, gid, card)
            solver = DRLQNSolver(drlqn_xs(), seed=0,
                                 schedule=PhaseSchedule(train_every=1, warmup=8,
                                                        batch=12))
            res = solver.run_env(env, budgets={l: 400 for l in range(1, 12)})
            print(gid, "levels:", res.completed, "actions:", res.total_actions)
    finally:
        print(client.close_scorecard(card))


if __name__ == "__main__":
    main()
