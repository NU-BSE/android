from __future__ import annotations

from drlqn.config import drlqn_xs
from drlqn.train.ablation_solver import AblationSubmissionDRLQNSolver


def test_ablation_switches_disable_history_and_differential_schedule():
    solver = AblationSubmissionDRLQNSolver(
        drlqn_xs(),
        seed=0,
        history_enabled=False,
        graph_planning=False,
        differential_complexity=False,
    )
    assert solver.history_len == 0
    assert solver._recent_effects.maxlen == 0
    assert not solver.graph_planning
    assert not solver.differential_complexity
    assert solver.sched.diff_losses_at > 10**9
    assert solver.sched.approx_at > 10**9
