from types import SimpleNamespace

import numpy as np

from drlqn.operators.library import lambda_for_action
from drlqn.planning.portfolio import MechanicClassifier, PlannerPortfolio


class FakeSolver:
    def __init__(self):
        self.graph = {"s0": {0: "s1"}, "s1": {1: "s2"}}
        self.nodes = {}
        self.edge_stats = {("s0", 0): [2.0, 2.0, 0.0], ("s1", 1): [1.0, 1.0, 1.0]}
        self.winning_actions = {"s1": 1}

    def _untried(self, node):
        tried = self.graph.get(node.key, {})
        return [int(lam) for lam in np.where(node.valid)[0] if int(lam) not in tried]

    def q_of(self, node):
        return np.zeros(len(node.valid))

    def _prior_bonus(self, node, lam):
        return 0.0


def _node(key, *, action7=False):
    valid = np.zeros(128, dtype=bool)
    valid[:5] = True
    if action7:
        valid[lambda_for_action(7)] = True
    return SimpleNamespace(key=key, valid=valid, level=1, objs=[], grid=np.zeros((64, 64), dtype=np.uint8))


def test_mechanic_classifier_prefers_movement_after_reliable_key_effects():
    solver = FakeSolver()
    node0 = _node("s0")
    node1 = _node("s1")
    node2 = _node("s2")
    solver.nodes = {"s0": node0, "s1": node1, "s2": node2}
    belief = MechanicClassifier().classify(solver, node0)
    assert belief.movement > belief.unknown


def test_portfolio_routes_over_shared_graph_to_known_level_advance():
    solver = FakeSolver()
    node0 = _node("s0")
    node1 = _node("s1")
    node2 = _node("s2")
    solver.nodes = {"s0": node0, "s1": node1, "s2": node2}
    selected = PlannerPortfolio().select(solver, node0)
    assert selected == 0


def test_selection_commit_specialist_tests_action7_without_game_specific_logic():
    solver = FakeSolver()
    node = _node("commit", action7=True)
    solver.graph = {"commit": {}}
    solver.nodes = {"commit": node}
    solver.edge_stats = {}
    solver.winning_actions = {}
    selected = PlannerPortfolio().select(solver, node)
    assert selected == lambda_for_action(7)
