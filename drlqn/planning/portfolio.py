"""Generic planner portfolio over DRLQN's discovered transition graph.

The specialists never inspect game source or game identifiers. They consume rendered
state-derived nodes, legal lambda actions and effect statistics accumulated online.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from ..operators.library import is_click_lambda, lambda_for_action, to_env_action


@dataclass(frozen=True)
class PlannerProposal:
    lambda_id: int
    score: float
    specialist: str
    reason: str


@dataclass(frozen=True)
class MechanicBelief:
    movement: float
    toggle: float
    click_construction: float
    selection_commit: float
    unknown: float

    def as_dict(self) -> dict[str, float]:
        return {
            "movement": self.movement,
            "toggle": self.toggle,
            "click_construction": self.click_construction,
            "selection_commit": self.selection_commit,
            "unknown": self.unknown,
        }


class MechanicClassifier:
    """Infer a soft mechanic mixture from observed action effects."""

    def classify(self, solver: Any, node: Any) -> MechanicBelief:
        evidence = {
            "movement": 0.35,
            "toggle": 0.25,
            "click_construction": 0.25,
            "selection_commit": 0.15,
            "unknown": 0.65,
        }
        observed = 0.0
        cycle_edges = 0
        graph = getattr(solver, "graph", {})
        for source, edges in graph.items():
            for lam, target in edges.items():
                stats = getattr(solver, "edge_stats", {}).get((source, int(lam)))
                if not stats or stats[0] <= 0:
                    continue
                tries, changes, advances = (float(stats[0]), float(stats[1]), float(stats[2]))
                observed += tries
                changed_rate = changes / tries
                advance_rate = advances / tries
                if 0 <= int(lam) < 4:
                    evidence["movement"] += 1.4 * changed_rate + 3.0 * advance_rate
                elif is_click_lambda(int(lam)):
                    evidence["click_construction"] += 1.2 * changed_rate + 2.5 * advance_rate
                    evidence["toggle"] += 0.35 * changed_rate
                else:
                    evidence["toggle"] += 0.8 * changed_rate
                    evidence["selection_commit"] += 2.5 * advance_rate
                if target == source or source in graph.get(target, {}).values():
                    cycle_edges += 1
        if cycle_edges:
            evidence["toggle"] += min(2.0, 0.25 * cycle_edges)

        valid = np.where(np.asarray(node.valid, dtype=bool))[0]
        click_fraction = float(np.mean([is_click_lambda(int(lam)) for lam in valid])) if len(valid) else 0.0
        evidence["click_construction"] += 1.25 * click_fraction
        try:
            commit = lambda_for_action(7)
        except ValueError:
            commit = -1
        if 0 <= commit < len(node.valid) and bool(node.valid[commit]):
            evidence["selection_commit"] += 1.5

        evidence["unknown"] += 1.5 / np.sqrt(1.0 + observed)
        names = tuple(evidence)
        logits = np.asarray([evidence[name] for name in names], dtype=np.float64)
        logits -= logits.max()
        weights = np.exp(logits)
        weights /= max(float(weights.sum()), 1e-12)
        values = dict(zip(names, weights.tolist()))
        return MechanicBelief(**values)


def _untried(solver: Any, node: Any) -> list[int]:
    if hasattr(solver, "_untried"):
        return [int(value) for value in solver._untried(node)]
    tried = getattr(solver, "graph", {}).get(node.key, {})
    return [int(value) for value in np.where(node.valid)[0] if int(value) not in tried]


def _q_scores(solver: Any, node: Any, lambdas: Iterable[int]) -> dict[int, float]:
    values = list(lambdas)
    if not values:
        return {}
    try:
        q = np.asarray(solver.q_of(node), dtype=np.float64)
    except Exception:
        q = np.zeros(len(node.valid), dtype=np.float64)
    result: dict[int, float] = {}
    for lam in values:
        bonus = 0.0
        try:
            bonus = float(solver._prior_bonus(node, int(lam)))
        except Exception:
            pass
        result[int(lam)] = float(q[int(lam)] + bonus)
    return result


def _first_edge_path(solver: Any, node: Any, predicate, max_depth: int = 12) -> tuple[int, int] | None:
    graph = getattr(solver, "graph", {})
    nodes = getattr(solver, "nodes", {})
    queue = deque([(node.key, 0)])
    first: dict[str, int | None] = {node.key: None}
    while queue:
        key, depth = queue.popleft()
        info = nodes.get(key)
        if info is None:
            continue
        if key != node.key and predicate(info):
            action = first[key]
            if action is not None:
                return int(action), depth
        if depth >= max_depth:
            continue
        for lam, next_key in graph.get(key, {}).items():
            if next_key in first or next_key == key or next_key not in nodes:
                continue
            if getattr(nodes[next_key], "level", node.level) != node.level:
                continue
            first[next_key] = first[key] if first[key] is not None else int(lam)
            queue.append((next_key, depth + 1))
    return None


class MovementPlanner:
    name = "movement"

    def propose(self, solver: Any, node: Any) -> PlannerProposal | None:
        winning = getattr(solver, "winning_actions", {}).get(node.key)
        if winning is not None and bool(node.valid[int(winning)]):
            return PlannerProposal(int(winning), 12.0, self.name, "known level-advance edge")
        routed = _first_edge_path(
            solver,
            node,
            lambda info: any(0 <= lam < 4 for lam in _untried(solver, info)),
        )
        if routed is not None:
            lam, depth = routed
            return PlannerProposal(lam, 7.0 - 0.15 * depth, self.name, "BFS to movement frontier")
        candidates = [lam for lam in _untried(solver, node) if 0 <= lam < 4]
        scores = _q_scores(solver, node, candidates)
        if not scores:
            return None
        lam = max(scores, key=scores.get)
        return PlannerProposal(lam, 3.0 + scores[lam], self.name, "untried movement effect")


class TogglePlanner:
    name = "toggle"

    def propose(self, solver: Any, node: Any) -> PlannerProposal | None:
        graph = getattr(solver, "graph", {})
        nodes = getattr(solver, "nodes", {})
        beam: list[tuple[float, str, int | None, int]] = [(0.0, node.key, None, 0)]
        best: tuple[float, int, str] | None = None
        seen = {node.key: 0.0}
        for _ in range(5):
            expanded: list[tuple[float, str, int | None, int]] = []
            for score, key, first, depth in beam:
                info = nodes.get(key)
                if info is None:
                    continue
                candidates = [lam for lam in _untried(solver, info) if not (0 <= lam < 4)]
                if candidates and first is not None:
                    local = _q_scores(solver, info, candidates)
                    value = score + max(local.values(), default=0.0) + 0.25 * len(candidates)
                    if best is None or value > best[0]:
                        best = (value, int(first), "beam search to deduplicated configuration frontier")
                for lam, next_key in graph.get(key, {}).items():
                    if next_key not in nodes or getattr(nodes[next_key], "level", node.level) != node.level:
                        continue
                    stats = getattr(solver, "edge_stats", {}).get((key, int(lam)), [1.0, 0.0, 0.0])
                    tries = max(float(stats[0]), 1.0)
                    edge_score = 0.4 * float(stats[1]) / tries + 2.0 * float(stats[2]) / tries
                    next_score = score + edge_score - 0.08 * depth
                    if next_score <= seen.get(next_key, -1e18):
                        continue
                    seen[next_key] = next_score
                    expanded.append((next_score, next_key, int(lam) if first is None else first, depth + 1))
            beam = sorted(expanded, reverse=True)[:12]
        if best is not None:
            return PlannerProposal(best[1], 5.0 + best[0], self.name, best[2])
        candidates = [lam for lam in _untried(solver, node) if not is_click_lambda(lam) and not (0 <= lam < 4)]
        scores = _q_scores(solver, node, candidates)
        if not scores:
            return None
        lam = max(scores, key=scores.get)
        return PlannerProposal(lam, 2.5 + scores[lam], self.name, "configuration hypothesis test")


class ClickConstructionPlanner:
    name = "click_construction"

    def _coordinates(self, node: Any, lambdas: Iterable[int]) -> dict[int, tuple[int, int]]:
        coords = {}
        for lam in lambdas:
            try:
                _, x, y = to_env_action(int(lam), node.objs)
            except (IndexError, ValueError):
                continue
            coords[int(lam)] = (int(x), int(y))
        return coords

    def propose(self, solver: Any, node: Any) -> PlannerProposal | None:
        candidates = [lam for lam in _untried(solver, node) if is_click_lambda(lam)]
        if not candidates:
            routed = _first_edge_path(
                solver,
                node,
                lambda info: any(is_click_lambda(lam) for lam in _untried(solver, info)),
                max_depth=8,
            )
            if routed is None:
                return None
            lam, depth = routed
            return PlannerProposal(lam, 5.5 - 0.15 * depth, self.name, "route to click frontier")

        scores = _q_scores(solver, node, candidates)
        coords = self._coordinates(node, candidates)
        changed_clicks: list[tuple[int, int]] = []
        for (source, lam), stats in getattr(solver, "edge_stats", {}).items():
            if source != node.key or not is_click_lambda(int(lam)) or not stats or stats[0] <= 0:
                continue
            if float(stats[1]) / float(stats[0]) > 0.5:
                try:
                    _, x, y = to_env_action(int(lam), node.objs)
                    changed_clicks.append((int(x), int(y)))
                except (IndexError, ValueError):
                    pass
        for lam, (x, y) in coords.items():
            for px, py in changed_clicks:
                symmetry_distance = min(
                    abs(x - (63 - px)) + abs(y - py),
                    abs(x - px) + abs(y - (63 - py)),
                    abs(x - (63 - px)) + abs(y - (63 - py)),
                )
                scores[lam] += 1.25 * max(0.0, 1.0 - symmetry_distance / 16.0)
            try:
                kind = str(getattr(node.objs[(int(lam) - 5) // 5], "kind", ""))
            except Exception:
                kind = ""
            if kind in {"tile", "changed"}:
                scores[lam] += 0.6
        lam = max(scores, key=scores.get)
        return PlannerProposal(lam, 3.5 + scores[lam], self.name, "tile-grid and symmetry coordinate search")


class SelectionCommitPlanner:
    name = "selection_commit"

    def propose(self, solver: Any, node: Any) -> PlannerProposal | None:
        try:
            commit = lambda_for_action(7)
        except ValueError:
            return None
        if commit >= len(node.valid) or not bool(node.valid[commit]):
            return None
        tried = getattr(solver, "graph", {}).get(node.key, {})
        if commit not in tried:
            changed_evidence = 0.0
            for (source, _lam), stats in getattr(solver, "edge_stats", {}).items():
                if source == node.key and stats and stats[0] > 0:
                    changed_evidence = max(changed_evidence, float(stats[1]) / float(stats[0]))
            score = 4.0 + 2.0 * changed_evidence
            return PlannerProposal(commit, score, self.name, "ACTION7 commit hypothesis")
        next_key = tried.get(commit)
        stats = getattr(solver, "edge_stats", {}).get((node.key, commit), [1.0, 0.0, 0.0])
        if next_key is not None and float(stats[2]) > 0:
            return PlannerProposal(commit, 12.0, self.name, "verified ACTION7 level advance")
        return None


class UnknownMechanicsPlanner:
    name = "unknown"

    def propose(self, solver: Any, node: Any) -> PlannerProposal | None:
        candidates = _untried(solver, node)
        scores = _q_scores(solver, node, candidates)
        if not scores:
            return None
        lam = max(scores, key=scores.get)
        return PlannerProposal(lam, scores[lam], self.name, "normal DRLQN exploration")


class PlannerPortfolio:
    """Blend generic specialist proposals using a soft mechanic classifier."""

    def __init__(self) -> None:
        self.classifier = MechanicClassifier()
        self.specialists = {
            "movement": MovementPlanner(),
            "toggle": TogglePlanner(),
            "click_construction": ClickConstructionPlanner(),
            "selection_commit": SelectionCommitPlanner(),
            "unknown": UnknownMechanicsPlanner(),
        }
        self.selection_counts: dict[str, int] = defaultdict(int)
        self.last_belief = MechanicBelief(0.0, 0.0, 0.0, 0.0, 1.0)
        self.last_proposals: tuple[PlannerProposal, ...] = ()

    def select(self, solver: Any, node: Any) -> int | None:
        belief = self.classifier.classify(solver, node)
        self.last_belief = belief
        weighted: dict[int, float] = defaultdict(float)
        provenance: dict[int, tuple[str, float]] = {}
        proposals: list[PlannerProposal] = []
        for name, specialist in self.specialists.items():
            proposal = specialist.propose(solver, node)
            if proposal is None or not bool(node.valid[proposal.lambda_id]):
                continue
            proposals.append(proposal)
            contribution = belief.as_dict()[name] * proposal.score
            weighted[proposal.lambda_id] += contribution
            if contribution > provenance.get(proposal.lambda_id, ("", -1e18))[1]:
                provenance[proposal.lambda_id] = (name, contribution)
        self.last_proposals = tuple(proposals)
        if not weighted:
            return None
        selected = max(weighted, key=weighted.get)
        specialist = provenance[selected][0]
        self.selection_counts[specialist] += 1
        return int(selected)

    def telemetry(self) -> dict[str, Any]:
        return {
            "belief": self.last_belief.as_dict(),
            "selection_counts": dict(self.selection_counts),
            "proposals": [proposal.__dict__ for proposal in self.last_proposals],
        }
