"""Generic specialist planner portfolio over the shared DRLQN transition graph.

The specialists use only rendered states, legal lambda actions and discovered
transitions. They deliberately contain no game identifiers or game-specific
rules. A light-weight mechanic classifier turns observed action effects into
mixture weights for movement, toggle/configuration, click-construction,
selection/commit and unknown-mechanic exploration policies.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
import hashlib
from typing import Any

import numpy as np

from ..operators.library import (
    click_slot_offset,
    is_click_lambda,
    lambda_for_action,
    to_env_action,
)
from .submission_solver import SubmissionDRLQNSolver


SPECIALISTS = ("movement", "toggle", "click", "commit", "unknown")


def _family(lam: int) -> str:
    lam = int(lam)
    if 0 <= lam < 4:
        return "movement"
    if is_click_lambda(lam):
        return "click"
    try:
        if lam == lambda_for_action(7):
            return "commit"
    except ValueError:
        pass
    if lam == 4 or lam >= 125:
        return "toggle"
    return "unknown"


def _visual_key(node: Any) -> str:
    return hashlib.sha1(
        np.asarray(node.grid, dtype=np.uint8).tobytes()
        + int(node.level).to_bytes(2, "little", signed=False)
    ).hexdigest()[:16]


def _changed_centroid(left: np.ndarray, right: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.where(np.asarray(left) != np.asarray(right))
    if len(ys) == 0:
        return None
    return float(ys.mean()), float(xs.mean())


@dataclass(frozen=True)
class MechanicScores:
    movement: float
    toggle: float
    click: float
    commit: float
    unknown: float

    def as_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in SPECIALISTS}


class MechanicClassifier:
    """Online effect statistics used to weight generic planner specialists."""

    def __init__(self) -> None:
        self.trials: Counter[str] = Counter()
        self.changed: Counter[str] = Counter()
        self.advanced: Counter[str] = Counter()
        self.self_loops: Counter[str] = Counter()
        self.local_changes: Counter[str] = Counter()
        self._movement_vectors: list[np.ndarray] = []
        self.last_family = "unknown"
        self.last_changed = False
        self.last_advanced = False

    def observe(self, current: Any, lam: int, nxt: Any, changed: bool, advanced: bool) -> None:
        family = _family(lam)
        self.trials[family] += 1
        self.changed[family] += int(changed)
        self.advanced[family] += int(advanced)
        self.self_loops[family] += int(current.key == nxt.key)
        self.last_family = family
        self.last_changed = bool(changed)
        self.last_advanced = bool(advanced)

        if changed:
            delta_cells = int(np.count_nonzero(np.asarray(current.grid) != np.asarray(nxt.grid)))
            self.local_changes[family] += int(delta_cells <= 128)
        if family == "movement" and changed and not advanced:
            centroid = _changed_centroid(current.grid, nxt.grid)
            if centroid is not None:
                vector = np.asarray([centroid[0] - 31.5, centroid[1] - 31.5])
                norm = float(np.linalg.norm(vector))
                if norm > 1e-6:
                    self._movement_vectors.append(vector / norm)
                    self._movement_vectors = self._movement_vectors[-32:]

    def scores(self) -> MechanicScores:
        def rate(counter: Counter[str], name: str) -> float:
            return float(counter[name] + 1.0) / float(self.trials[name] + 3.0)

        movement_consistency = 0.0
        if self._movement_vectors:
            movement_consistency = float(
                np.linalg.norm(np.mean(np.stack(self._movement_vectors), axis=0))
            )
        movement = 0.25 + 1.8 * rate(self.changed, "movement") + 0.8 * movement_consistency
        toggle = (
            0.25
            + 1.4 * rate(self.changed, "toggle")
            + 0.7 * rate(self.local_changes, "toggle")
        )
        click = 0.25 + 1.8 * rate(self.changed, "click") + 0.5 * rate(self.local_changes, "click")
        commit = (
            0.15
            + 2.5 * rate(self.advanced, "commit")
            + (0.8 if self.last_changed and self.last_family in {"click", "toggle"} else 0.0)
        )
        observed = sum(self.trials.values())
        unknown = 1.2 / np.sqrt(float(observed + 1)) + 0.15
        raw = np.asarray([movement, toggle, click, commit, unknown], dtype=np.float64)
        raw = np.maximum(raw, 1e-6)
        raw /= raw.sum()
        return MechanicScores(*[float(value) for value in raw])


@dataclass(frozen=True)
class SpecialistProposal:
    specialist: str
    lam: int
    score: float
    rationale: str


class PlannerPortfolio:
    """Combine generic specialists over one discovered transition graph."""

    def __init__(self, solver: SubmissionDRLQNSolver, beam_width: int = 12, beam_depth: int = 5):
        self.solver = solver
        self.classifier = MechanicClassifier()
        self.beam_width = max(2, int(beam_width))
        self.beam_depth = max(1, int(beam_depth))
        self.decisions: Counter[str] = Counter()
        self.last_proposals: list[SpecialistProposal] = []

    def observe(self, current: Any, lam: int, nxt: Any, changed: bool, advanced: bool) -> None:
        self.classifier.observe(current, lam, nxt, changed, advanced)

    def _untried_family(self, node: Any, family: str) -> list[int]:
        return [lam for lam in self.solver._untried(node) if _family(lam) == family]

    def _known_path(
        self,
        node: Any,
        *,
        allowed_families: set[str],
        target_predicate,
        max_depth: int = 16,
    ) -> tuple[int, int] | None:
        queue = deque([(node.key, 0)])
        first: dict[str, int | None] = {node.key: None}
        while queue:
            key, depth = queue.popleft()
            info = self.solver.nodes.get(key)
            if info is None:
                continue
            if key != node.key and target_predicate(info):
                action = first[key]
                if action is not None:
                    return int(action), depth
            if depth >= max_depth:
                continue
            for lam, next_key in self.solver.graph.get(key, {}).items():
                if _family(lam) not in allowed_families:
                    continue
                if next_key in first or next_key not in self.solver.nodes:
                    continue
                if self.solver.nodes[next_key].level != node.level:
                    continue
                first[next_key] = int(lam) if first[key] is None else first[key]
                queue.append((next_key, depth + 1))
        return None

    def movement_proposal(self, node: Any) -> SpecialistProposal | None:
        winning = self.solver.winning_actions.get(node.key)
        if winning is not None and _family(winning) == "movement" and node.valid[winning]:
            return SpecialistProposal("movement", int(winning), 6.0, "known movement advance")

        path = self._known_path(
            node,
            allowed_families={"movement"},
            target_predicate=lambda info: bool(self._untried_family(info, "movement"))
            or info.key in self.solver.winning_actions,
        )
        if path is not None:
            action, distance = path
            return SpecialistProposal("movement", action, 4.0 - 0.12 * distance, "BFS to movement frontier")

        candidates = self._untried_family(node, "movement")
        if not candidates:
            return None
        q_values = self.solver.q_of(node)
        action = max(candidates, key=lambda lam: q_values[lam] + self.solver._prior_bonus(node, lam))
        return SpecialistProposal("movement", int(action), 2.0, "A* local expansion")

    def toggle_proposal(self, node: Any) -> SpecialistProposal | None:
        candidates = self._untried_family(node, "toggle")
        q_values = self.solver.q_of(node)
        if candidates:
            action = max(candidates, key=lambda lam: q_values[lam] + self.solver._prior_bonus(node, lam))
            return SpecialistProposal("toggle", int(action), 2.2, "untried configuration action")

        # Beam-search only over already discovered transitions. Visual-state
        # deduplication prevents history aliases from dominating the beam.
        beam: list[tuple[float, str, int | None, int]] = [(0.0, node.key, None, 0)]
        seen_visual = {_visual_key(node)}
        best: tuple[float, int] | None = None
        for _ in range(self.beam_depth):
            expanded: list[tuple[float, str, int | None, int]] = []
            for score, key, first, depth in beam:
                info = self.solver.nodes.get(key)
                if info is None:
                    continue
                for lam, next_key in self.solver.graph.get(key, {}).items():
                    if _family(lam) != "toggle" or next_key not in self.solver.nodes:
                        continue
                    nxt = self.solver.nodes[next_key]
                    if nxt.level != node.level:
                        continue
                    visual = _visual_key(nxt)
                    if visual in seen_visual:
                        continue
                    seen_visual.add(visual)
                    first_action = int(lam) if first is None else int(first)
                    stats = self.solver.edge_stats.get((key, int(lam)), [1.0, 0.0, 0.0])
                    tried = max(float(stats[0]), 1.0)
                    changed_rate = float(stats[1]) / tried
                    advance_rate = float(stats[2]) / tried
                    novelty = min(8, len(self.solver._untried(nxt)))
                    value = score + 1.4 * changed_rate + 5.0 * advance_rate + 0.12 * novelty - 0.08 * depth
                    expanded.append((value, next_key, first_action, depth + 1))
                    if best is None or value > best[0]:
                        best = (value, first_action)
            beam = sorted(expanded, reverse=True)[: self.beam_width]
            if not beam:
                break
        if best is None:
            return None
        return SpecialistProposal("toggle", best[1], 1.5 + best[0], "deduplicated configuration beam")

    def click_proposal(self, node: Any) -> SpecialistProposal | None:
        candidates = self._untried_family(node, "click")
        if not candidates:
            return None
        q_values = self.solver.q_of(node)
        coords: dict[int, tuple[int, int]] = {}
        for lam in candidates:
            try:
                _, x, y = to_env_action(lam, node.objs)
                coords[int(lam)] = (int(x), int(y))
            except (IndexError, ValueError):
                continue

        def symmetry_bonus(lam: int) -> float:
            xy = coords.get(int(lam))
            if xy is None:
                return 0.0
            x, y = xy
            targets = ((63 - x, y), (x, 63 - y), (63 - x, 63 - y), (y, x))
            return 0.3 if any(
                abs(tx - ox) + abs(ty - oy) <= 3
                for tx, ty in targets
                for ox, oy in coords.values()
            ) else 0.0

        def score(lam: int) -> float:
            slot, offset = click_slot_offset(lam)
            kind = ""
            if slot < len(getattr(node, "proposal_kinds", [])):
                kind = str(node.proposal_kinds[slot])
            kind_bonus = {
                "changed": 1.6,
                "tile": 0.9,
                "object": 0.7,
                "lattice": 0.2,
            }.get(kind, 0.35)
            center_bonus = 0.35 if offset == 0 else 0.0
            return (
                float(q_values[lam])
                + self.solver._prior_bonus(node, lam)
                + kind_bonus
                + center_bonus
                + symmetry_bonus(lam)
            )

        action = max(candidates, key=score)
        return SpecialistProposal("click", int(action), 2.4 + score(action) * 0.05, "tile/symmetry coordinate proposal")

    def commit_proposal(self, node: Any) -> SpecialistProposal | None:
        try:
            commit = lambda_for_action(7)
        except ValueError:
            return None
        if commit >= len(node.valid) or not node.valid[commit]:
            return None
        tried = self.solver.graph.get(node.key, {})
        recent = list(getattr(self.solver, "_recent_effects", ()))
        changed_configuration = bool(
            recent and recent[-1][1] and _family(recent[-1][0]) in {"click", "toggle"}
        )
        if commit not in tried:
            score = 3.0 if changed_configuration else 1.2
            return SpecialistProposal("commit", int(commit), score, "ACTION7 hypothesis test")
        next_key = tried.get(commit)
        if next_key is not None and next_key != node.key:
            return SpecialistProposal("commit", int(commit), 2.0, "repeat known commit transition")
        return None

    def unknown_proposal(self, node: Any) -> SpecialistProposal | None:
        candidates = self.solver._untried(node)
        if not candidates:
            routed = self.solver._route_to_frontier(node)
            if routed is None:
                return None
            return SpecialistProposal("unknown", int(routed), 1.0, "DRLQN graph frontier")
        q_values = self.solver.q_of(node)
        action = max(candidates, key=lambda lam: q_values[lam] + self.solver._prior_bonus(node, lam))
        return SpecialistProposal("unknown", int(action), 1.0, "normal DRLQN expansion")

    def propose(self, node: Any) -> SpecialistProposal | None:
        scores = self.classifier.scores().as_dict()
        proposals = [
            self.movement_proposal(node),
            self.toggle_proposal(node),
            self.click_proposal(node),
            self.commit_proposal(node),
            self.unknown_proposal(node),
        ]
        proposals = [proposal for proposal in proposals if proposal is not None]
        self.last_proposals = proposals
        if not proposals:
            return None
        selected = max(
            proposals,
            key=lambda proposal: scores[proposal.specialist] * proposal.score,
        )
        self.decisions[selected.specialist] += 1
        return selected

    def telemetry(self) -> dict[str, Any]:
        return {
            "mechanic_scores": self.classifier.scores().as_dict(),
            "decisions": dict(self.decisions),
            "trials": dict(self.classifier.trials),
            "changed": dict(self.classifier.changed),
            "advanced": dict(self.classifier.advanced),
            "last_proposals": [proposal.__dict__ for proposal in self.last_proposals],
        }


class PortfolioSubmissionDRLQNSolver(SubmissionDRLQNSolver):
    """Submission solver augmented with the generic specialist portfolio."""

    def __init__(self, *args, portfolio_enabled: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.portfolio_enabled = bool(portfolio_enabled)
        self.portfolio = PlannerPortfolio(self)

    def select(self, node: Any, actions_taken: int) -> int | None:
        if not self.portfolio_enabled:
            return super().select(node, actions_taken)
        winning = self.winning_actions.get(node.key)
        if winning is not None and node.valid[winning]:
            self.route_counts["known_win"] += 1
            self._pending_lambda = int(winning)
            return int(winning)
        proposal = self.portfolio.propose(node)
        if proposal is None:
            return super().select(node, actions_taken)
        selected = int(proposal.lam)
        self._pending_lambda = selected
        return selected

    def _update_priors(self, cur: Any, lam: int, nxt: Any, changed: bool):
        super()._update_priors(cur, lam, nxt, changed)
        self.portfolio.observe(cur, int(lam), nxt, bool(changed), cur.level != nxt.level)

    def run_env(self, env: Any, budgets: dict[int, int]):
        result = super().run_env(env, budgets)
        result.telemetry = dict(result.telemetry)
        result.telemetry["planner_portfolio"] = {
            "enabled": self.portfolio_enabled,
            **self.portfolio.telemetry(),
        }
        return result
