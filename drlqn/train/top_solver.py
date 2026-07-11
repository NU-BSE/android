"""History-aware, planner-biased DRLQN for the ARC-AGI-3 submission route.

This solver keeps the fixed DRLQN tensor geometry, but closes three practical
performance gaps exposed by the official environments and Duck Harness:

* state keys include a short action/effect history to reduce visual-state aliasing;
* trace features are live (visits, last action, level steps, novelty), not constants;
* known winning edges and high-value frontiers are routed to before blind expansion.

The state encoder still receives object tokens, while ACTION6 uses separate dynamic
spatial anchors. This avoids the hybrid v1 mistake of replacing semantic objects
with click proposals in relational priors.
"""
from __future__ import annotations

from collections import deque
import hashlib

import numpy as np

from ..operators.library import (
    arg_features,
    click_slot_offset,
    is_click_lambda,
    spatial_anchors,
    to_env_action,
    valid_mask,
)
from ..state import (
    MAX_TOKENS,
    TOKEN_DIM,
    TraceSummary,
    extract_objects,
    tokenize,
)
from .hybrid_solver import _as_obj
from .trainer import DRLQNSolver, NodeInfo


class TopRouteDRLQNSolver(DRLQNSolver):
    """DRLQN with short-history memory and deterministic graph planning.

    The recent history is intentionally compact: four ``(lambda, changed,
    advanced)`` events. Duck retains a full transcript; this bounded summary is
    the low-cost student analogue and prevents many hidden-mode states from
    collapsing solely because their rendered grids are identical.
    """

    def __init__(self, *args, history_len: int = 4, **kwargs):
        super().__init__(*args, **kwargs)
        self.history_len = max(1, int(history_len))
        self._recent_effects: deque[tuple[int, bool, bool]] = deque(
            maxlen=self.history_len
        )
        self._pending_lambda: int | None = None
        self._previous_grid: np.ndarray | None = None
        self._context_level: int | None = None
        self._last_env_action_count = 0
        self._steps_in_level = 0
        self._visual_seen: set[str] = set()
        self._visits: dict[str, int] = {}

        self.winning_actions: dict[str, int] = {}
        self.edge_stats: dict[tuple[str, int], list[float]] = {}
        self.route_counts = {
            "known_win": 0,
            "frontier": 0,
            "q_expand": 0,
            "random_expand": 0,
        }

    # ---------------------------------------------------------- history state
    def _reset_history(self, level: int, grid: np.ndarray | None = None) -> None:
        self._recent_effects.clear()
        self._pending_lambda = None
        self._context_level = int(level)
        self._steps_in_level = 0
        self._visual_seen.clear()
        self._previous_grid = None if grid is None else np.asarray(grid).copy()

    def _context_key(self, grid: np.ndarray, level: int) -> str:
        payload = bytearray()
        for lam, changed, advanced in self._recent_effects:
            payload.extend(int(lam).to_bytes(2, "little", signed=False))
            payload.append(int(bool(changed)))
            payload.append(int(bool(advanced)))
        payload.extend(int(self._steps_in_level).to_bytes(2, "little", signed=False))
        return hashlib.sha1(
            np.asarray(grid, dtype=np.uint8).tobytes()
            + int(level).to_bytes(2, "little", signed=False)
            + bytes(payload)
        ).hexdigest()[:16]

    @staticmethod
    def _visual_key(grid: np.ndarray, level: int) -> str:
        return hashlib.sha1(
            np.asarray(grid, dtype=np.uint8).tobytes()
            + int(level).to_bytes(2, "little", signed=False)
        ).hexdigest()[:16]

    def _ingest_pending_transition(self, env, grid: np.ndarray) -> int:
        """Update bounded history before keying the newly observed state."""
        level = int(env.level)
        action_count = int(getattr(env, "action_count", self._last_env_action_count))
        last_lam = self._recent_effects[-1][0] if self._recent_effects else -1

        if self._context_level is None:
            self._context_level = level
        elif level != self._context_level:
            # A level transition re-grounds the state, as in Duck's prompt loop.
            self._reset_history(level)
        elif self._pending_lambda is None and action_count > self._last_env_action_count:
            # The parent solver performs RESET without a lambda when a reachable
            # graph is exhausted. Do not leak pre-reset hidden context forward.
            self._reset_history(level)

        if self._pending_lambda is not None and self._previous_grid is not None:
            changed = not np.array_equal(self._previous_grid, grid)
            advanced = level != self._context_level
            self._recent_effects.append(
                (int(self._pending_lambda), bool(changed), bool(advanced))
            )
            last_lam = int(self._pending_lambda)
            self._steps_in_level += 1

        self._pending_lambda = None
        self._context_level = level
        self._last_env_action_count = action_count
        return last_lam

    # ------------------------------------------------------------- perception
    def perceive(self, env, grid) -> NodeInfo:
        grid = np.asarray(grid, dtype=np.uint8)
        previous = None if self._previous_grid is None else self._previous_grid.copy()
        last_lam = self._ingest_pending_transition(env, grid)

        key = self._context_key(grid, env.level)
        visual_key = self._visual_key(grid, env.level)
        self._visual_seen.add(visual_key)
        self._visits[key] = self._visits.get(key, 0) + 1
        novelty = min(1.0, len(self._visual_seen) / max(1, self._steps_in_level + 1))
        trace = TraceSummary(
            visit_count=self._visits[key],
            last_lambda=last_lam,
            steps_in_level=self._steps_in_level,
            novelty_rate=novelty,
        ).features()

        raw_objects = extract_objects(grid)
        proposals = spatial_anchors(grid, raw_objects, previous)
        anchors = [_as_obj(proposal, grid) for proposal in proposals]
        features = tokenize(raw_objects)
        n_tokens = min(features.shape[0], MAX_TOKENS)
        tokens = np.zeros((MAX_TOKENS, TOKEN_DIM))
        tokens[:n_tokens] = features[:n_tokens]
        token_mask = np.zeros(MAX_TOKENS, dtype=bool)
        token_mask[:n_tokens] = True

        node = self.nodes.get(key)
        if node is None:
            node = NodeInfo(
                key=key,
                tokens=tokens,
                token_mask=token_mask,
                trace=trace,
                objs=anchors,
                valid=valid_mask(env, anchors),
                level=int(env.level),
                lam_feats=arg_features(anchors),
                grid=grid.copy(),
            )
            self.nodes[key] = node
        else:
            # Dynamic trace and changed-region anchors are part of the decision
            # state even when the rendered frame repeats.
            node.tokens = tokens
            node.token_mask = token_mask
            node.trace = trace
            node.objs = anchors
            node.valid = valid_mask(env, anchors)
            node.lam_feats = arg_features(anchors)
            node.grid = grid.copy()
            self.caches.pop(key, None)

        node.raw_objects = raw_objects
        node.proposals = proposals
        node.proposal_kinds = [proposal.kind for proposal in proposals]
        node.hybrid_spatial = True
        self._previous_grid = grid.copy()
        return node

    # -------------------------------------------------------- semantic helpers
    @staticmethod
    def _raw_objects(node: NodeInfo):
        return getattr(node, "raw_objects", node.objs)

    def _afford_key(self, node: NodeInfo, lam: int):
        if not is_click_lambda(lam):
            return ("key", int(lam), 0)
        slot, offset = click_slot_offset(lam)
        if slot >= len(node.objs):
            return None
        anchor = node.objs[slot]
        return (anchor.color, offset, self._size_bucket(anchor.area))

    def _potential(self, node: NodeInfo) -> float:
        objects = self._raw_objects(node)
        self_color = self._self_color()
        salient = [
            obj
            for obj in objects
            if obj.area < 400
            and obj.color != self_color
            and sum(1 for other in objects if other.color == obj.color) == 1
        ]
        best = None
        for i, left in enumerate(salient):
            for right in salient[i + 1 :]:
                if left.color == right.color:
                    continue
                distance = abs(left.cy - right.cy) + abs(left.cx - right.cx)
                best = distance if best is None else min(best, distance)
        return -(best or 0.0) / 64.0

    def _click_group(self, node: NodeInfo, lam: int):
        if not is_click_lambda(lam):
            return ("key", int(lam))
        _, x, y = to_env_action(lam, node.objs)
        color = int(node.grid[y, x])
        best = None
        for obj in self._raw_objects(node):
            if (
                obj.area < 200
                and obj.color == color
                and obj.y0 <= y <= obj.y1
                and obj.x0 <= x <= obj.x1
            ):
                if best is None or obj.area < best.area:
                    best = obj
        if best is not None:
            return ("obj", best.color, best.y0, best.x0)
        return ("px", x, y)

    def _prior_bonus(self, node: NodeInfo, lam: int) -> float:
        bonus = 0.0
        key = self._afford_key(node, lam)
        if key is not None:
            tried, changed = self.afford.get(key, [0, 0])
            bonus += 1.5 * (changed + 1) / (tried + 2)

        if is_click_lambda(lam):
            slot, offset = click_slot_offset(lam)
            if slot >= len(node.objs):
                return -10.0
            anchor = node.objs[slot]
            _, x, y = to_env_action(lam, node.objs)
            bonus += 0.08 * float(getattr(anchor, "proposal_score", 0.0))
            kind = str(getattr(anchor, "kind", ""))
            if kind == "changed":
                bonus += 1.4
            elif kind == "tile":
                bonus += 0.45
            elif kind == "lattice":
                bonus -= 0.1
            if offset == 0:
                bonus += 0.25

            objects = self._raw_objects(node)
            special = [
                obj
                for obj in objects
                if obj.area < 400
                and sum(1 for other in objects if other.color == obj.color) == 1
            ]
            if special:
                near = min(abs(obj.cy - y) + abs(obj.cx - x) for obj in special)
                bonus += 0.8 * max(0.0, 1.0 - near / 24.0)
            if self.last_change is not None:
                distance = abs(self.last_change[0] - y) + abs(self.last_change[1] - x)
                bonus += 1.0 * max(0.0, 1.0 - distance / 12.0)

        if 0 <= int(lam) < 4:
            self_color = self._self_color()
            direction = self.move_dir.get(int(lam))
            objects = self._raw_objects(node)
            if self_color is not None and direction is not None and np.abs(direction).sum() > 0.3:
                actor = next((obj for obj in objects if obj.color == self_color), None)
                targets = [
                    obj
                    for obj in objects
                    if obj.color != self_color
                    and obj.area < 400
                    and sum(1 for other in objects if other.color == obj.color) <= 2
                ]
                if actor is not None and targets:
                    target = min(
                        targets,
                        key=lambda obj: abs(obj.cy - actor.cy) + abs(obj.cx - actor.cx),
                    )
                    vector = np.array([target.cy - actor.cy, target.cx - actor.cx])
                    nv, nd = np.linalg.norm(vector), np.linalg.norm(direction)
                    if nv > 1e-6 and nd > 1e-6:
                        bonus += 0.8 * float(vector @ direction) / (nv * nd)

        stats = self.edge_stats.get((node.key, int(lam)))
        if stats is not None and stats[0] > 0:
            bonus += 0.5 * stats[1] / stats[0]
            bonus += 3.0 * stats[2] / stats[0]
        return bonus

    def _update_priors(self, cur: NodeInfo, lam: int, nxt: NodeInfo, changed: bool):
        key = self._afford_key(cur, lam)
        if key is not None:
            tally = self.afford.setdefault(key, [0, 0])
            tally[0] += 1
            tally[1] += int(changed)

        advanced = cur.level != nxt.level
        stats = self.edge_stats.setdefault((cur.key, int(lam)), [0.0, 0.0, 0.0])
        stats[0] += 1.0
        stats[1] += float(changed)
        stats[2] += float(advanced)
        if advanced:
            self.winning_actions[cur.key] = int(lam)

        if changed and not advanced:
            if is_click_lambda(lam):
                _, x, y = to_env_action(lam, cur.objs)
                self.last_change = (float(y), float(x))
            else:
                ys, xs = np.where(cur.grid != nxt.grid)
                if len(ys):
                    self.last_change = (float(ys.mean()), float(xs.mean()))
        elif advanced:
            self.last_change = None

        if 0 <= int(lam) < 4 and changed:
            current_by_color: dict[int, list] = {}
            current_objects = self._raw_objects(cur)
            next_objects = self._raw_objects(nxt)
            for obj in current_objects:
                current_by_color.setdefault(obj.color, []).append(obj)
            for next_obj in next_objects:
                group = current_by_color.get(next_obj.color, [])
                if len(group) != 1 or sum(
                    1 for obj in next_objects if obj.color == next_obj.color
                ) != 1:
                    continue
                current_obj = group[0]
                if abs(current_obj.area - next_obj.area) > 0.2 * max(current_obj.area, 1):
                    continue
                delta = np.array(
                    [next_obj.cy - current_obj.cy, next_obj.cx - current_obj.cx]
                )
                if 0.5 < np.abs(delta).sum() < 16:
                    self.self_votes[next_obj.color] = self.self_votes.get(next_obj.color, 0) + 1
                    if (
                        self.self_votes[next_obj.color] >= 2
                        and self.self_votes[next_obj.color] == max(self.self_votes.values())
                    ):
                        accumulator = self.move_dir.setdefault(int(lam), np.zeros(2))
                        self.move_dir[int(lam)] = 0.7 * accumulator + 0.3 * delta

    # --------------------------------------------------------------- planning
    def _untried(self, node: NodeInfo) -> list[int]:
        tried = self.graph.get(node.key, {})
        seen_groups = {
            self._click_group(node, int(lam))
            for lam in tried
            if is_click_lambda(int(lam))
        }
        result: list[int] = []
        for raw_lam in np.where(node.valid)[0]:
            lam = int(raw_lam)
            if lam in tried:
                continue
            if is_click_lambda(lam):
                group = self._click_group(node, lam)
                if group in seen_groups:
                    continue
                seen_groups.add(group)
            result.append(lam)
        return result

    def _route_to_frontier(self, node: NodeInfo) -> int | None:
        queue = deque([(node.key, 0)])
        first: dict[str, int | None] = {node.key: None}
        best: tuple[float, int] | None = None

        while queue:
            key, distance = queue.popleft()
            info = self.nodes[key]
            first_action = first[key]

            winning = self.winning_actions.get(key)
            if winning is not None and info.valid[winning]:
                action = winning if key == node.key else first_action
                if action is not None:
                    self.route_counts["known_win"] += 1
                    return int(action)

            if key != node.key:
                candidates = self._untried(info)
                if candidates:
                    q_values = self.q_of(info)
                    frontier_value = max(
                        q_values[lam] + self._prior_bonus(info, lam)
                        for lam in candidates
                    )
                    score = (
                        float(frontier_value)
                        + 0.15 * min(len(candidates), 12)
                        - 0.12 * distance
                    )
                    if first_action is not None and (best is None or score > best[0]):
                        best = (score, int(first_action))

            for lam, next_key in self.graph.get(key, {}).items():
                if next_key in first or next_key == key or next_key not in self.nodes:
                    continue
                if self.nodes[next_key].level != node.level:
                    continue
                first[next_key] = first_action if first_action is not None else int(lam)
                queue.append((next_key, distance + 1))

        if best is not None:
            self.route_counts["frontier"] += 1
            return best[1]
        return None

    def select(self, node: NodeInfo, actions_taken: int) -> int | None:
        winning = self.winning_actions.get(node.key)
        if winning is not None and node.valid[winning]:
            selected = int(winning)
            self.route_counts["known_win"] += 1
            self._pending_lambda = selected
            return selected

        untried = self._untried(node)
        epsilon = max(0.05, 0.28 * np.exp(-actions_taken / 350.0))
        if untried:
            if self.rng.random() < epsilon:
                selected = int(self.rng.choice(untried))
                self.route_counts["random_expand"] += 1
            else:
                q_values = self.q_of(node)
                prior_weight = max(0.25, 1.4 * np.exp(-self.updates / 300.0))
                selected = int(
                    max(
                        untried,
                        key=lambda lam: q_values[lam]
                        + prior_weight * self._prior_bonus(node, lam),
                    )
                )
                self.route_counts["q_expand"] += 1
            self._pending_lambda = selected
            return selected

        selected = self._route_to_frontier(node)
        self._pending_lambda = selected
        return selected

    def run_env(self, env, budgets: dict[int, int]):
        result = super().run_env(env, budgets)
        result.telemetry = dict(result.telemetry)
        result.telemetry["top_route"] = {
            **self.route_counts,
            "context_nodes": len(self.nodes),
            "winning_edges": len(self.winning_actions),
            "effect_edges": len(self.edge_stats),
            "history_len": self.history_len,
        }
        return result
