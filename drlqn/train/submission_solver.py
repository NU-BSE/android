"""Submission-facing top-route solver.

The exploratory ``TopRouteDRLQNSolver`` keeps detailed trace counters. For graph
identity, however, absolute step count must not be hashed: doing so would make
every visit unique and disable deterministic route reuse. This submission class
keys only the rendered state, level, and bounded recent action/effect context;
step count remains available to the neural encoder through trace features.
"""
from __future__ import annotations

import hashlib
import time

import numpy as np

from ..operators.library import to_env_action
from ..runtime.shadow import shadow_report
from .replay import Transition
from .top_solver import TopRouteDRLQNSolver
from .trainer import SolveResult


class SubmissionDRLQNSolver(TopRouteDRLQNSolver):
    """Top-route DRLQN with bounded history and visual-effect learning targets."""

    def _context_key(self, grid: np.ndarray, level: int) -> str:
        payload = bytearray()
        for lam, changed, advanced in self._recent_effects:
            payload.extend(int(lam).to_bytes(2, "little", signed=False))
            payload.append(int(bool(changed)))
            payload.append(int(bool(advanced)))
        return hashlib.sha1(
            np.asarray(grid, dtype=np.uint8).tobytes()
            + int(level).to_bytes(2, "little", signed=False)
            + bytes(payload)
        ).hexdigest()[:16]

    def run_env(self, env, budgets: dict[int, int]) -> SolveResult:
        """Run with graph-state and visual-state changes kept distinct.

        A no-op action can legitimately create a new history-conditioned graph
        node. It must still receive the no-change penalty and must not become a
        positive world-model transition merely because its context key changed.
        """
        started = time.time()
        result = SolveResult(env_id=env.env_id)
        frame = env.reset()
        current = self.perceive(env, frame.grid)
        previous_delta = None

        while env.state == "NOT_FINISHED":
            level = env.level
            used = env._level_action_counts.get(level, 0)
            if used >= budgets.get(level, 10 ** 9):
                break

            lam = self.select(current, used)
            if lam is None:
                frame = env.reset_level()
                current = self.perceive(env, frame.grid)
                previous_delta = None
                continue

            action, x, y = to_env_action(lam, current.objs)
            previous_grid = current.grid.copy()
            step = env.step(action, x, y)
            next_node = self.perceive(env, step.frame.grid)

            visual_changed = not np.array_equal(previous_grid, step.frame.grid)
            context_changed = next_node.key != current.key
            is_new = (
                next_node.key not in self.graph
                and context_changed
                and visual_changed
                and not step.level_won
            )
            reward = (
                -0.02
                + (20.0 if step.env_won else 0.0)
                + (10.0 if step.level_won and not step.env_won else 0.0)
                + (0.5 if is_new else 0.0)
                + (-0.3 if not visual_changed else 0.0)
            )
            if visual_changed and not step.level_won and current.level == next_node.level:
                reward += 0.6 * (self._potential(next_node) - self._potential(current))

            self.graph.setdefault(current.key, {})[lam] = next_node.key
            self._update_priors(current, lam, next_node, visual_changed)
            self.replay.push(
                Transition(
                    current.key,
                    lam,
                    reward,
                    next_node.key,
                    step.env_won,
                    stable_pair=visual_changed and not step.level_won,
                    changed=visual_changed,
                )
            )

            if step.level_won:
                previous_delta = None
            elif self.updates >= self.sched.approx_at and visual_changed:
                current_cache = self.latent(current)
                delta_input = (
                    previous_delta
                    if previous_delta is not None
                    else np.zeros_like(current_cache.Z)
                )

                def exact_fn():
                    cache = self.runtime.exact_encode(
                        next_node.tokens,
                        next_node.token_mask,
                        next_node.trace,
                    )
                    self.caches[next_node.key] = (self._cache_version(), cache)
                    return cache

                effective_lambda = self.model.q_router.effective_embedding_np(
                    lam,
                    current.lam_feats[lam],
                )
                next_cache, approximated = self.runtime.apply_delta(
                    current_cache,
                    lam,
                    delta_input,
                    np.zeros(self.cfg.d_route),
                    effective_lambda,
                    exact_fn,
                )
                self.caches[next_node.key] = (self._cache_version(), next_cache)
                previous_delta = (
                    next_cache.Z - current_cache.Z
                    if approximated
                    else None
                )
                if next_cache.margin is not None:
                    report = shadow_report(
                        next_cache.margin,
                        np.zeros_like(next_cache.margin),
                    )
                    next_cache.shadow_weights = report[0]["p"]
                if next_cache.active_indices is not None:
                    self.column_usage[next_cache.active_indices] += 1
                support = float(
                    np.mean(
                        np.abs(next_cache.Z - current_cache.Z).sum(-1) > 1e-6
                    )
                )
                self.delta_supports.append(support)
            else:
                previous_delta = None

            if (
                len(self.replay) >= self.sched.warmup
                and env.action_count % self.sched.train_every == 0
            ):
                self.update()

            current = next_node
            if step.level_won:
                result.completed = level
            if step.env_won:
                result.won = True
                break

        result.actions_per_level = env.actions_per_level
        result.total_actions = env.action_count
        result.updates = self.updates
        result.telemetry = self.runtime.tel.summary()
        result.telemetry["top_route"] = {
            **self.route_counts,
            "context_nodes": len(self.nodes),
            "winning_edges": len(self.winning_actions),
            "effect_edges": len(self.edge_stats),
            "history_len": self.history_len,
        }
        result.wall_clock_s = time.time() - started
        result.column_usage = self.column_usage.tolist()
        result.q_margin_mean = (
            float(np.mean(self.q_margins)) if self.q_margins else 0.0
        )
        result.delta_support_mean = (
            float(np.mean(self.delta_supports)) if self.delta_supports else 0.0
        )
        return result
