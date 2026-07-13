"""Submission solver switches used by cold-versus-distilled ablations."""
from __future__ import annotations

from collections import deque

import numpy as np

from ..autodiff import T
from ..planning.portfolio import PlannerPortfolio
from .losses import double_dqn_targets, q_loss
from .submission_solver import SubmissionDRLQNSolver


class AblationSubmissionDRLQNSolver(SubmissionDRLQNSolver):
    """Expose history, graph, planner-portfolio and differential switches.

    The production solver remains unchanged. When differential complexity is
    disabled, online learning uses only the Double-DQN router loss and never
    activates the JVP runtime or executor/world-model auxiliary losses.
    """

    def __init__(
        self,
        *args,
        history_enabled: bool = True,
        history_len: int = 4,
        graph_planning: bool = True,
        planner_portfolio: bool = False,
        differential_complexity: bool = True,
        **kwargs,
    ) -> None:
        requested_history = int(history_len) if history_enabled else 0
        super().__init__(*args, history_len=max(1, requested_history), **kwargs)
        self.history_enabled = bool(history_enabled)
        self.history_len = max(0, requested_history)
        self._recent_effects = deque(maxlen=self.history_len)
        self.graph_planning = bool(graph_planning)
        self.planner_portfolio_enabled = bool(planner_portfolio)
        self.planner_portfolio = PlannerPortfolio()
        self.route_counts.setdefault("portfolio", 0)
        self.differential_complexity = bool(differential_complexity)
        if not self.differential_complexity:
            self.sched.diff_losses_at = 10**12
            self.sched.approx_at = 10**12

    def select(self, node, actions_taken: int) -> int | None:
        if self.planner_portfolio_enabled:
            selected = self.planner_portfolio.select(self, node)
            if selected is not None:
                self.route_counts["portfolio"] += 1
                self._pending_lambda = int(selected)
                return int(selected)

        if self.graph_planning:
            return super().select(node, actions_taken)

        untried = self._untried(node)
        if not untried:
            self._pending_lambda = None
            return None
        epsilon = max(0.05, 0.28 * np.exp(-actions_taken / 350.0))
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

    def update(self):
        if self.differential_complexity:
            return super().update()

        batch = self.replay.sample(self.sched.batch)
        size = len(batch)
        cfg = self.cfg

        def gather(keys):
            tokens = np.stack([self.nodes[key].tokens for key in keys])
            masks = np.stack([self.nodes[key].token_mask for key in keys])
            traces = np.stack([self.nodes[key].trace for key in keys])
            valid = np.stack([self.nodes[key].valid for key in keys])
            features = np.stack([self.nodes[key].lam_feats for key in keys])
            return tokens, masks, traces, valid, features

        state_keys = [transition.key for transition in batch]
        next_keys = [transition.next_key for transition in batch]
        tokens, masks, traces, valid, features = gather(state_keys)
        next_tokens, next_masks, next_traces, next_valid, next_features = gather(next_keys)
        lambdas = np.asarray([transition.lam for transition in batch], dtype=np.int64)
        rewards = np.asarray([transition.reward for transition in batch], dtype=np.float64)
        dones = np.asarray([float(transition.done) for transition in batch], dtype=np.float64)

        targets = double_dqn_targets(
            self.model,
            self.target,
            (next_tokens, next_masks, next_traces),
            next_valid,
            rewards,
            dones,
            self.gamma,
            next_features,
        )
        route_context = T(np.zeros((size, cfg.d_route), dtype=np.float64))
        latent = self.model.encode(tokens, masks, traces)
        total = q_loss(
            self.model,
            latent,
            route_context,
            valid,
            lambdas,
            targets,
            features,
        )
        self.opt.zero_grad()
        total.backward()
        self.opt.step()
        self.updates += 1
        if self.updates % self.sched.target_sync == 0:
            self.target.load_state_dict(self.model.state_dict())

    def run_env(self, env, budgets):
        result = super().run_env(env, budgets)
        result.telemetry = dict(result.telemetry)
        result.telemetry["ablation"] = {
            "history": self.history_enabled,
            "history_len": self.history_len,
            "graph_planning": self.graph_planning,
            "planner_portfolio": self.planner_portfolio_enabled,
            "differential_complexity": self.differential_complexity,
        }
        if self.planner_portfolio_enabled:
            result.telemetry["planner_portfolio"] = self.planner_portfolio.telemetry()
        return result
