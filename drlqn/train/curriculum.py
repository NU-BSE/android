"""Multi-stage Duck, environment-model and planner/value distillation curriculum."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..autodiff import Adam, T
from .dense_sft import AuxiliaryHeads, PreparedDenseSFT, _bce, _weighted


@dataclass(frozen=True)
class CurriculumStageMetrics:
    loss: float
    action_loss: float = 0.0
    frame_change_loss: float = 0.0
    level_advance_loss: float = 0.0
    next_embedding_loss: float = 0.0
    planner_value_loss: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return self.__dict__.copy()


def planner_value_target(item: PreparedDenseSFT, gamma: float = 0.97) -> float:
    """Return a dense generic value target including negative evidence."""
    value = 0.0
    if item.level_advanced == 1:
        value += 1.0
    if item.success == 1:
        value += max(0.1, gamma)
    if item.visual_changed == 0:
        value -= 0.25
    trajectory = item.trajectory_id.lower()
    if "random" in trajectory:
        value *= 0.5
    return float(np.clip(value, -1.0, 2.0))


class EnvironmentTransitionTrainer:
    """Train effect and next-state representations without cloning random actions."""

    def __init__(
        self,
        solver,
        *,
        seed: int = 0,
        lr: float = 3e-3,
        frame_change_coef: float = 0.6,
        level_advance_coef: float = 0.8,
        next_embedding_coef: float = 0.5,
    ) -> None:
        self.solver = solver
        self.heads = AuxiliaryHeads(solver.cfg, np.random.default_rng(seed + 311))
        self.opt = Adam([*solver.model.parameters(), *self.heads.parameters()], lr=lr)
        self.frame_change_coef = float(frame_change_coef)
        self.level_advance_coef = float(level_advance_coef)
        self.next_embedding_coef = float(next_embedding_coef)

    def train_batch(self, batch: Sequence[PreparedDenseSFT]) -> dict[str, float]:
        if not batch:
            raise ValueError("transition batch must not be empty")
        model = self.solver.model
        tokens = np.stack([item.tokens for item in batch])
        masks = np.stack([item.token_mask for item in batch])
        traces = np.stack([item.trace for item in batch])
        labels = np.asarray([item.label for item in batch], dtype=np.int64)
        lam_features = np.stack([item.lam_features for item in batch])
        weights = np.asarray([item.weight for item in batch], dtype=np.float64)

        latent = model.encode(tokens, masks, traces)
        selected_features = lam_features[np.arange(len(batch)), labels]
        action_embedding = (
            model.q_router.base_embeddings().gather_rows(labels)
            + model.q_router.obj_proj(T(selected_features))
        )
        auxiliary_input = T.concat([latent.mean(axis=1), action_embedding], axis=-1)
        change_logits = self.heads.frame_change(auxiliary_input).reshape(len(batch))
        advance_logits = self.heads.level_advance(auxiliary_input).reshape(len(batch))
        predicted_next = self.heads.next_embedding(auxiliary_input)

        changed = np.asarray([max(0, item.visual_changed) for item in batch], dtype=np.float64)
        changed_mask = np.asarray([item.visual_changed >= 0 for item in batch], dtype=np.float64)
        advanced = np.asarray([max(0, item.level_advanced) for item in batch], dtype=np.float64)
        advanced_mask = np.asarray([item.level_advanced >= 0 for item in batch], dtype=np.float64)
        has_next = np.asarray([item.has_next for item in batch], dtype=np.float64)
        change_loss = _weighted(_bce(change_logits, changed), weights, changed_mask)
        advance_loss = _weighted(_bce(advance_logits, advanced), weights, advanced_mask)

        next_tokens = np.stack([item.next_tokens for item in batch])
        next_masks = np.stack([item.next_token_mask for item in batch])
        next_latent = model.encode(next_tokens, next_masks, traces)
        next_target = T(next_latent.data.mean(axis=1))
        next_losses = ((predicted_next - next_target) ** 2).mean(axis=1)
        next_loss = _weighted(next_losses, weights, has_next)
        total = (
            self.frame_change_coef * change_loss
            + self.level_advance_coef * advance_loss
            + self.next_embedding_coef * next_loss
        )
        self.opt.zero_grad()
        total.backward()
        self.opt.step()
        self.solver.updates += 1
        self.solver.caches.clear()
        return CurriculumStageMetrics(
            loss=float(total.data),
            frame_change_loss=float(change_loss.data),
            level_advance_loss=float(advance_loss.data),
            next_embedding_loss=float(next_loss.data),
        ).as_dict()


class PlannerValueDistiller:
    """Distill specialist choices and progress targets into persistent Q logits."""

    def __init__(
        self,
        solver,
        *,
        lr: float = 2e-3,
        imitation_coef: float = 0.35,
        value_coef: float = 1.0,
    ) -> None:
        self.solver = solver
        self.opt = Adam(solver.model.parameters(), lr=lr)
        self.imitation_coef = float(imitation_coef)
        self.value_coef = float(value_coef)

    def train_batch(self, batch: Sequence[PreparedDenseSFT]) -> dict[str, float]:
        if not batch:
            raise ValueError("planner batch must not be empty")
        model = self.solver.model
        cfg = self.solver.cfg
        tokens = np.stack([item.tokens for item in batch])
        masks = np.stack([item.token_mask for item in batch])
        traces = np.stack([item.trace for item in batch])
        valid = np.stack([item.valid for item in batch])
        lam_features = np.stack([item.lam_features for item in batch])
        labels = np.asarray([item.label for item in batch], dtype=np.int64)
        weights = np.asarray([item.weight for item in batch], dtype=np.float64)
        planner_mask = np.asarray(
            [0.0 if "random" in item.trajectory_id.lower() else 1.0 for item in batch],
            dtype=np.float64,
        )

        latent = model.encode(tokens, masks, traces)
        route_context = T(np.zeros((len(batch), cfg.d_route), dtype=np.float64))
        q_values = model.q_values(latent, route_context, valid, lam_features)
        selected_q = q_values[np.arange(len(batch)), labels]

        shift = np.max(q_values.data, axis=1)
        log_partition = (
            (q_values - T(shift[:, None])).exp().sum(axis=1) + 1e-12
        ).log() + T(shift)
        imitation_losses = log_partition - selected_q
        imitation_loss = _weighted(imitation_losses, weights, planner_mask)

        targets = np.asarray([planner_value_target(item) for item in batch], dtype=np.float64)
        value_losses = (selected_q - T(targets)) ** 2
        value_loss = _weighted(value_losses, weights)
        total = self.imitation_coef * imitation_loss + self.value_coef * value_loss
        self.opt.zero_grad()
        total.backward()
        self.opt.step()
        self.solver.updates += 1
        self.solver.caches.clear()
        return CurriculumStageMetrics(
            loss=float(total.data),
            action_loss=float(imitation_loss.data),
            planner_value_loss=float(value_loss.data),
        ).as_dict()
