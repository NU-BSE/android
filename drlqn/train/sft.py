"""Supervised distillation from conservative Duck state/action examples."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from ..autodiff import T
from ..operators.library import arg_features, spatial_anchors, valid_mask
from ..state import MAX_TOKENS, TOKEN_DIM, TraceSummary, extract_objects, tokenize
from .distill import match_teacher_action
from .duck_dataset import DuckExample
from .hybrid_solver import _as_obj


class _TeacherActionSpace:
    action_space = tuple(range(1, 10))
    uses_click = True


@dataclass
class PreparedSFT:
    tokens: np.ndarray
    token_mask: np.ndarray
    trace: np.ndarray
    valid: np.ndarray
    lam_features: np.ndarray
    label: int
    weight: float
    game_id: str


def prepare_example(example: DuckExample, max_click_distance: float = 12.0) -> PreparedSFT | None:
    grid = np.asarray(example.grid, dtype=np.uint8)
    objects = extract_objects(grid)
    proposals = spatial_anchors(grid, objects, None)
    anchors = [_as_obj(proposal, grid) for proposal in proposals]
    try:
        match = match_teacher_action(
            example.action,
            anchors,
            x=example.x,
            y=example.y,
        )
    except ValueError:
        return None
    if example.action == 6 and match.distance > max_click_distance:
        return None

    object_features = tokenize(objects)
    n_tokens = min(len(object_features), MAX_TOKENS)
    tokens = np.zeros((MAX_TOKENS, TOKEN_DIM), dtype=np.float64)
    tokens[:n_tokens] = object_features[:n_tokens]
    token_mask = np.zeros(MAX_TOKENS, dtype=bool)
    token_mask[:n_tokens] = True
    valid = valid_mask(_TeacherActionSpace(), anchors)
    if not valid[match.lambda_id]:
        return None
    distance_weight = max(0.05, 1.0 - match.distance / max(max_click_distance, 1e-6))
    return PreparedSFT(
        tokens=tokens,
        token_mask=token_mask,
        trace=TraceSummary().features(),
        valid=valid,
        lam_features=arg_features(anchors),
        label=int(match.lambda_id),
        weight=float(distance_weight),
        game_id=example.game_id,
    )


def prepare_examples(examples: Iterable[DuckExample], max_click_distance: float = 12.0) -> list[PreparedSFT]:
    prepared = []
    for example in examples:
        item = prepare_example(example, max_click_distance=max_click_distance)
        if item is not None:
            prepared.append(item)
    return prepared


class DuckSFTTrainer:
    """Train the DRLQN encoder/Q router with weighted behavioral cloning."""

    def __init__(self, solver):
        self.solver = solver

    def train_batch(self, batch: list[PreparedSFT]) -> dict[str, float]:
        if not batch:
            raise ValueError("SFT batch must not be empty")
        model = self.solver.model
        cfg = self.solver.cfg
        tokens = np.stack([item.tokens for item in batch])
        masks = np.stack([item.token_mask for item in batch])
        traces = np.stack([item.trace for item in batch])
        valid = np.stack([item.valid for item in batch])
        lam_features = np.stack([item.lam_features for item in batch])
        labels = np.asarray([item.label for item in batch], dtype=np.int64)
        weights = np.asarray([item.weight for item in batch], dtype=np.float64)

        latent = model.encode(tokens, masks, traces)
        route_context = T(np.zeros((len(batch), cfg.d_route), dtype=np.float64))
        q_values = model.q_values(latent, route_context, valid, lam_features)
        teacher_q = q_values[np.arange(len(batch)), labels]
        shift_np = np.max(q_values.data, axis=1)
        log_partition = (
            (q_values - T(shift_np[:, None])).exp().sum(axis=1) + 1e-12
        ).log() + T(shift_np)
        per_example = log_partition - teacher_q
        loss = (per_example * T(weights)).sum() / float(max(weights.sum(), 1e-8))

        self.solver.opt.zero_grad()
        loss.backward()
        self.solver.opt.step()
        self.solver.updates += 1
        self.solver.caches.clear()

        predictions = np.argmax(q_values.data, axis=1)
        accuracy = float(np.average(predictions == labels, weights=weights))
        return {
            "loss": float(loss.data),
            "accuracy": accuracy,
            "mean_weight": float(weights.mean()),
        }
