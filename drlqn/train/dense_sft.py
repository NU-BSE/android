"""Dense Duck distillation with grouped evaluation and auxiliary supervision."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from ..autodiff import Adam, MLP, Module, T
from ..operators.library import (
    K,
    arg_features,
    is_click_lambda,
    spatial_anchors,
    to_env_action,
    valid_mask,
)
from ..state import MAX_TOKENS, TOKEN_DIM, TraceSummary, extract_objects, tokenize
from .distill import match_teacher_action
from .duck_dataset import DuckExample
from .hybrid_solver import _as_obj


class _TeacherActionSpace:
    action_space = tuple(range(1, 10))
    uses_click = True


@dataclass(frozen=True)
class PreparedDenseSFT:
    tokens: np.ndarray
    token_mask: np.ndarray
    trace: np.ndarray
    valid: np.ndarray
    lam_features: np.ndarray
    next_tokens: np.ndarray
    next_token_mask: np.ndarray
    label: int
    weight: float
    game_id: str
    seed: int
    trajectory_id: str
    raw_action: int
    teacher_x: int
    teacher_y: int
    lambda_xy: np.ndarray
    proposal_distance: float
    visual_changed: int
    level_advanced: int
    has_next: bool
    success: int


@dataclass(frozen=True)
class DatasetSplit:
    train: tuple[PreparedDenseSFT, ...]
    validation: tuple[PreparedDenseSFT, ...]
    held_out: tuple[PreparedDenseSFT, ...]
    held_out_games: tuple[str, ...]
    train_groups: tuple[tuple[str, int], ...]
    validation_groups: tuple[tuple[str, int], ...]


def _tokenize_grid(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    features = tokenize(extract_objects(np.asarray(grid, dtype=np.uint8)))
    count = min(features.shape[0], MAX_TOKENS)
    tokens = np.zeros((MAX_TOKENS, TOKEN_DIM), dtype=np.float64)
    tokens[:count] = features[:count]
    mask = np.zeros(MAX_TOKENS, dtype=bool)
    mask[:count] = True
    return tokens, mask


def _lambda_coordinates(anchors: list) -> np.ndarray:
    coords = np.full((K, 2), -1, dtype=np.int16)
    for lam in range(K):
        if not is_click_lambda(lam):
            continue
        try:
            _, x, y = to_env_action(lam, anchors)
        except (IndexError, ValueError):
            continue
        coords[lam] = (int(x), int(y))
    return coords


def prepare_dense_example(
    example: DuckExample,
    max_click_distance: float = 12.0,
    drop_uncovered_clicks: bool = False,
    success_weight: float = 2.5,
    failure_weight: float = 0.6,
    informative_failure_weight: float = 1.5,
) -> PreparedDenseSFT | None:
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
    if (
        drop_uncovered_clicks
        and example.action == 6
        and match.distance > max_click_distance
    ):
        return None

    tokens, token_mask = _tokenize_grid(grid)
    next_tokens = np.zeros_like(tokens)
    next_mask = np.zeros_like(token_mask)
    has_next = example.next_grid is not None
    if has_next:
        next_tokens, next_mask = _tokenize_grid(np.asarray(example.next_grid, dtype=np.uint8))

    valid = valid_mask(_TeacherActionSpace(), anchors)
    if match.lambda_id < 0 or match.lambda_id >= len(valid) or not valid[match.lambda_id]:
        return None
    distance_weight = max(
        0.05,
        1.0 - match.distance / max(float(max_click_distance), 1e-6),
    )
    if example.success is True:
        outcome_weight = float(success_weight)
    elif example.success is False:
        outcome_weight = float(failure_weight)
        if example.visual_changed or example.level_advanced:
            outcome_weight *= float(informative_failure_weight)
    else:
        outcome_weight = 1.0
    if example.remaining_path_length is not None:
        # Retain all steps but gently prioritize decisions near observed success.
        outcome_weight *= 1.0 + 0.5 / (1.0 + max(0, example.remaining_path_length))

    return PreparedDenseSFT(
        tokens=tokens,
        token_mask=token_mask,
        trace=TraceSummary().features(),
        valid=valid,
        lam_features=arg_features(anchors),
        next_tokens=next_tokens,
        next_token_mask=next_mask,
        label=int(match.lambda_id),
        weight=float(distance_weight * outcome_weight),
        game_id=example.game_id,
        seed=-1 if example.seed is None else int(example.seed),
        trajectory_id=example.trajectory_id or example.source,
        raw_action=int(example.action),
        teacher_x=-1 if example.x is None else int(example.x),
        teacher_y=-1 if example.y is None else int(example.y),
        lambda_xy=_lambda_coordinates(anchors),
        proposal_distance=float(match.distance),
        visual_changed=-1 if example.visual_changed is None else int(example.visual_changed),
        level_advanced=-1 if example.level_advanced is None else int(example.level_advanced),
        has_next=bool(has_next),
        success=-1 if example.success is None else int(example.success),
    )


def prepare_dense_examples(
    examples: Iterable[DuckExample],
    **kwargs,
) -> list[PreparedDenseSFT]:
    prepared: list[PreparedDenseSFT] = []
    for example in examples:
        item = prepare_dense_example(example, **kwargs)
        if item is not None:
            prepared.append(item)
    return prepared


def split_by_game_and_seed(
    items: Sequence[PreparedDenseSFT],
    held_out_games: Sequence[str] = (),
    held_out_game_count: int = 1,
    validation_fraction: float = 0.2,
    seed: int = 0,
) -> DatasetSplit:
    """Split whole games and whole ``(game, seed)`` groups, never transitions."""
    if not items:
        return DatasetSplit((), (), (), (), (), ())
    rng = np.random.default_rng(seed)
    all_games = sorted({item.game_id for item in items})
    requested = [game for game in held_out_games if game in all_games]
    if requested:
        held_games = tuple(sorted(set(requested)))
    else:
        count = min(max(0, int(held_out_game_count)), max(0, len(all_games) - 1))
        shuffled_games = list(all_games)
        rng.shuffle(shuffled_games)
        held_games = tuple(sorted(shuffled_games[:count]))

    held_set = set(held_games)
    held = tuple(item for item in items if item.game_id in held_set)
    remainder = [item for item in items if item.game_id not in held_set]
    groups = sorted({(item.game_id, item.seed) for item in remainder})
    shuffled_groups = list(groups)
    rng.shuffle(shuffled_groups)
    if len(shuffled_groups) <= 1:
        validation_groups: set[tuple[str, int]] = set()
    else:
        validation_count = max(1, int(round(len(shuffled_groups) * validation_fraction)))
        validation_count = min(validation_count, len(shuffled_groups) - 1)
        validation_groups = set(shuffled_groups[:validation_count])
    validation = tuple(
        item for item in remainder if (item.game_id, item.seed) in validation_groups
    )
    train = tuple(
        item for item in remainder if (item.game_id, item.seed) not in validation_groups
    )
    return DatasetSplit(
        train=train,
        validation=validation,
        held_out=held,
        held_out_games=held_games,
        train_groups=tuple(sorted({(item.game_id, item.seed) for item in train})),
        validation_groups=tuple(sorted(validation_groups)),
    )


class AuxiliaryHeads(Module):
    def __init__(self, cfg, rng: np.random.Generator):
        d_input = cfg.d_c + cfg.d_A
        hidden = max(32, d_input)
        self.frame_change = MLP([d_input, hidden, 1], rng)
        self.level_advance = MLP([d_input, hidden, 1], rng)
        self.next_embedding = MLP([d_input, hidden, cfg.d_c], rng)


def _bce(logits: T, targets: np.ndarray) -> T:
    probabilities = logits.sigmoid()
    target = T(targets)
    return -(target * (probabilities + 1e-8).log() + (1.0 - target) * (1.0 - probabilities + 1e-8).log())


def _weighted(losses: T, weights: np.ndarray, mask: np.ndarray | None = None) -> T:
    effective = np.asarray(weights, dtype=np.float64)
    if mask is not None:
        effective = effective * np.asarray(mask, dtype=np.float64)
    return (losses * T(effective)).sum() / float(max(effective.sum(), 1e-8))


class DenseDuckSFTTrainer:
    """Jointly clone the router and train temporary representation-shaping heads."""

    def __init__(
        self,
        solver,
        seed: int = 0,
        lr: float = 3e-3,
        frame_change_coef: float = 0.3,
        level_advance_coef: float = 0.5,
        next_embedding_coef: float = 0.2,
        click_coverage_distance: float = 12.0,
    ) -> None:
        self.solver = solver
        self.heads = AuxiliaryHeads(solver.cfg, np.random.default_rng(seed + 991))
        self.opt = Adam([*solver.model.parameters(), *self.heads.parameters()], lr=lr)
        self.frame_change_coef = float(frame_change_coef)
        self.level_advance_coef = float(level_advance_coef)
        self.next_embedding_coef = float(next_embedding_coef)
        self.click_coverage_distance = float(click_coverage_distance)

    def _forward(self, batch: Sequence[PreparedDenseSFT]) -> dict[str, object]:
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
        shift = np.max(q_values.data, axis=1)
        log_partition = (
            (q_values - T(shift[:, None])).exp().sum(axis=1) + 1e-12
        ).log() + T(shift)
        action_losses = log_partition - teacher_q
        action_loss = _weighted(action_losses, weights)

        selected_features = lam_features[np.arange(len(batch)), labels]
        action_embedding = (
            model.q_router.base_embeddings().gather_rows(labels)
            + model.q_router.obj_proj(T(selected_features))
        )
        pooled = latent.mean(axis=1)
        auxiliary_input = T.concat([pooled, action_embedding], axis=-1)
        change_logits = self.heads.frame_change(auxiliary_input).reshape(len(batch))
        advance_logits = self.heads.level_advance(auxiliary_input).reshape(len(batch))
        predicted_next = self.heads.next_embedding(auxiliary_input)

        changed = np.asarray([max(0, item.visual_changed) for item in batch], dtype=np.float64)
        changed_mask = np.asarray([item.visual_changed >= 0 for item in batch], dtype=np.float64)
        advanced = np.asarray([max(0, item.level_advanced) for item in batch], dtype=np.float64)
        advanced_mask = np.asarray([item.level_advanced >= 0 for item in batch], dtype=np.float64)
        next_mask = np.asarray([item.has_next for item in batch], dtype=np.float64)

        change_loss = _weighted(_bce(change_logits, changed), weights, changed_mask)
        advance_loss = _weighted(_bce(advance_logits, advanced), weights, advanced_mask)
        next_tokens = np.stack([item.next_tokens for item in batch])
        next_token_masks = np.stack([item.next_token_mask for item in batch])
        next_latent = model.encode(next_tokens, next_token_masks, traces)
        next_target = T(next_latent.data.mean(axis=1))
        next_losses = ((predicted_next - next_target) ** 2).mean(axis=1)
        next_loss = _weighted(next_losses, weights, next_mask)
        total = (
            action_loss
            + self.frame_change_coef * change_loss
            + self.level_advance_coef * advance_loss
            + self.next_embedding_coef * next_loss
        )
        return {
            "total": total,
            "action_loss": action_loss,
            "change_loss": change_loss,
            "advance_loss": advance_loss,
            "next_loss": next_loss,
            "q_values": q_values,
            "change_logits": change_logits,
            "advance_logits": advance_logits,
            "predicted_next": predicted_next,
            "next_target": next_target,
            "labels": labels,
            "weights": weights,
            "changed": changed,
            "changed_mask": changed_mask,
            "advanced": advanced,
            "advanced_mask": advanced_mask,
            "next_mask": next_mask,
        }

    def train_batch(self, batch: Sequence[PreparedDenseSFT]) -> dict[str, float]:
        output = self._forward(batch)
        self.opt.zero_grad()
        output["total"].backward()
        self.opt.step()
        self.solver.updates += 1
        self.solver.caches.clear()
        return self._metrics(batch, output)

    def evaluate(self, batch: Sequence[PreparedDenseSFT]) -> dict[str, object]:
        if not batch:
            return {"count": 0}
        output = self._forward(batch)
        metrics: dict[str, object] = self._metrics(batch, output)
        metrics["count"] = len(batch)
        per_game: dict[str, dict[str, float]] = {}
        labels = output["labels"]
        predictions = np.argmax(output["q_values"].data, axis=1)
        advance_predictions = output["advance_logits"].data >= 0.0
        for game in sorted({item.game_id for item in batch}):
            indices = np.asarray(
                [index for index, item in enumerate(batch) if item.game_id == game],
                dtype=np.int64,
            )
            key_indices = np.asarray(
                [index for index in indices if batch[int(index)].raw_action != 6],
                dtype=np.int64,
            )
            click_indices = np.asarray(
                [index for index in indices if batch[int(index)].raw_action == 6],
                dtype=np.int64,
            )
            click_hits = [
                self._click_hit(batch[int(index)], int(predictions[int(index)]))
                for index in click_indices
            ]
            positive_advance = np.asarray(
                [
                    int(index)
                    for index in indices
                    if batch[int(index)].level_advanced == 1
                ],
                dtype=np.int64,
            )
            per_game[game] = {
                "count": float(len(indices)),
                "action_accuracy": float(np.mean(predictions[indices] == labels[indices])),
                "key_action_accuracy": (
                    float(np.mean(predictions[key_indices] == labels[key_indices]))
                    if len(key_indices)
                    else 0.0
                ),
                "click_coordinate_coverage": (
                    float(np.mean(click_hits)) if click_hits else 0.0
                ),
                "level_advance_recall": (
                    float(np.mean(advance_predictions[positive_advance]))
                    if len(positive_advance)
                    else 0.0
                ),
            }
        metrics["per_game"] = per_game
        return metrics

    def _click_hit(self, item: PreparedDenseSFT, prediction: int) -> bool:
        if item.teacher_x < 0 or item.teacher_y < 0:
            return False
        xy = item.lambda_xy[prediction] if 0 <= prediction < len(item.lambda_xy) else (-1, -1)
        if int(xy[0]) < 0 or int(xy[1]) < 0:
            return False
        distance = float(
            np.hypot(int(xy[0]) - item.teacher_x, int(xy[1]) - item.teacher_y)
        )
        return distance <= self.click_coverage_distance

    def _metrics(
        self,
        batch: Sequence[PreparedDenseSFT],
        output: dict[str, object],
    ) -> dict[str, float]:
        labels = output["labels"]
        weights = output["weights"]
        q_values = output["q_values"].data
        predictions = np.argmax(q_values, axis=1)
        key_mask = np.asarray([item.raw_action != 6 for item in batch], dtype=bool)
        key_accuracy = (
            float(np.mean(predictions[key_mask] == labels[key_mask]))
            if np.any(key_mask)
            else 0.0
        )

        click_covered = []
        proposal_covered = []
        for index, item in enumerate(batch):
            if item.raw_action != 6:
                continue
            proposal_covered.append(
                item.proposal_distance <= self.click_coverage_distance
            )
            click_covered.append(self._click_hit(item, int(predictions[index])))

        advance_mask = output["advanced_mask"].astype(bool)
        advance_targets = output["advanced"].astype(bool)
        advance_predictions = output["advance_logits"].data >= 0.0
        positives = advance_mask & advance_targets
        advance_recall = (
            float(np.mean(advance_predictions[positives])) if np.any(positives) else 0.0
        )
        change_mask = output["changed_mask"].astype(bool)
        change_predictions = output["change_logits"].data >= 0.0
        change_accuracy = (
            float(np.mean(change_predictions[change_mask] == output["changed"][change_mask]))
            if np.any(change_mask)
            else 0.0
        )
        next_mask = output["next_mask"].astype(bool)
        if np.any(next_mask):
            predicted_next = output["predicted_next"].data[next_mask]
            next_target = output["next_target"].data[next_mask]
            numerator = np.sum(predicted_next * next_target, axis=1)
            denominator = np.linalg.norm(predicted_next, axis=1) * np.linalg.norm(next_target, axis=1)
            next_cosine = float(np.mean(numerator / np.maximum(denominator, 1e-8)))
        else:
            next_cosine = 0.0
        return {
            "loss": float(output["total"].data),
            "action_loss": float(output["action_loss"].data),
            "frame_change_loss": float(output["change_loss"].data),
            "level_advance_loss": float(output["advance_loss"].data),
            "next_embedding_loss": float(output["next_loss"].data),
            "accuracy": float(np.average(predictions == labels, weights=weights)),
            "key_action_accuracy": key_accuracy,
            "click_coordinate_coverage": float(np.mean(click_covered)) if click_covered else 0.0,
            "click_proposal_coverage": float(np.mean(proposal_covered)) if proposal_covered else 0.0,
            "level_advance_recall": advance_recall,
            "frame_change_accuracy": change_accuracy,
            "next_embedding_cosine": next_cosine,
            "mean_weight": float(np.mean(weights)),
        }
