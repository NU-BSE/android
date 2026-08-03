"""MRPO group selection and checkpoint-prefix evaluation for GUI trajectories.

The helpers are intentionally model- and environment-agnostic. They implement
three pieces used by the Android settings distiller:

* on-policy oversample/select groups that avoid all-success/all-failure collapse;
* group-relative terminal advantages;
* verified longest-prefix progress over a documented settings ``ui_path``.

All candidate trajectories supplied to :func:`select_training_group` must have
been sampled from the same frozen policy version. The function never introduces
replay data, so a swapped trajectory remains on-policy.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Mapping, Sequence, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class GroupSelection:
    """Selected indices plus whether a policy-gradient comparison is usable."""

    indices: tuple[int, ...]
    informative: bool
    swapped: bool


def is_collapsed(outcomes: Sequence[bool | int]) -> bool:
    """Return true when every rollout has the same binary terminal outcome."""
    if not outcomes:
        return True
    successes = sum(bool(value) for value in outcomes)
    return successes in {0, len(outcomes)}


def pool_diversity_probability(success_probability: float, n: int, k: int) -> float:
    """Probability that an on-policy pool of ``k*n`` contains both outcomes."""
    p = float(success_probability)
    if not 0.0 <= p <= 1.0:
        raise ValueError("success_probability must be in [0, 1]")
    if n < 1 or k < 1:
        raise ValueError("n and k must be positive")
    size = n * k
    return 1.0 - math.pow(p, size) - math.pow(1.0 - p, size)


def select_training_group(
    outcomes: Sequence[bool | int],
    n: int,
    rng: random.Random,
) -> GroupSelection:
    """Uniformly select ``n`` rollouts, swapping one if a mixed pool permits it.

    This follows the Mobile-Agent-v3.5 MRPO construction: first uniformly
    downsample the current-policy pool. If that group is collapsed while the
    pool contains both outcomes, replace one random member with a random member
    carrying the opposite terminal outcome. A collapsed pool is still returned
    for supervised learning, but ``informative`` is false so the caller can zero
    its policy-gradient advantage.
    """
    if n < 2:
        raise ValueError("MRPO group size n must be at least 2")
    if len(outcomes) < n:
        raise ValueError("on-policy pool must contain at least n rollouts")

    indices = rng.sample(range(len(outcomes)), n)
    selected_outcomes = [bool(outcomes[index]) for index in indices]
    if not is_collapsed(selected_outcomes):
        return GroupSelection(tuple(indices), True, False)

    pool_outcomes = [bool(value) for value in outcomes]
    if is_collapsed(pool_outcomes):
        return GroupSelection(tuple(indices), False, False)

    selected_value = selected_outcomes[0]
    opposite = [
        index
        for index, outcome in enumerate(pool_outcomes)
        if outcome != selected_value and index not in indices
    ]
    if not opposite:
        return GroupSelection(tuple(indices), False, False)
    replace_at = rng.randrange(n)
    indices[replace_at] = rng.choice(opposite)
    return GroupSelection(tuple(indices), True, True)


def group_relative_advantages(outcomes: Sequence[bool | int]) -> tuple[float, ...]:
    """Centered, variance-normalized binary outcome advantages.

    Collapsed groups yield all zeros. This prevents meaningless all-success or
    all-failure groups from contributing a policy-gradient term while allowing
    the same samples to retain supervised/grounding losses.
    """
    if not outcomes:
        return ()
    values = [1.0 if bool(value) else 0.0 for value in outcomes]
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    if variance <= 0.0:
        return tuple(0.0 for _ in values)
    scale = math.sqrt(variance) + 1e-8
    return tuple((value - mean) / scale for value in values)


def route_from_task(task: object) -> tuple[str, ...]:
    """Extract a normalized documented settings route from a curriculum task."""
    expected = getattr(task, "expected_output", {})
    evidence = getattr(task, "evidence", {})
    raw: object = ()
    if isinstance(expected, Mapping):
        raw = expected.get("ui_path") or ()
    if not raw and isinstance(evidence, Mapping):
        raw = evidence.get("ui_path") or ()
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return ()
    route = tuple(str(value).strip() for value in raw if str(value).strip())
    if route and route[0].casefold() in {"settings", "настройки"}:
        route = route[1:]
    return route


def label_satisfied(label: str, visible_labels: Sequence[str]) -> bool:
    """Case-insensitive exact/containment predicate used as ``phi_k``."""
    needle = " ".join(str(label).casefold().split())
    if not needle:
        return False
    for value in visible_labels:
        candidate = " ".join(str(value).casefold().split())
        if candidate == needle or needle in candidate:
            return True
    return False


def completed_prefix_length(
    route: Sequence[str], observations: Sequence[object]
) -> int:
    """Compute the longest route prefix whose checkpoint appeared in rollout.

    For route checkpoint ``k`` this implements ``c_k(tau)=max_t phi_k(o_t)``;
    the result is ``m(tau)=max{m: all k<=m have c_k=1}``.
    """
    if not route:
        return 0
    completed = 0
    for checkpoint in route:
        found = any(
            label_satisfied(
                checkpoint,
                tuple(getattr(observation, "visible_labels", ()) or ()),
            )
            for observation in observations
        )
        if not found:
            break
        completed += 1
    return completed


def last_checkpoint_step(
    route: Sequence[str], observations: Sequence[object], prefix_length: int
) -> int | None:
    """Return the last observation index satisfying the completed prefix."""
    if prefix_length <= 0 or prefix_length > len(route):
        return None
    target = route[prefix_length - 1]
    matches = [
        index
        for index, observation in enumerate(observations)
        if label_satisfied(
            target, tuple(getattr(observation, "visible_labels", ()) or ())
        )
    ]
    return max(matches) if matches else None


def truncate_trajectory_to_prefix(
    trajectory: Sequence[T], route: Sequence[str], observations: Sequence[object]
) -> tuple[T, ...]:
    """Keep only the clean segment through the last verified checkpoint.

    ``observations`` should contain the initial observation followed by each
    transition's next observation when present. A trajectory item at index
    ``t`` is the action that produced observation ``t+1``.
    """
    prefix = completed_prefix_length(route, observations)
    step = last_checkpoint_step(route, observations, prefix)
    if step is None:
        return tuple(trajectory)
    return tuple(trajectory[: max(0, step)])


__all__ = [
    "GroupSelection",
    "completed_prefix_length",
    "group_relative_advantages",
    "is_collapsed",
    "label_satisfied",
    "last_checkpoint_step",
    "pool_diversity_probability",
    "route_from_task",
    "select_training_group",
    "truncate_trajectory_to_prefix",
]
