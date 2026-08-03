import random
from types import SimpleNamespace

import pytest

from android_drlqn.mrpo import (
    completed_prefix_length,
    group_relative_advantages,
    is_collapsed,
    pool_diversity_probability,
    select_training_group,
    truncate_trajectory_to_prefix,
)


def observation(*labels):
    return SimpleNamespace(visible_labels=labels)


def test_pool_diversity_probability_matches_mrpo_formula():
    assert pool_diversity_probability(0.5, n=2, k=2) == pytest.approx(0.875)
    assert pool_diversity_probability(0.0, n=2, k=2) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        pool_diversity_probability(1.1, n=2, k=2)


def test_oversample_select_repairs_collapsed_subgroup_from_same_pool():
    outcomes = (False, False, False, True)
    selection = select_training_group(outcomes, n=2, rng=random.Random(4))
    selected = [outcomes[index] for index in selection.indices]
    assert selection.informative
    assert not is_collapsed(selected)
    assert set(selection.indices).issubset(set(range(len(outcomes))))


def test_collapsed_pool_stays_supervised_only():
    selection = select_training_group((True, True, True, True), 2, random.Random(0))
    assert not selection.informative
    assert is_collapsed([True for _ in selection.indices])
    assert group_relative_advantages((True, True)) == (0.0, 0.0)


def test_group_relative_advantages_are_centered():
    advantages = group_relative_advantages((False, True))
    assert sum(advantages) == pytest.approx(0.0)
    assert advantages[0] < 0 < advantages[1]


def test_checkpoint_prefix_and_clean_truncation():
    route = ("Network & internet", "Internet", "Private DNS")
    observations = (
        observation("Settings", "Network & internet"),
        observation("Network & internet", "Internet"),
        observation("SIMs"),
    )
    assert completed_prefix_length(route, observations) == 2
    trajectory = ("tap-network", "tap-internet", "wrong-tap")
    assert truncate_trajectory_to_prefix(trajectory, route, observations) == (
        "tap-network",
    )
