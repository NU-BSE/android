"""RHAE scoring (technical report section 4.1).

S_l = min(1.15, (h_l / a_l)^2)          per-level efficiency, squared, capped
E_e = min(cap, sum(w_l S_l)/sum(w_l))   w_l = l, cap = weighted completed frac
T   = mean_e E_e
"""
from __future__ import annotations


def level_score(h: int, a: int | None) -> float:
    if a is None or a <= 0:
        return 0.0
    return min(1.15, (h / a) ** 2)


def environment_score(baselines: dict[int, int], actions: dict[int, int | None],
                      completed: int) -> float:
    levels = sorted(baselines)
    wsum = sum(levels)
    cap = sum(l for l in levels if l <= completed) / wsum
    score = sum(l * level_score(baselines[l], actions.get(l))
                for l in levels if l <= completed) / wsum
    return min(cap, score)


def total_score(env_scores: list[float]) -> float:
    return sum(env_scores) / len(env_scores) if env_scores else 0.0
