"""Generate generic offline experience from official ARC environments.

Collectors use only rendered frames, legal actions and the transition graph built
while acting. They deliberately avoid environment source inspection and game-specific
solution logic.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
from typing import Any, Literal

import numpy as np

from ..operators.library import to_env_action
from .duck_dataset import DuckExample

PolicyName = Literal["random", "frontier", "specialist"]


def _frame_key(grid: np.ndarray, level: int) -> str:
    return hashlib.sha1(
        np.asarray(grid, dtype=np.uint8).tobytes()
        + int(level).to_bytes(2, "little", signed=False)
    ).hexdigest()[:16]


def _select_lambda(solver: Any, node: Any, policy: PolicyName, actions_taken: int) -> int | None:
    if policy == "random":
        valid = np.where(node.valid)[0]
        if not len(valid):
            return None
        selected = int(solver.rng.choice(valid))
        solver._pending_lambda = selected
        return selected
    return solver.select(node, actions_taken)


def collect_trajectory(
    env: Any,
    solver: Any,
    *,
    policy: PolicyName,
    seed: int,
    max_actions_per_level: int,
    max_levels: int = 128,
) -> list[DuckExample]:
    """Collect one dense trajectory without performing online gradient updates."""
    frame = env.reset()
    node = solver.perceive(env, frame.grid)
    trajectory_id = f"generated/{policy}/{env.env_id}/seed-{int(seed)}"
    examples: list[DuckExample] = []
    visual_visits: dict[tuple[int, str], int] = {}

    while env.state == "NOT_FINISHED":
        level = int(env.level)
        used = int(env._level_action_counts.get(level, 0))
        if level > int(max_levels) or used >= int(max_actions_per_level):
            break
        lam = _select_lambda(solver, node, policy, used)
        if lam is None:
            reset_frame = env.reset_level()
            node = solver.perceive(env, reset_frame.grid)
            continue

        action, x, y = to_env_action(lam, node.objs)
        before = np.asarray(node.grid, dtype=np.uint8).copy()
        level_before = int(env.level)
        step = env.step(action, x, y)
        next_node = solver.perceive(env, step.frame.grid)
        after = np.asarray(next_node.grid, dtype=np.uint8).copy()
        level_after = int(env.level)
        changed = not np.array_equal(before, after)
        advanced = bool(step.level_won or level_after > level_before)
        key = (level_after, _frame_key(after, level_after))
        prior_visits = visual_visits.get(key, 0)
        visual_visits[key] = prior_visits + 1

        evidence = []
        if not changed:
            evidence.append("negative:no_change")
        if changed and prior_visits:
            evidence.append("negative:cycle_or_regression")
        if advanced:
            evidence.append("positive:level_advance")
        if step.env_won:
            evidence.append("positive:environment_win")
        if env.state == "GAME_OVER":
            evidence.append("negative:terminal_failure")

        solver.graph.setdefault(node.key, {})[int(lam)] = next_node.key
        solver._update_priors(node, int(lam), next_node, changed)
        examples.append(
            DuckExample(
                game_id=str(env.env_id),
                message_index=len(examples),
                grid=before,
                action=int(action),
                x=x,
                y=y,
                reasoning=";".join(evidence) or f"generated:{policy}",
                source=f"environment-generated:{policy}",
                next_grid=after,
                visual_changed=changed,
                level_advanced=advanced,
                terminal_result=str(env.state),
                seed=int(seed),
                trajectory_id=trajectory_id,
                level_before=level_before,
                level_after=level_after,
                action_index=len(examples),
            )
        )
        node = next_node
        if step.env_won or env.state == "GAME_OVER":
            break

    won = bool(env.state == "WIN")
    total = len(examples)
    return [
        replace(
            example,
            remaining_path_length=total - index - 1,
            success=won,
        )
        for index, example in enumerate(examples)
    ]


def experience_summary(examples: list[DuckExample]) -> dict[str, Any]:
    trajectories = {example.trajectory_id for example in examples}
    games = {example.game_id for example in examples}
    return {
        "transitions": len(examples),
        "trajectories": len(trajectories),
        "games": len(games),
        "visual_change_rate": float(np.mean([bool(item.visual_changed) for item in examples])) if examples else 0.0,
        "level_advances": sum(int(bool(item.level_advanced)) for item in examples),
        "successful_transitions": sum(int(item.success is True) for item in examples),
        "no_change_negative_evidence": sum("negative:no_change" in item.reasoning for item in examples),
        "cycle_negative_evidence": sum("negative:cycle_or_regression" in item.reasoning for item in examples),
        "terminal_failures": sum("negative:terminal_failure" in item.reasoning for item in examples),
    }
