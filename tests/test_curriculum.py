import numpy as np

from drlqn.config import drlqn_xs
from drlqn.train.curriculum import (
    EnvironmentTransitionTrainer,
    PlannerValueDistiller,
    planner_value_target,
)
from drlqn.train.dense_sft import DenseDuckSFTTrainer, prepare_dense_example
from drlqn.train.duck_dataset import DuckExample
from drlqn.train.submission_solver import SubmissionDRLQNSolver


def _prepared(*, trajectory: str, changed: bool, advanced: bool, success: bool):
    grid = np.zeros((64, 64), dtype=np.uint8)
    grid[8:12, 8:12] = 7
    next_grid = grid.copy()
    if changed:
        next_grid[20:24, 20:24] = 9
    example = DuckExample(
        game_id="demo",
        message_index=0,
        grid=grid,
        action=1,
        source=trajectory,
        next_grid=next_grid,
        visual_changed=changed,
        level_advanced=advanced,
        terminal_result="NOT_FINISHED",
        seed=0,
        success=success,
        trajectory_id=trajectory,
        level_before=1,
        level_after=2 if advanced else 1,
    )
    prepared = prepare_dense_example(example)
    assert prepared is not None
    return prepared


def test_curriculum_shares_auxiliary_heads_and_learns_negative_evidence():
    random_noop = _prepared(
        trajectory="generated/random/demo/seed-0",
        changed=False,
        advanced=False,
        success=False,
    )
    specialist_advance = _prepared(
        trajectory="generated/specialist/demo/seed-0",
        changed=True,
        advanced=True,
        success=True,
    )
    assert planner_value_target(random_noop) < planner_value_target(specialist_advance)

    solver = SubmissionDRLQNSolver(drlqn_xs(), seed=0)
    duck_trainer = DenseDuckSFTTrainer(solver, seed=0)
    first = duck_trainer.train_batch([specialist_advance])
    assert np.isfinite(first["loss"])

    world = EnvironmentTransitionTrainer(
        solver,
        seed=0,
        heads=duck_trainer.heads,
    )
    assert world.heads is duck_trainer.heads
    second = world.train_batch([random_noop, specialist_advance])
    assert np.isfinite(second["loss"])

    planner = PlannerValueDistiller(solver)
    third = planner.train_batch([random_noop, specialist_advance])
    assert np.isfinite(third["loss"])
    assert np.isfinite(third["planner_value_loss"])
