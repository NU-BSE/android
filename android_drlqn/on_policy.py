"""Coverage-safe on-policy runtime.

A documentation section can emit several VLM/LLM tasks. The section is marked
validated only after every linked task has been teacher-reviewed on every
selected profile for its OS family and at least one checkpoint was written.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np

from .distillation import (
    DistillationMetrics,
    DistillationSample,
    OnPolicyDistiller as _BaseDistiller,
)
from .document_curriculum import CoverageStatus, CurriculumTask
from .emulator_lab import EmulatorMatrix


class OnPolicyDistiller(_BaseDistiller):
    def train(self, tasks: Sequence[CurriculumTask], matrix: EmulatorMatrix) -> DistillationMetrics:
        if not tasks:
            return DistillationMetrics(0, 0, 0, 0.0, 0.0, None)
        samples: list[DistillationSample] = []
        rewards: list[float] = []
        scores: list[float] = []
        verified = updates = 0
        checkpoint = None
        task_profiles: dict[str, set[str]] = defaultdict(set)
        reviewed_task_profiles: set[tuple[str, str]] = set()

        for task in tasks:
            profiles = matrix.select_for_os(task.os_family)
            for profile in profiles:
                task_profiles[task.task_id].add(profile.profile_id)
                environment = self.environment_factory(profile)
                observation = environment.reset(task, profile)
                trajectory = []
                for _ in range(self.config.max_steps_per_task):
                    proposal = self.student.propose(task, profile, observation)
                    lag = self.student.policy_version - proposal.policy_version
                    if lag > self.config.max_policy_lag:
                        raise RuntimeError(f"off-policy rollout rejected: policy lag {lag}")
                    outcome = environment.step(proposal)
                    judgement = self.teacher.evaluate(task, profile, observation, proposal, outcome)
                    if judgement.confidence < self.config.min_teacher_confidence:
                        raise RuntimeError(
                            f"teacher confidence {judgement.confidence:.3f} below "
                            f"configured threshold {self.config.min_teacher_confidence:.3f}; "
                            f"teacher={judgement.teacher_id}; tags={judgement.tags}"
                        )
                    reward = float(outcome.reward) + 0.5 * judgement.total_score + 0.25 * judgement.safety_score
                    trajectory.append((observation, proposal, outcome, judgement, reward))
                    rewards.append(reward)
                    scores.append(judgement.total_score)
                    verified += int(outcome.verified)
                    reviewed_task_profiles.add((task.task_id, profile.profile_id))
                    if self.coverage:
                        entry = self.coverage.entries.get(task.coverage_id)
                        if entry and entry.status not in {CoverageStatus.NON_ACTIONABLE, CoverageStatus.BLOCKED, CoverageStatus.VALIDATED}:
                            self.coverage.mark_rollout(task.coverage_id, profile.profile_id)
                            self.coverage.mark_teacher_review(task.coverage_id, judgement.teacher_id)
                    if outcome.terminal or outcome.next_observation is None:
                        break
                    observation = outcome.next_observation

                advantages = self._advantages([row[4] for row in trajectory])
                for row, advantage in zip(trajectory, advantages):
                    observation, proposal, outcome, judgement, reward = row
                    samples.append(DistillationSample(task, profile, observation, proposal, outcome, judgement, reward, advantage))
                    if len(samples) >= self.config.rollout_batch_size:
                        checkpoint = self._update(samples)
                        updates += 1
                        samples.clear()

        if samples:
            checkpoint = self._update(samples)
            updates += 1

        if self.coverage and checkpoint:
            tasks_by_coverage: dict[str, list[CurriculumTask]] = defaultdict(list)
            for task in tasks:
                tasks_by_coverage[task.coverage_id].append(task)
            for coverage_id, linked_tasks in tasks_by_coverage.items():
                entry = self.coverage.entries.get(coverage_id)
                if not entry or entry.status in {CoverageStatus.NON_ACTIONABLE, CoverageStatus.BLOCKED}:
                    continue
                self.coverage.mark_student_update(coverage_id, checkpoint)
                complete = all(
                    all((task.task_id, profile_id) in reviewed_task_profiles for profile_id in task_profiles[task.task_id])
                    and bool(task_profiles[task.task_id])
                    for task in linked_tasks
                )
                if complete:
                    self.coverage.mark_validated(coverage_id, "all linked tasks and selected profiles were teacher-reviewed and checkpointed")

        return DistillationMetrics(
            rollouts=len(rewards),
            updates=updates,
            verified_successes=verified,
            mean_reward=float(np.mean(rewards)) if rewards else 0.0,
            mean_teacher_score=float(np.mean(scores)) if scores else 0.0,
            checkpoint_id=checkpoint,
        )
