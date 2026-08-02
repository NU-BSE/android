"""Crash-resilient on-policy training progress and emergency checkpoints.

Installed only by ``scripts/train_on_policy_ollama.py``. The wrapper keeps the
existing public API intact while adding terminal progress, a JSONL event log,
an atomically replaced state file, emergency checkpoints, and bounded retries
when an emulator disappears from ADB.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .distillation import (
    DistillationMetrics,
    DistillationSample,
    EnvironmentOutcome,
    RolloutObservation,
    StudentProposal,
    TeacherJudgement,
)

ENV_PROGRESS_FILE = "ANDROID_DRLQN_TRAINING_PROGRESS_FILE"
ENV_STATE_FILE = "ANDROID_DRLQN_TRAINING_STATE_FILE"
ENV_QUIET = "ANDROID_DRLQN_TRAINING_QUIET"
ENV_EMULATOR_RETRIES = "ANDROID_DRLQN_EMULATOR_RETRIES"
ENV_EMULATOR_RETRY_DELAY = "ANDROID_DRLQN_EMULATOR_RETRY_DELAY_S"
DEFAULT_PROGRESS_FILE = "artifacts/training-progress.jsonl"
DEFAULT_STATE_FILE = "artifacts/training-state.json"

_ADB_DISCONNECT_MARKERS = (
    "device not found",
    "device offline",
    "no devices/emulators found",
    "no device/emulator",
    "transport error",
    "transport is closing",
    "connection reset",
    "closed",
    "cannot connect to daemon",
    "adb server is out of date",
)


class TrainingProgress:
    """Small dependency-free terminal and JSON progress reporter."""

    def __init__(self) -> None:
        self.progress_path = Path(
            os.environ.get(ENV_PROGRESS_FILE, DEFAULT_PROGRESS_FILE)
        )
        self.state_path = Path(os.environ.get(ENV_STATE_FILE, DEFAULT_STATE_FILE))
        self.quiet = os.environ.get(ENV_QUIET) == "1"
        self.started_at = time.time()
        self.progress_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: str, **values: object) -> None:
        now = time.time()
        payload: dict[str, object] = {
            "event": event,
            "timestamp": round(now, 3),
            "elapsed_s": round(now - self.started_at, 3),
            **values,
        }
        with self.progress_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            if event in {"checkpoint", "retry", "crash", "complete"}:
                try:
                    os.fsync(handle.fileno())
                except OSError:
                    pass

        temporary = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.state_path)

        if not self.quiet:
            print(self._line(payload), file=sys.stderr, flush=True)

    @staticmethod
    def _line(payload: Mapping[str, object]) -> str:
        event = str(payload.get("event") or "progress").upper()
        parts = [f"[{time.strftime('%H:%M:%S')}]", f"[{event}]"]
        profile = payload.get("profile_id")
        task_index = payload.get("task_index")
        task_total = payload.get("task_total")
        step = payload.get("step")
        max_steps = payload.get("max_steps")
        attempt = payload.get("attempt")
        if profile:
            parts.append(str(profile))
        if task_index is not None and task_total is not None:
            parts.append(f"task={task_index}/{task_total}")
        if step is not None and max_steps is not None:
            parts.append(f"step={step}/{max_steps}")
        if attempt is not None:
            parts.append(f"attempt={attempt}")
        for name in ("rollouts", "updates", "verified", "pending_samples"):
            if name in payload:
                parts.append(f"{name}={payload[name]}")
        checkpoint = payload.get("checkpoint_id")
        if checkpoint:
            parts.append(f"checkpoint={checkpoint}")
        error = payload.get("error")
        if error:
            parts.append(f"error={error}")
        return " ".join(parts)


def _int_env(name: str, default: int, *, minimum: int = 0) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _float_env(name: str, default: float, *, minimum: float = 0.0) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _is_adb_disconnect(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".casefold()
    quoted_device_not_found = (
        "adb" in text and "device" in text and "not found" in text
    )
    return quoted_device_not_found or any(
        marker in text for marker in _ADB_DISCONNECT_MARKERS
    )


def _trajectory_samples(
    distiller: object,
    task: object,
    profile: object,
    trajectory: Sequence[
        tuple[
            RolloutObservation,
            StudentProposal,
            EnvironmentOutcome,
            TeacherJudgement,
            float,
        ]
    ],
) -> list[DistillationSample]:
    if not trajectory:
        return []
    advantages = distiller._advantages([item[4] for item in trajectory])
    return [
        DistillationSample(
            task,
            profile,
            observation,
            proposal,
            outcome,
            judgement,
            reward,
            advantage,
        )
        for (observation, proposal, outcome, judgement, reward), advantage in zip(
            trajectory, advantages
        )
    ]


def _existing_checkpoint_id(directory: Path, step: int) -> str | None:
    identifier = directory / f"step-{step:08d}" / "checkpoint.id"
    if identifier.is_file():
        value = identifier.read_text(encoding="utf-8").strip()
        return value or None
    return None


def _save_current_weights(distiller: object) -> str | None:
    """Best-effort serialization when no completed sample can be updated."""
    student = getattr(distiller, "student", None)
    updater = getattr(distiller, "updater", None)
    directory = Path(getattr(updater, "checkpoint_dir", "artifacts/checkpoints"))
    directory.mkdir(parents=True, exist_ok=True)
    step = max(
        int(getattr(updater, "step_count", 0) or 0),
        int(getattr(student, "policy_version", 0) or 0),
    )

    existing = _existing_checkpoint_id(directory, step)
    if existing:
        return existing

    save_checkpoint = getattr(student, "save_checkpoint", None)
    if callable(save_checkpoint):
        return str(save_checkpoint(directory, step))

    save = getattr(student, "save", None)
    if callable(save):
        path = directory / f"emergency-policy-v{step}.npz"
        return str(save(path))
    return None


def _close_environment(environment: object | None) -> str | None:
    close = getattr(environment, "close", None)
    if not callable(close):
        return None
    try:
        close()
    except Exception as exc:  # closing must never hide the training error
        return f"{type(exc).__name__}: {exc}"
    return None


def _emergency_checkpoint(
    distiller: object,
    samples: list[DistillationSample],
    *,
    update_in_progress: bool,
) -> tuple[str | None, bool, str | None]:
    """Flush safe samples or serialize current weights without masking failure."""
    try:
        if samples and not update_in_progress:
            checkpoint = str(distiller._update(tuple(samples)))
            samples.clear()
            return checkpoint, True, None
        return _save_current_weights(distiller), False, None
    except Exception as exc:
        return None, False, f"{type(exc).__name__}: {exc}"


def _resilient_train(
    self: object, tasks: Sequence[object], matrix: object
) -> DistillationMetrics:
    progress = TrainingProgress()
    if not tasks:
        progress.emit("complete", rollouts=0, updates=0, verified=0)
        return DistillationMetrics(0, 0, 0, 0.0, 0.0, None)

    emulator_retries = _int_env(ENV_EMULATOR_RETRIES, 1, minimum=0)
    retry_delay_s = _float_env(ENV_EMULATOR_RETRY_DELAY, 5.0, minimum=0.0)
    samples: list[DistillationSample] = []
    rewards: list[float] = []
    teacher_scores: list[float] = []
    verified = 0
    updates = 0
    checkpoint: str | None = None
    total_episodes = sum(len(matrix.select_for_os(task.os_family)) for task in tasks)
    episode_index = 0

    progress.emit(
        "start",
        task_total=total_episodes,
        rollouts=0,
        updates=0,
        verified=0,
        policy_version=int(getattr(self.student, "policy_version", 0)),
        emulator_retries=emulator_retries,
    )

    for task in tasks:
        profiles = matrix.select_for_os(task.os_family)
        for profile in profiles:
            episode_index += 1
            attempt = 0
            episode_done = False
            while not episode_done:
                attempt += 1
                environment = None
                update_in_progress = False
                trajectory: list[
                    tuple[
                        RolloutObservation,
                        StudentProposal,
                        EnvironmentOutcome,
                        TeacherJudgement,
                        float,
                    ]
                ] = []
                progress.emit(
                    "episode",
                    task_id=task.task_id,
                    task_index=episode_index,
                    task_total=total_episodes,
                    profile_id=profile.profile_id,
                    attempt=attempt,
                    rollouts=len(rewards),
                    updates=updates,
                    verified=verified,
                    pending_samples=len(samples),
                    checkpoint_id=checkpoint,
                )
                try:
                    environment = self.environment_factory(profile)
                    observation = environment.reset(task, profile)
                    for step_index in range(self.config.max_steps_per_task):
                        progress.emit(
                            "step",
                            task_id=task.task_id,
                            task_index=episode_index,
                            task_total=total_episodes,
                            profile_id=profile.profile_id,
                            attempt=attempt,
                            step=step_index + 1,
                            max_steps=self.config.max_steps_per_task,
                            rollouts=len(rewards),
                            updates=updates,
                            verified=verified,
                            pending_samples=len(samples),
                            checkpoint_id=checkpoint,
                        )
                        proposal = self.student.propose(task, profile, observation)
                        lag = self.student.policy_version - proposal.policy_version
                        if lag > self.config.max_policy_lag:
                            raise RuntimeError(
                                f"off-policy rollout rejected: policy lag {lag}"
                            )
                        outcome = environment.step(proposal)
                        judgement = self.teacher.evaluate(
                            task, profile, observation, proposal, outcome
                        )
                        if judgement.confidence < self.config.min_teacher_confidence:
                            raise RuntimeError(
                                "teacher confidence below configured threshold"
                            )
                        reward = (
                            float(outcome.reward)
                            + 0.5 * judgement.total_score
                            + 0.25 * judgement.safety_score
                        )
                        trajectory.append(
                            (observation, proposal, outcome, judgement, reward)
                        )
                        rewards.append(reward)
                        teacher_scores.append(judgement.total_score)
                        verified += int(outcome.verified)
                        if self.coverage:
                            self.coverage.mark_rollout(
                                task.coverage_id, profile.profile_id
                            )
                            self.coverage.mark_teacher_review(
                                task.coverage_id, judgement.teacher_id
                            )
                        if outcome.terminal or outcome.next_observation is None:
                            break
                        observation = outcome.next_observation

                    samples.extend(_trajectory_samples(self, task, profile, trajectory))
                    trajectory.clear()
                    while len(samples) >= self.config.rollout_batch_size:
                        update_in_progress = True
                        batch = samples[: self.config.rollout_batch_size]
                        checkpoint = str(self._update(batch))
                        del samples[: self.config.rollout_batch_size]
                        updates += 1
                        update_in_progress = False
                        progress.emit(
                            "checkpoint",
                            task_id=task.task_id,
                            task_index=episode_index,
                            task_total=total_episodes,
                            profile_id=profile.profile_id,
                            attempt=attempt,
                            rollouts=len(rewards),
                            updates=updates,
                            verified=verified,
                            pending_samples=len(samples),
                            checkpoint_id=checkpoint,
                            policy_version=int(self.student.policy_version),
                        )
                    episode_done = True
                except KeyboardInterrupt as exc:
                    samples.extend(_trajectory_samples(self, task, profile, trajectory))
                    emergency, did_update, emergency_error = _emergency_checkpoint(
                        self, samples, update_in_progress=update_in_progress
                    )
                    if emergency:
                        checkpoint = emergency
                    updates += int(did_update)
                    close_error = _close_environment(environment)
                    progress.emit(
                        "crash",
                        task_id=task.task_id,
                        task_index=episode_index,
                        task_total=total_episodes,
                        profile_id=profile.profile_id,
                        attempt=attempt,
                        rollouts=len(rewards),
                        updates=updates,
                        verified=verified,
                        pending_samples=len(samples),
                        checkpoint_id=checkpoint,
                        policy_version=int(getattr(self.student, "policy_version", 0)),
                        error=f"{type(exc).__name__}: {exc}",
                        emergency_checkpoint_error=emergency_error,
                        close_error=close_error,
                    )
                    raise
                except Exception as exc:
                    samples.extend(_trajectory_samples(self, task, profile, trajectory))
                    trajectory.clear()
                    emergency, did_update, emergency_error = _emergency_checkpoint(
                        self, samples, update_in_progress=update_in_progress
                    )
                    if emergency:
                        checkpoint = emergency
                    updates += int(did_update)
                    close_error = _close_environment(environment)
                    can_retry = (
                        not update_in_progress
                        and emergency_error is None
                        and _is_adb_disconnect(exc)
                        and attempt <= emulator_retries
                    )
                    if can_retry:
                        progress.emit(
                            "retry",
                            task_id=task.task_id,
                            task_index=episode_index,
                            task_total=total_episodes,
                            profile_id=profile.profile_id,
                            attempt=attempt,
                            next_attempt=attempt + 1,
                            retry_delay_s=retry_delay_s,
                            rollouts=len(rewards),
                            updates=updates,
                            verified=verified,
                            pending_samples=len(samples),
                            checkpoint_id=checkpoint,
                            policy_version=int(
                                getattr(self.student, "policy_version", 0)
                            ),
                            error=f"{type(exc).__name__}: {exc}",
                            close_error=close_error,
                        )
                        if retry_delay_s:
                            time.sleep(retry_delay_s)
                        continue
                    progress.emit(
                        "crash",
                        task_id=task.task_id,
                        task_index=episode_index,
                        task_total=total_episodes,
                        profile_id=profile.profile_id,
                        attempt=attempt,
                        rollouts=len(rewards),
                        updates=updates,
                        verified=verified,
                        pending_samples=len(samples),
                        checkpoint_id=checkpoint,
                        policy_version=int(getattr(self.student, "policy_version", 0)),
                        error=f"{type(exc).__name__}: {exc}",
                        emergency_checkpoint_error=emergency_error,
                        close_error=close_error,
                    )
                    raise
                finally:
                    # This is intentionally idempotent; the exception paths close
                    # before sleeping/re-raising so the emulator lease is released.
                    _close_environment(environment)

    if samples:
        update_in_progress = True
        try:
            checkpoint = str(self._update(samples))
            samples.clear()
            updates += 1
            update_in_progress = False
            progress.emit(
                "checkpoint",
                task_index=total_episodes,
                task_total=total_episodes,
                rollouts=len(rewards),
                updates=updates,
                verified=verified,
                pending_samples=0,
                checkpoint_id=checkpoint,
                policy_version=int(self.student.policy_version),
            )
        except (Exception, KeyboardInterrupt) as exc:
            emergency, did_update, emergency_error = _emergency_checkpoint(
                self, samples, update_in_progress=update_in_progress
            )
            if emergency:
                checkpoint = emergency
            updates += int(did_update)
            progress.emit(
                "crash",
                task_index=total_episodes,
                task_total=total_episodes,
                rollouts=len(rewards),
                updates=updates,
                verified=verified,
                pending_samples=len(samples),
                checkpoint_id=checkpoint,
                policy_version=int(getattr(self.student, "policy_version", 0)),
                error=f"{type(exc).__name__}: {exc}",
                emergency_checkpoint_error=emergency_error,
            )
            raise

    if self.coverage and checkpoint:
        for task in tasks:
            entry = self.coverage.entries.get(task.coverage_id)
            if entry and entry.status not in {
                entry.status.NON_ACTIONABLE,
                entry.status.BLOCKED,
            }:
                self.coverage.mark_student_update(task.coverage_id, checkpoint)
                same_os_profiles = {
                    profile.profile_id
                    for profile in matrix.select_for_os(task.os_family)
                }
                if same_os_profiles and same_os_profiles.issubset(
                    set(entry.emulator_profile_ids)
                ):
                    self.coverage.mark_validated(
                        task.coverage_id,
                        "all selected profiles reviewed and checkpointed",
                    )

    metrics = DistillationMetrics(
        len(rewards),
        updates,
        verified,
        float(np.mean(rewards)) if rewards else 0.0,
        float(np.mean(teacher_scores)) if teacher_scores else 0.0,
        checkpoint,
    )
    progress.emit(
        "complete",
        task_index=total_episodes,
        task_total=total_episodes,
        rollouts=metrics.rollouts,
        updates=metrics.updates,
        verified=metrics.verified_successes,
        pending_samples=0,
        checkpoint_id=metrics.checkpoint_id,
        mean_reward=metrics.mean_reward,
        mean_teacher_score=metrics.mean_teacher_score,
        policy_version=int(getattr(self.student, "policy_version", 0)),
    )
    return metrics


def install_resilient_training(runtime_module: object | None = None) -> None:
    """Patch ``OnPolicyDistiller.train`` once without changing its constructor."""
    if runtime_module is None:
        from . import on_policy as runtime_module
    if getattr(runtime_module, "_resilient_training_installed", False):
        return
    distiller = getattr(runtime_module, "OnPolicyDistiller", None)
    if distiller is None:
        raise RuntimeError("android_drlqn.on_policy lacks OnPolicyDistiller")
    if not hasattr(distiller, "_original_train"):
        setattr(distiller, "_original_train", distiller.train)
    distiller.train = _resilient_train
    setattr(runtime_module, "_resilient_training_installed", True)


__all__ = [
    "DEFAULT_PROGRESS_FILE",
    "DEFAULT_STATE_FILE",
    "ENV_EMULATOR_RETRIES",
    "ENV_EMULATOR_RETRY_DELAY",
    "ENV_PROGRESS_FILE",
    "ENV_QUIET",
    "ENV_STATE_FILE",
    "TrainingProgress",
    "install_resilient_training",
]
