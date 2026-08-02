from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from android_drlqn.resilient_training import _resilient_train


class FakeStudent:
    def __init__(self) -> None:
        self.policy_version = 0

    def propose(self, task, profile, observation):
        del task, profile, observation
        return SimpleNamespace(policy_version=self.policy_version)

    def save(self, path: str | Path) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"weights")
        return "emergency-weights"


class FakeUpdater:
    def __init__(self, student: FakeStudent, checkpoint_dir: Path) -> None:
        self.student = student
        self.checkpoint_dir = checkpoint_dir
        self.step_count = 0
        self.batch_sizes: list[int] = []

    def update(self, samples, config):
        del config
        self.batch_sizes.append(len(samples))
        self.step_count += 1
        self.student.policy_version += 1
        return f"checkpoint-{self.step_count}"


class FakeTeacher:
    teacher_id = "teacher"

    def evaluate(self, task, profile, observation, proposal, outcome):
        del task, profile, observation, proposal, outcome
        return SimpleNamespace(
            confidence=1.0,
            total_score=0.8,
            safety_score=1.0,
            teacher_id=self.teacher_id,
            should_update=True,
            target=SimpleNamespace(),
        )


class FakeMatrix:
    def __init__(self, profile) -> None:
        self.profile = profile

    def select_for_os(self, os_family):
        del os_family
        return [self.profile]


class FakeDistiller:
    def __init__(self, tmp_path: Path, environment_factory) -> None:
        self.student = FakeStudent()
        self.updater = FakeUpdater(self.student, tmp_path / "checkpoints")
        self.teacher = FakeTeacher()
        self.environment_factory = environment_factory
        self.coverage = None
        self.config = SimpleNamespace(
            max_steps_per_task=3,
            max_policy_lag=0,
            min_teacher_confidence=0.5,
            rollout_batch_size=16,
            updates_per_batch=1,
            gamma=0.98,
            advantage_clip=2.0,
        )

    def _advantages(self, rewards):
        return tuple(0.0 for _ in rewards)

    def _update(self, samples):
        return self.updater.update(samples, self.config)


def _configure_progress(monkeypatch, tmp_path: Path) -> Path:
    progress = tmp_path / "progress.jsonl"
    monkeypatch.setenv("ANDROID_DRLQN_TRAINING_PROGRESS_FILE", str(progress))
    monkeypatch.setenv(
        "ANDROID_DRLQN_TRAINING_STATE_FILE", str(tmp_path / "state.json")
    )
    monkeypatch.setenv("ANDROID_DRLQN_TRAINING_QUIET", "1")
    monkeypatch.setenv("ANDROID_DRLQN_EMULATOR_RETRY_DELAY_S", "0")
    return progress


def _task_and_matrix():
    task = SimpleNamespace(task_id="task", os_family="android-aosp", coverage_id="c")
    profile = SimpleNamespace(profile_id="aosp10-low-phone", os_family="android-aosp")
    return task, FakeMatrix(profile)


def test_retries_when_adb_device_disappears(monkeypatch, tmp_path):
    progress_path = _configure_progress(monkeypatch, tmp_path)
    monkeypatch.setenv("ANDROID_DRLQN_EMULATOR_RETRIES", "1")
    calls = {"steps": 0}

    class Environment:
        def reset(self, task, profile):
            del task, profile
            return SimpleNamespace(step_index=0)

        def step(self, proposal):
            del proposal
            calls["steps"] += 1
            if calls["steps"] == 1:
                raise RuntimeError("adb: device 'emulator-5554' not found")
            return SimpleNamespace(
                reward=1.0,
                terminal=True,
                verified=True,
                next_observation=None,
                reason="ok",
            )

        def close(self):
            return None

    distiller = FakeDistiller(tmp_path, lambda profile: Environment())
    task, matrix = _task_and_matrix()

    metrics = _resilient_train(distiller, [task], matrix)

    assert calls["steps"] == 2
    assert metrics.rollouts == 1
    assert metrics.updates == 1
    assert metrics.checkpoint_id == "checkpoint-1"
    events = [json.loads(line)["event"] for line in progress_path.read_text().splitlines()]
    assert "retry" in events
    assert events[-1] == "complete"


def test_flushes_scored_partial_trajectory_before_retry(monkeypatch, tmp_path):
    progress_path = _configure_progress(monkeypatch, tmp_path)
    monkeypatch.setenv("ANDROID_DRLQN_EMULATOR_RETRIES", "1")
    calls = {"steps": 0}

    class Environment:
        def reset(self, task, profile):
            del task, profile
            return SimpleNamespace(step_index=0)

        def step(self, proposal):
            del proposal
            calls["steps"] += 1
            if calls["steps"] == 1:
                return SimpleNamespace(
                    reward=0.2,
                    terminal=False,
                    verified=False,
                    next_observation=SimpleNamespace(step_index=1),
                    reason="continue",
                )
            if calls["steps"] == 2:
                raise RuntimeError("adb: device 'emulator-5554' not found")
            return SimpleNamespace(
                reward=1.0,
                terminal=True,
                verified=True,
                next_observation=None,
                reason="ok",
            )

        def close(self):
            return None

    distiller = FakeDistiller(tmp_path, lambda profile: Environment())
    task, matrix = _task_and_matrix()

    metrics = _resilient_train(distiller, [task], matrix)

    assert distiller.updater.batch_sizes == [1, 1]
    assert metrics.updates == 2
    assert metrics.checkpoint_id == "checkpoint-2"
    events = [json.loads(line) for line in progress_path.read_text().splitlines()]
    retry = next(event for event in events if event["event"] == "retry")
    assert retry["checkpoint_id"] == "checkpoint-1"
    assert events[-1]["event"] == "complete"
