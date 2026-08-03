"""MRPO-safe GUI-Owl student factory.

The base :mod:`project_gui_owl_train` plugin implements the model, interaction
format and multimodal losses. MRPO collects several trajectories from one
frozen policy before an update, so this adapter additionally guarantees that:

* each rollout owns immutable, content-addressed screenshot files;
* inference traces are keyed by the exact ``StudentProposal`` object rather
  than by task/profile/step, which would collide across grouped rollouts;
* consumed token-id traces are released after an optimizer update.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping, Sequence

from android_drlqn.distillation import DistillationSample, OnPolicyConfig, RolloutObservation
from android_drlqn.document_curriculum import CurriculumTask
from android_drlqn.emulator_lab import EmulatorProfile

import project_gui_owl_train as base

DEFAULT_MODEL_ID = base.DEFAULT_MODEL_ID


class MRPOSafeGUIOwlStudent(base.GUIOwlStudent):
    """GUI-Owl student with rollout-identity trace storage."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._mrpo_rollout_serial: dict[tuple[str, str], int] = {}
        self._mrpo_trace_by_proposal: dict[int, base.RolloutTrace] = {}

    @staticmethod
    def snapshot_observation(observation: RolloutObservation) -> RolloutObservation:
        """Copy a mutable lab screenshot to a content-addressed rollout path."""
        source = Path(observation.screenshot_reference)
        if not source.is_file():
            return observation
        payload = source.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()[:24]
        directory = source.parent / "gui-owl-rollouts"
        directory.mkdir(parents=True, exist_ok=True)
        suffix = source.suffix or ".png"
        target = directory / f"{digest}{suffix}"
        if not target.exists():
            temporary = target.with_suffix(target.suffix + ".tmp")
            temporary.write_bytes(payload)
            temporary.replace(target)
        return RolloutObservation(
            str(target),
            observation.visible_labels,
            observation.accessibility_summary,
            observation.package,
            observation.activity,
            observation.step_index,
        )

    def propose(
        self,
        task: CurriculumTask,
        profile: EmulatorProfile,
        observation: RolloutObservation,
    ):
        history_key = (task.task_id, profile.profile_id)
        if int(observation.step_index) == 0:
            # Each environment reset starts an independent grouped trajectory.
            self._history[history_key] = []
            self._mrpo_rollout_serial[history_key] = (
                self._mrpo_rollout_serial.get(history_key, 0) + 1
            )
        serial = self._mrpo_rollout_serial.get(history_key, 1)
        snapshot = self.snapshot_observation(observation)
        unique_observation = RolloutObservation(
            snapshot.screenshot_reference,
            snapshot.visible_labels,
            snapshot.accessibility_summary,
            snapshot.package,
            snapshot.activity,
            serial * 1_000_000 + int(snapshot.step_index),
        )
        proposal = super().propose(task, profile, unique_observation)
        key = self._key(task, profile, unique_observation, self.policy_version)
        trace = self._traces.pop(key, None)
        if trace is not None:
            self._mrpo_trace_by_proposal[id(proposal)] = trace
        return proposal

    def trace_for(self, sample: DistillationSample) -> base.RolloutTrace | None:
        return self._mrpo_trace_by_proposal.get(id(sample.proposal))

    def discard_trace(self, sample: DistillationSample) -> None:
        self._mrpo_trace_by_proposal.pop(id(sample.proposal), None)


class MRPOSafeGUIOwlLoraUpdater(base.GUIOwlLoraUpdater):
    """Release exact-token trace sidecars after each grouped update."""

    student: MRPOSafeGUIOwlStudent

    def update(
        self, samples: Sequence[DistillationSample], config: OnPolicyConfig
    ) -> str:
        try:
            return super().update(samples, config)
        finally:
            for sample in samples:
                self.student.discard_trace(sample)


def create_student(
    student_config: Mapping[str, object], output_dir: str | Path
) -> tuple[MRPOSafeGUIOwlStudent, MRPOSafeGUIOwlLoraUpdater]:
    """Default grouped-rollout student factory."""
    config = dict(student_config or {})
    model_id = str(config.get("model_id") or DEFAULT_MODEL_ID)
    checkpoint_dir = Path(output_dir) / "checkpoints"
    adapter = None
    if base._config_bool(config.get("resume"), True):
        configured = config.get("resume_checkpoint")
        adapter = (
            Path(str(configured)).expanduser()
            if configured
            else base._latest_checkpoint(checkpoint_dir)
        )
    lora = config.get("lora")
    student = MRPOSafeGUIOwlStudent(
        model_id,
        device=str(config["device"]) if config.get("device") else None,
        dtype=str(config["dtype"]) if config.get("dtype") else None,
        max_new_tokens=int(config.get("max_new_tokens", 192)),
        history_n=int(config.get("history_n", base.DEFAULT_HISTORY_N)),
        seed=int(config.get("seed", 0)),
        lora=lora if isinstance(lora, Mapping) else None,
        adapter_checkpoint=adapter,
    )
    updater = MRPOSafeGUIOwlLoraUpdater(
        student,
        checkpoint_dir,
        learning_rate=float(config.get("learning_rate", 2e-5)),
        grad_clip=float(config.get("grad_clip", 1.0)),
    )
    return student, updater


__all__ = [
    "DEFAULT_MODEL_ID",
    "MRPOSafeGUIOwlLoraUpdater",
    "MRPOSafeGUIOwlStudent",
    "create_student",
]
