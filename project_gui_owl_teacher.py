"""Local GUI-Owl teacher for emulator rollouts.

`config/distillation.json` configures `flux3_early_access` with
`early_access_acknowledged: false` and a null endpoint, so the only teachers the
training entrypoint can construct are an HTTP service and a replay fixture that
returns confidence 1.0 for everything. Neither can grade a rollout.

This teacher runs a local multimodal checkpoint over the same observation the
student saw and returns its own action as the correction target. It is a
drop-in for `StrictJsonTeacher`: same `teacher_id` property, same `evaluate`
signature, same `TeacherJudgement`.

Unlike the HTTP teacher it sees the screenshot rather than a label list, and it
never leaves the host.

**Its reliability is the reference model's measured reliability, and that is not
uniform across actions.** For the adapter-merged GUI-Owl-2B on this corpus:
click 0.860, swipe 0.867, type 0.875, wait 1.000, long_press 0.200, and
`system_button` and `answer` both **0.000**. A judgement about a language turn
from a teacher that never emits `mobile.answer` carries no information, so
`--language-confidence` defaults to 0.0 and those samples are dropped by
`min_teacher_confidence` rather than being learned from a guess.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Mapping

from android_drlqn.distillation import (
    EnvironmentOutcome,
    ProposalKind,
    RolloutObservation,
    StudentProposal,
    TeacherJudgement,
    TeacherTarget,
)
from android_drlqn.document_curriculum import CurriculumTask
from android_drlqn.emulator_lab import EmulatorProfile

import project_gui_owl_train as base

DEFAULT_TEACHER_MODEL = "artifacts/teachers/gui-owl-2b-merged"

# The proposal API carries no press duration, so parse_gui_owl_output folds
# long_press into TAP. A teacher must not be stricter than the action space it
# grades against.
_FORBIDDEN = ("adb shell", "input tap", "dispatchgesture", "performaction", "su -c")


def _bbox_centre(bbox: tuple[float, float, float, float] | None) -> tuple[float, float] | None:
    if bbox is None:
        return None
    x0, y0, x1, y1 = bbox
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


class LocalGuiOwlTeacher:
    """Grade a student proposal against a local reference model's own action."""

    def __init__(
        self,
        model_id: str = DEFAULT_TEACHER_MODEL,
        *,
        device: str | None = None,
        dtype: str | None = None,
        max_new_tokens: int = 96,
        history_n: int = base.DEFAULT_HISTORY_N,
        teacher_id: str = "local-gui-owl-2b",
        grounding_tolerance: float = 0.10,
        language_confidence: float = 0.0,
    ) -> None:
        self._teacher_id = teacher_id
        self.grounding_tolerance = float(grounding_tolerance)
        self.language_confidence = float(language_confidence)
        self._reference = base.GUIOwlStudent(
            model_id,
            device=device,
            dtype=dtype,
            max_new_tokens=max_new_tokens,
            history_n=history_n,
        )
        # A freshly initialised LoRA is the identity, but grading must not
        # depend on that: take the reference out of training mode outright.
        self._reference.model.eval()

    @property
    def teacher_id(self) -> str:
        return self._teacher_id

    def evaluate(
        self,
        task: CurriculumTask,
        profile: EmulatorProfile,
        observation: RolloutObservation,
        proposal: StudentProposal,
        outcome: EnvironmentOutcome,
    ) -> TeacherJudgement:
        reference = self._reference.propose(task, profile, observation)
        # An unparseable generation degrades to WAIT inside `propose`. Grading
        # against that would endorse whatever the student did whenever the
        # student also waited, so the sample is dropped instead: confidence 0
        # falls below `min_teacher_confidence` and the updater discards it.
        if reference.message == base.UNPARSEABLE_FALLBACK_MESSAGE:
            return TeacherJudgement(
                teacher_id=self._teacher_id,
                confidence=0.0,
                total_score=0.0,
                grounding_score=0.0,
                planning_score=0.0,
                language_score=0.0,
                safety_score=1.0,
                should_update=False,
                target=TeacherTarget(preferred_kind=proposal.kind),
                tags=("local-reference", "reference-unparseable"),
            )
        target = TeacherTarget(
            preferred_kind=reference.kind,
            target_label=reference.target_label,
            bbox_norm=reference.bbox_norm,
            response=reference.message,
        )

        kind_match = proposal.kind is reference.kind
        planning = 1.0 if kind_match else 0.0

        grounding = float(kind_match)
        if kind_match and proposal.kind is ProposalKind.TAP:
            here, there = _bbox_centre(proposal.bbox_norm), _bbox_centre(reference.bbox_norm)
            if here is None or there is None:
                grounding = 0.5
            else:
                distance = max(abs(here[0] - there[0]), abs(here[1] - there[1]))
                grounding = 1.0 if distance <= self.grounding_tolerance else 0.0

        message = str(proposal.message or "").casefold()
        safety = 0.0 if any(marker in message for marker in _FORBIDDEN) else 1.0
        language = 1.0 if str(proposal.message or "").strip() else 0.0

        # The environment already knows whether a non-language action worked.
        # Where it does, that verdict outranks the reference model's opinion.
        if outcome.verified:
            planning = grounding = 1.0

        language_turn = proposal.kind in (ProposalKind.EXPLAIN, ProposalKind.ASK_CONFIRMATION)
        confidence = self.language_confidence if language_turn else 1.0
        total = (grounding + planning + language + safety) / 4.0
        return TeacherJudgement(
            teacher_id=self._teacher_id,
            confidence=confidence,
            total_score=total,
            grounding_score=grounding,
            planning_score=planning,
            language_score=language,
            safety_score=safety,
            should_update=not (kind_match and grounding >= 1.0),
            target=target,
            tags=("local-reference", "verified" if outcome.verified else "unverified"),
        )


def create_teacher(config: Mapping[str, object] | None = None) -> LocalGuiOwlTeacher:
    """`module:factory` entry point mirroring the student plugin convention."""
    values = dict(config or {})
    return LocalGuiOwlTeacher(
        str(values.get("model_id") or os.environ.get("GUI_OWL_TEACHER_MODEL") or DEFAULT_TEACHER_MODEL),
        device=str(values["device"]) if values.get("device") else None,
        dtype=str(values["dtype"]) if values.get("dtype") else None,
        max_new_tokens=int(values.get("max_new_tokens", 96)),
        teacher_id=str(values.get("teacher_id") or "local-gui-owl-2b"),
        grounding_tolerance=float(values.get("grounding_tolerance", 0.10)),
        language_confidence=float(values.get("language_confidence", 0.0)),
    )


__all__ = ["DEFAULT_TEACHER_MODEL", "LocalGuiOwlTeacher", "create_teacher"]
