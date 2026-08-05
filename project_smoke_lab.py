"""Deterministic screenshot environment for GUI-Owl pipeline smoke tests.

This plugin does not use ADB and does not represent real OS navigation. It exists
only to validate model loading, multimodal preprocessing, teacher calls, rollout
collection, LoRA updates and checkpoint persistence when genuine custom AVDs are
not installed.
"""
from __future__ import annotations

import hashlib
import textwrap
from pathlib import Path
from typing import Mapping, Sequence

from android_drlqn.distillation import (
    EnvironmentOutcome,
    ProposalKind,
    RolloutObservation,
    StudentProposal,
)
from android_drlqn.document_curriculum import CurriculumTask
from android_drlqn.emulator_lab import EmulatorProfile

ARTIFACT_DIR = Path("artifacts/smoke-environment")


def _strings(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        text = value.strip()
        return (text,) if text else ()
    if isinstance(value, Sequence) and not isinstance(
        value, (bytes, bytearray, str)
    ):
        result = []
        for item in value:
            text = str(item).strip()
            if text and text not in result:
                result.append(text)
        return tuple(result)
    text = str(value).strip()
    return (text,) if text else ()


def _first(mapping: Mapping[str, object], *keys: str) -> str | None:
    for key in keys:
        value = mapping.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _unique(values: Sequence[str]) -> tuple[str, ...]:
    result: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in result:
            result.append(text)
    return tuple(result)


def _write_screenshot(path: Path, title: str, labels: Sequence[str]) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise RuntimeError(
            "project_smoke_lab requires Pillow: python -m pip install pillow"
        ) from exc

    width, height = 1080, 1920
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    draw.rectangle((0, 0, width, 150), fill=(245, 245, 245))
    draw.text((48, 55), title, fill="black", font=font)

    y = 220
    for label in labels[:12]:
        draw.rounded_rectangle(
            (48, y, width - 48, y + 120),
            radius=18,
            fill=(248, 248, 248),
            outline=(190, 190, 190),
            width=2,
        )
        wrapped = textwrap.wrap(label, width=65) or [label]
        draw.text((82, y + 42), wrapped[0], fill="black", font=font)
        y += 145

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")


class DeterministicCurriculumEnvironment:
    def __init__(
        self,
        profile: EmulatorProfile,
        os_settings_context: object = None,
        artifact_dir: str | Path = ARTIFACT_DIR,
    ) -> None:
        self.profile = profile
        self.os_settings_context = os_settings_context
        self.artifact_dir = Path(artifact_dir)
        self.task: CurriculumTask | None = None
        self.expected_label: str | None = None
        self.observation: RolloutObservation | None = None

    def reset(
        self,
        task: CurriculumTask,
        profile: EmulatorProfile,
    ) -> RolloutObservation:
        if profile.profile_id != self.profile.profile_id:
            raise ValueError("environment profile does not match factory profile")

        expected = (
            dict(task.expected_output)
            if isinstance(task.expected_output, Mapping)
            else {}
        )
        evidence = dict(task.evidence) if isinstance(task.evidence, Mapping) else {}

        ui_path = _strings(expected.get("ui_path"))
        if not ui_path:
            ui_path = _strings(evidence.get("ui_path"))

        labels = _unique(
            [
                *_strings(expected.get("display_name")),
                *_strings(expected.get("target_label")),
                *_strings(expected.get("label")),
                *ui_path[-3:],
            ]
        )
        if not labels:
            kind = getattr(getattr(task, "kind", None), "value", "setting")
            labels = ("Settings", str(kind).replace("_", " ").title())

        self.task = task
        self.expected_label = (
            _first(expected, "target_label", "display_name", "label")
            or (ui_path[-1] if ui_path else None)
        )

        task_id = str(getattr(task, "task_id", "task"))
        safe_task = "".join(
            character if character.isalnum() or character in "._-" else "-"
            for character in task_id
        ).strip("-")[:48] or "task"
        digest = hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:8]
        screenshot = (
            self.artifact_dir
            / str(profile.os_family)
            / str(profile.profile_id)
            / f"{safe_task}-{digest}.png"
        )
        _write_screenshot(
            screenshot,
            f"SMOKE · {profile.os_family} · {profile.profile_id}",
            labels,
        )

        self.observation = RolloutObservation(
            screenshot_reference=screenshot.as_posix(),
            visible_labels=labels,
            accessibility_summary={
                "mode": "deterministic-smoke-test",
                "not_real_navigation": True,
                "task_id": task_id,
                "os_family": profile.os_family,
                "os_settings_context": self.os_settings_context,
            },
            package="com.android.settings",
            activity=".Settings",
            step_index=0,
        )
        return self.observation

    def step(self, proposal: StudentProposal) -> EnvironmentOutcome:
        if self.task is None or self.observation is None:
            raise RuntimeError("reset() must be called before step()")

        expected = (self.expected_label or "").casefold()
        proposed = (proposal.target_label or "").casefold()
        label_match = bool(expected and proposed and expected == proposed)
        explanatory = proposal.kind in {
            ProposalKind.EXPLAIN,
            ProposalKind.ASK_CONFIRMATION,
        }
        verified = label_match or (not expected and explanatory)

        return EnvironmentOutcome(
            reward=1.0 if verified else -0.1,
            terminal=True,
            verified=verified,
            next_observation=None,
            reason=(
                "synthetic expectation matched"
                if verified
                else "synthetic expectation did not match"
            ),
        )

    def close(self) -> None:
        return None


def create_environment(
    profile: EmulatorProfile,
    os_settings_context: object = None,
) -> DeterministicCurriculumEnvironment:
    return DeterministicCurriculumEnvironment(
        profile=profile,
        os_settings_context=os_settings_context,
    )
