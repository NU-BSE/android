#!/usr/bin/env python3
"""Build a distillation plan or run grouped fixture/plugin-backed training."""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import urllib.request
from pathlib import Path

# Allow repo-root plugin modules (project_train/project_lab) when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from android_drlqn.distillation import (
    DistillationRunManifest,
    EnvironmentOutcome,
    NumpyLinearStudent,
    NumpyPolicyUpdater,
    ProposalKind,
    ReplayTeacher,
    RolloutObservation,
    StrictJsonTeacher,
    TeacherJudgement,
    TeacherProfile,
    TeacherTarget,
    load_distillation_config,
)
from android_drlqn.document_curriculum import (
    CoverageLedger,
    DocumentationCurriculumBuilder,
    load_source_registry,
)
from android_drlqn.emulator_lab import load_emulator_matrix
from android_drlqn.on_policy import OnPolicyDistiller

DEFAULT_REAL_STUDENT_PLUGIN = "project_gui_owl_mrpo_train:create_student"


class FixtureEnvironment:
    """Deterministic environment for CI/replay fixtures, not an Android driver."""

    def __init__(self, profile, fixture):
        self.profile = profile
        self.fixture = fixture
        self.index = 0

    def reset(self, task, profile):
        self.index = 0
        return self._observation(self.fixture.get("initial") or {}, 0)

    def step(self, proposal):
        steps = self.fixture.get("steps") or ()
        raw = (
            steps[min(self.index, len(steps) - 1)]
            if steps
            else {"terminal": True, "verified": True, "reward": 1.0}
        )
        self.index += 1
        next_raw = raw.get("next")
        return EnvironmentOutcome(
            reward=float(raw.get("reward", 0.0)),
            terminal=bool(raw.get("terminal", True)),
            verified=bool(raw.get("verified", False)),
            next_observation=(
                self._observation(next_raw, self.index)
                if isinstance(next_raw, dict)
                else None
            ),
            reason=str(raw.get("reason") or ""),
        )

    @staticmethod
    def _observation(raw, index):
        return RolloutObservation(
            screenshot_reference=str(
                raw.get("screenshot_reference") or f"fixture://screen/{index}"
            ),
            visible_labels=tuple(str(value) for value in raw.get("visible_labels") or ()),
            accessibility_summary=dict(raw.get("accessibility_summary") or {}),
            package=str(raw.get("package") or "com.android.settings"),
            activity=str(raw.get("activity") or ".Settings"),
            step_index=index,
        )


class HttpTransport:
    def __call__(self, endpoint, headers, payload):
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode(),
            headers=dict(headers),
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            value = json.loads(response.read().decode())
        if not isinstance(value, dict):
            raise ValueError("teacher response must be a JSON object")
        return value


def load_plugin(spec):
    module_name, separator, factory_name = spec.partition(":")
    if not separator:
        raise ValueError("plugin format is module:factory")
    return getattr(importlib.import_module(module_name), factory_name)


def replay_teacher(tasks, matrix):
    judgements = {}
    for task in tasks:
        for profile in matrix.select_for_os(task.os_family):
            key = f"{task.task_id}:{profile.profile_id}:0"
            judgements[key] = TeacherJudgement(
                "replay-teacher",
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                True,
                TeacherTarget(
                    preferred_kind=ProposalKind.TAP,
                    target_label=str(task.expected_output.get("display_name") or "")
                    or None,
                    response=str(
                        task.expected_output.get("display_name") or task.prompt
                    ),
                ),
                ("fixture",),
            )
    return ReplayTeacher(judgements)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--documents", required=True, help="normalized ORM JSON array")
    parser.add_argument("--sources", default="config/document_sources.json")
    parser.add_argument("--matrix", default="config/emulator_matrix.json")
    parser.add_argument("--config", default="config/distillation.json")
    parser.add_argument("--fixture", help="fixture environment JSON")
    parser.add_argument(
        "--student-plugin",
        help=(
            "module:factory returning (student, updater); real environment runs "
            "default to the MRPO-safe GUI-Owl-1.5-2B-Instruct plugin"
        ),
    )
    parser.add_argument(
        "--environment-plugin", help="module:factory(profile) -> environment"
    )
    parser.add_argument(
        "--teacher-plugin",
        help=(
            "module:factory(config) -> teacher, overriding the configured "
            "profile; project_gui_owl_teacher:create_teacher grades rollouts "
            "with a local checkpoint and no network"
        ),
    )
    parser.add_argument("--mrpo-group-size", type=int, default=2)
    parser.add_argument("--mrpo-oversample-factor", type=int, default=2)
    parser.add_argument("--checkpoint-reward-weight", type=float, default=0.5)
    parser.add_argument("--output", default="artifacts/distillation")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    documents = json.loads(Path(args.documents).read_text(encoding="utf-8"))
    if not isinstance(documents, list):
        raise SystemExit("--documents must contain a JSON array")
    sources = load_source_registry(args.sources)
    matrix = load_emulator_matrix(args.matrix)
    matrix.require_complete()
    sections, tasks = DocumentationCurriculumBuilder(sources).build(documents)
    coverage = CoverageLedger(sources)
    for section in sections:
        coverage.register_section(section)
        coverage.register_source_revision(section.source_key, section.snapshot_sha256)
    for task in tasks:
        if coverage.entries[task.coverage_id].section.actionable:
            coverage.link_task(task.coverage_id, task.task_id)
    actionable_tasks = tuple(
        task for task in tasks if coverage.entries[task.coverage_id].section.actionable
    )
    teacher_config, policy_config, student_config = load_distillation_config(args.config)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    resolved_student_plugin = args.student_plugin or (
        DEFAULT_REAL_STUDENT_PLUGIN if args.environment_plugin else None
    )
    plan = {
        "sections": len(sections),
        "actionable_tasks": len(actionable_tasks),
        "emulator_profiles": [profile.profile_id for profile in matrix.profiles],
        "expanded_variants": matrix.report().expanded_variant_count,
        "teacher": teacher_config.profile.value,
        "teacher_network_ready": (
            teacher_config.profile is not TeacherProfile.FLUX3_EARLY_ACCESS
            or teacher_config.early_access_acknowledged
        ),
        "student_plugin": resolved_student_plugin or "numpy-ci-student",
        "teacher_plugin": args.teacher_plugin or "",
        "mrpo": {
            "group_size": args.mrpo_group_size,
            "oversample_factor": args.mrpo_oversample_factor,
            "checkpoint_reward_weight": args.checkpoint_reward_weight,
        },
    }
    (output / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return

    if resolved_student_plugin:
        student, updater = load_plugin(resolved_student_plugin)(student_config, output)
    else:
        # Keep deterministic NumPy weights for fixture-only CI/replay runs.
        student = NumpyLinearStudent(seed=policy_config.seed)
        updater = NumpyPolicyUpdater(student, checkpoint_dir=output / "checkpoints")

    fixture = json.loads(Path(args.fixture).read_text()) if args.fixture else {}
    if args.environment_plugin:
        environment_factory = load_plugin(args.environment_plugin)
    elif args.fixture:
        environment_factory = lambda profile: FixtureEnvironment(profile, fixture)
    else:
        raise SystemExit(
            "real training requires --environment-plugin; CI replay requires --fixture"
        )

    if args.teacher_plugin:
        # A local teacher outranks the configured profile: the shipped default
        # is flux3_early_access with early_access_acknowledged false and a null
        # endpoint, and the replay teacher returns confidence 1.0 for every
        # rollout, so neither can grade one.
        teacher = load_plugin(args.teacher_plugin)(student_config)
    elif teacher_config.profile is TeacherProfile.REPLAY:
        teacher = replay_teacher(actionable_tasks, matrix)
    else:
        teacher = StrictJsonTeacher(teacher_config, HttpTransport())
    metrics = OnPolicyDistiller(
        student,
        teacher,
        updater,
        environment_factory,
        policy_config,
        coverage,
        group_size=args.mrpo_group_size,
        oversample_factor=args.mrpo_oversample_factor,
        checkpoint_reward_weight=args.checkpoint_reward_weight,
    ).train(actionable_tasks, matrix)
    coverage.write_json(output / "coverage.json")
    manifest = DistillationRunManifest.create(
        teacher_config,
        str(student_config.get("vlm") or "unknown"),
        str(student_config.get("llm") or "unknown"),
        policy_config,
        matrix,
        coverage.pinned_revisions,
        metrics.checkpoint_id,
    )
    manifest.write_json(output / "run.json")
    (output / "metrics.json").write_text(
        json.dumps(metrics.__dict__, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics.__dict__, indent=2))


if __name__ == "__main__":
    main()
