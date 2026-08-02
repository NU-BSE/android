#!/usr/bin/env python3
"""Run OS-aware on-policy Android distillation with local Ollama teachers."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Mapping

# Allow repo-root plugin modules (project_train/project_lab) when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from android_drlqn.resilient_training import install_resilient_training

install_resilient_training()

from android_drlqn import ollama_runtime as _ollama_runtime
from android_drlqn.ollama_text_teacher import install_text_only_teacher

install_text_only_teacher(_ollama_runtime)

from android_drlqn.distillation import (
    DistillationMetrics,
    DistillationRunManifest,
    NumpyLinearStudent,
    NumpyPolicyUpdater,
)
from android_drlqn.document_curriculum import (
    CoverageLedger,
    CurriculumTask,
    DocumentationCurriculumBuilder,
)
from android_drlqn.model_launch import (
    DistillationMode,
    ModelLaunchBundle,
    aggregate_distillation_metrics,
)
from android_drlqn.on_policy import OnPolicyDistiller


def load_plugin(spec: str):
    module_name, separator, factory_name = spec.partition(":")
    if not separator:
        raise ValueError("plugin format is module:factory")
    return getattr(importlib.import_module(module_name), factory_name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--documents", required=True, help="normalized ORM JSON array")
    parser.add_argument("--launch-config", default="config/model_launch.json")
    parser.add_argument("--sources", help="override source registry from launch config")
    parser.add_argument("--matrix", help="override emulator matrix from launch config")
    parser.add_argument("--distillation-config", help="override distillation config")
    parser.add_argument("--ollama-config", help="override Ollama config")
    parser.add_argument("--student-plugin", help="module:factory returning (student, updater)")
    parser.add_argument(
        "--environment-plugin",
        help="module:factory(profile[, os_settings_context]) -> emulator environment",
    )
    parser.add_argument("--output", default="artifacts/ollama-distillation")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    documents = json.loads(Path(args.documents).read_text(encoding="utf-8"))
    if not isinstance(documents, list):
        raise SystemExit("--documents must contain a JSON array")

    launch = ModelLaunchBundle.from_json(
        args.launch_config,
        args.documents,
        ollama_config=args.ollama_config,
        distillation_config=args.distillation_config,
        source_registry=args.sources,
        emulator_matrix=args.matrix,
    )
    sections, tasks = DocumentationCurriculumBuilder(
        launch.sources,
        max_evidence_tier=launch.config.max_evidence_tier,
    ).build(documents)
    coverage = CoverageLedger(launch.sources)
    for section in sections:
        coverage.register_section(section)
        coverage.register_source_revision(section.source_key, section.snapshot_sha256)
    for task in tasks:
        if coverage.entries[task.coverage_id].section.actionable:
            coverage.link_task(task.coverage_id, task.task_id)
    actionable_tasks = tuple(
        task for task in tasks if coverage.entries[task.coverage_id].section.actionable
    )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    plan = dict(launch.plan())
    plan.update(
        {
            "sections": len(sections),
            "actionable_tasks": len(actionable_tasks),
            "expanded_variants": launch.matrix.report().expanded_variant_count,
        }
    )
    (output / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return

    if not args.environment_plugin:
        raise SystemExit("real Ollama training requires --environment-plugin")

    student_config = dict(launch.student_config)
    student_config["os_launch"] = plan["os_families"]
    if args.student_plugin:
        student, updater = load_plugin(args.student_plugin)(student_config, output)
    else:
        student = NumpyLinearStudent(seed=launch.base_policy.seed)
        updater = NumpyPolicyUpdater(student, checkpoint_dir=output / "checkpoints")

    raw_environment_factory = load_plugin(args.environment_plugin)
    metrics_by_os: dict[str, DistillationMetrics] = {}
    model_statuses: dict[str, object] = {}
    teacher_ids: list[str] = []
    selected_profile_ids: list[str] = []

    for os_family in launch.os_families:
        os_launch = launch.config.for_os(os_family)
        os_tasks = tuple(task for task in actionable_tasks if task.os_family == os_family)
        if not os_tasks:
            continue
        if os_launch.distillation_mode is DistillationMode.DISABLED:
            _block_tasks(coverage, os_tasks, "distillation disabled by model launch policy")
            continue

        os_matrix = launch.matrix_for(os_family)
        if not os_matrix.profiles:
            if os_launch.distillation_mode is DistillationMode.REQUIRED:
                raise RuntimeError(f"required OS has no rollout profiles: {os_family}")
            _block_tasks(coverage, os_tasks, "no materialized emulator profile configured")
            continue

        enriched_tasks = launch.enrich_tasks(os_tasks, os_family)
        runtime = launch.runtime_for(os_family)
        statuses = runtime.preflight()
        model_statuses[os_family] = [status.__dict__ for status in statuses]
        teacher_ids.append(runtime.teacher.teacher_id)
        selected_profile_ids.extend(profile.profile_id for profile in os_matrix.profiles)
        os_settings_context = launch.settings.context(
            os_family,
            setting_id=None,
            limit=os_launch.max_settings_context,
        )
        environment_factory = _environment_adapter(
            raw_environment_factory,
            os_settings_context,
        )
        distiller = OnPolicyDistiller(
            student,
            runtime.teacher,
            updater,
            environment_factory,
            launch.policy_for(os_family),
            coverage,
        )
        metrics_by_os[os_family] = distiller.train(enriched_tasks, os_matrix)

    (output / "ollama-models.json").write_text(
        json.dumps(model_statuses, indent=2), encoding="utf-8"
    )
    coverage.write_json(output / "coverage.json")
    metrics = aggregate_distillation_metrics(metrics_by_os)
    (output / "metrics-by-os.json").write_text(
        json.dumps(
            {family: asdict(value) for family, value in metrics_by_os.items()},
            indent=2,
        ),
        encoding="utf-8",
    )

    policy_by_os = {
        family: asdict(launch.policy_for(family))
        for family in launch.os_families
        if launch.config.for_os(family).distillation_mode is not DistillationMode.DISABLED
    }
    teacher_id = "|".join(sorted(set(teacher_ids))) if teacher_ids else "ollama-local"
    manifest_seed = json.dumps(
        {
            "teacher": teacher_id,
            "student": student_config,
            "profiles": selected_profile_ids,
            "revisions": coverage.pinned_revisions,
            "policies": policy_by_os,
        },
        sort_keys=True,
    )
    manifest = DistillationRunManifest(
        run_id=hashlib.sha256(manifest_seed.encode()).hexdigest()[:24],
        teacher_profile="ollama_local_ensemble",
        teacher_id=teacher_id,
        student_vlm=str(student_config.get("vlm") or "unknown"),
        student_llm=str(student_config.get("llm") or "unknown"),
        policy_config=policy_by_os,
        emulator_profiles=tuple(selected_profile_ids),
        document_revisions=dict(coverage.pinned_revisions),
        checkpoint_id=metrics.checkpoint_id,
    )
    manifest.write_json(output / "run.json")
    (output / "metrics.json").write_text(
        json.dumps(asdict(metrics), indent=2), encoding="utf-8"
    )
    print(json.dumps(asdict(metrics), indent=2))


def _environment_adapter(
    factory: Callable[..., object],
    os_settings_context: Mapping[str, object],
) -> Callable[[object], object]:
    signature = inspect.signature(factory)
    positional = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind
        in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
    ]
    accepts_context = any(
        parameter.kind is inspect.Parameter.VAR_POSITIONAL
        for parameter in signature.parameters.values()
    ) or len(positional) >= 2

    def create(profile: object) -> object:
        if accepts_context:
            return factory(profile, os_settings_context)
        return factory(profile)

    return create


def _block_tasks(
    coverage: CoverageLedger,
    tasks: tuple[CurriculumTask, ...],
    reason: str,
) -> None:
    for task in tasks:
        entry = coverage.entries.get(task.coverage_id)
        if entry is not None:
            coverage.mark_blocked(task.coverage_id, reason)


if __name__ == "__main__":
    main()

