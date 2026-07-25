#!/usr/bin/env python3
"""Run on-policy Android VLM/LLM distillation with local Ollama teachers."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from dataclasses import asdict
from pathlib import Path

from android_drlqn.distillation import (
    DistillationRunManifest,
    NumpyLinearStudent,
    NumpyPolicyUpdater,
    load_distillation_config,
)
from android_drlqn.document_curriculum import (
    CoverageLedger,
    DocumentationCurriculumBuilder,
    load_source_registry,
)
from android_drlqn.emulator_lab import load_emulator_matrix
from android_drlqn.ollama_runtime import OllamaRuntimeBundle
from android_drlqn.on_policy import OnPolicyDistiller


def load_plugin(spec: str):
    module_name, separator, factory_name = spec.partition(":")
    if not separator:
        raise ValueError("plugin format is module:factory")
    return getattr(importlib.import_module(module_name), factory_name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--documents", required=True, help="normalized ORM JSON array")
    parser.add_argument("--sources", default="config/document_sources.json")
    parser.add_argument("--matrix", default="config/emulator_matrix.json")
    parser.add_argument("--distillation-config", default="config/distillation.json")
    parser.add_argument("--ollama-config", default="config/ollama_models.json")
    parser.add_argument("--student-plugin", help="module:factory returning (student, updater)")
    parser.add_argument("--environment-plugin", help="module:factory(profile) -> emulator environment")
    parser.add_argument("--output", default="artifacts/ollama-distillation")
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
    actionable_tasks = tuple(task for task in tasks if coverage.entries[task.coverage_id].section.actionable)

    _, policy_config, student_config = load_distillation_config(args.distillation_config)
    bundle = OllamaRuntimeBundle.from_json(args.ollama_config)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    plan = {
        "teacher_profile": "ollama_local_ensemble",
        "teacher_id": bundle.teacher.teacher_id,
        "ollama_base_url": bundle.config.server.normalized_base_url,
        "sections": len(sections),
        "actionable_tasks": len(actionable_tasks),
        "emulator_profiles": [profile.profile_id for profile in matrix.profiles],
        "expanded_variants": matrix.report().expanded_variant_count,
        "student": dict(student_config),
    }
    (output / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return

    if not args.environment_plugin:
        raise SystemExit("real Ollama training requires --environment-plugin")
    statuses = bundle.preflight()
    (output / "ollama-models.json").write_text(
        json.dumps([status.__dict__ for status in statuses], indent=2), encoding="utf-8"
    )

    if args.student_plugin:
        student, updater = load_plugin(args.student_plugin)(student_config, output)
    else:
        student = NumpyLinearStudent(seed=policy_config.seed)
        updater = NumpyPolicyUpdater(student, checkpoint_dir=output / "checkpoints")
    environment_factory = load_plugin(args.environment_plugin)

    distiller = OnPolicyDistiller(
        student,
        bundle.teacher,
        updater,
        environment_factory,
        policy_config,
        coverage,
    )
    metrics = distiller.train(actionable_tasks, matrix)
    coverage.write_json(output / "coverage.json")

    manifest_seed = json.dumps(
        {
            "teacher": bundle.teacher.teacher_id,
            "student": dict(student_config),
            "profiles": [profile.profile_id for profile in matrix.profiles],
            "revisions": coverage.pinned_revisions,
            "seed": policy_config.seed,
        },
        sort_keys=True,
    )
    manifest = DistillationRunManifest(
        run_id=hashlib.sha256(manifest_seed.encode()).hexdigest()[:24],
        teacher_profile="ollama_local_ensemble",
        teacher_id=bundle.teacher.teacher_id,
        student_vlm=str(student_config.get("vlm") or "unknown"),
        student_llm=str(student_config.get("llm") or "unknown"),
        policy_config=asdict(policy_config),
        emulator_profiles=tuple(profile.profile_id for profile in matrix.profiles),
        document_revisions=dict(coverage.pinned_revisions),
        checkpoint_id=metrics.checkpoint_id,
    )
    manifest.write_json(output / "run.json")
    (output / "metrics.json").write_text(json.dumps(asdict(metrics), indent=2), encoding="utf-8")
    print(json.dumps(asdict(metrics), indent=2))


if __name__ == "__main__":
    main()
