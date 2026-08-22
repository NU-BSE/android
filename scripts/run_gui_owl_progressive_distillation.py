#!/usr/bin/env python3
"""Run calibrated progressive GUI-Owl compression for one target size.

Every destructive architecture change is followed by gated recovery. The runner
also normalizes calibration head selections so Qwen3-VL grouped-query attention
keeps its original query/KV group relationships during 0.5B width pruning.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence

from android_drlqn.gui_owl_student import load_student_plan, text_layer_selection_is_safe

# Support both documented direct execution (`python scripts/...py`) and package
# imports used by tests (`from scripts.run_... import ...`). When a file inside
# scripts/ is executed directly, Python puts scripts/ rather than the repo root
# at sys.path[0], so an absolute `scripts.normalize_...` import cannot resolve.
if __package__:
    from .normalize_gui_owl_calibration import normalize_manifest
else:
    from normalize_gui_owl_calibration import normalize_manifest


RECOVERY_TRAINER = "scripts/distill_gui_owl_grounded.py"
# Policy stages keep the K3/first-token repair entrypoint. Structural recovery
# uses the base losses but must share the same action/coordinate correctness
# evaluator, so it runs through the grounded wrapper rather than the bare base
# trainer. That prevents a syntactically valid center-tap policy from becoming
# a structural predecessor.
POLICY_TRAINER = "scripts/distill_gui_owl_policy_aligned.py"


def _student_entry(plan: Mapping[str, object], name: str) -> dict[str, object]:
    students = plan.get("students")
    if not isinstance(students, Mapping) or not isinstance(students.get(name), Mapping):
        raise ValueError(f"student plan does not define {name}")
    return dict(students[name])


def _progressive_entry(plan: Mapping[str, object], name: str) -> dict[str, object]:
    value = plan.get("progressive")
    if not isinstance(value, Mapping) or not isinstance(value.get(name), Mapping):
        raise ValueError(f"student plan lacks progressive configuration for {name}")
    return dict(value[name])


def _tail_protected_lineage(path: Path) -> bool:
    """Return whether stored structural lineage satisfies current tail/coverage safety."""

    lineage: Mapping[str, object] | None = None
    for name in ("compression-lineage.json", "student-build-manifest.json"):
        candidate = path / name
        if not candidate.is_file():
            continue
        value = json.loads(candidate.read_text(encoding="utf-8"))
        if isinstance(value, Mapping):
            lineage = value
            break
    if lineage is None:
        return False
    try:
        source = tuple(int(value) for value in lineage.get("source_text_original_layer_ids") or ())
        selected = tuple(int(value) for value in lineage.get("text_original_layer_ids") or ())
    except (TypeError, ValueError):
        return False
    return text_layer_selection_is_safe(source, selected)


def _gui_correctness_measured(manifest: Mapping[str, object]) -> bool:
    final_evaluation = manifest.get("final_evaluation")
    if not isinstance(final_evaluation, Mapping):
        return False
    required = (
        "gui_action_name_match_rate",
        "gui_action_records",
        "gui_median_coordinate_error",
        "gui_coordinate_records",
    )
    if any(name not in final_evaluation for name in required):
        return False
    try:
        return (
            float(final_evaluation["gui_action_records"]) > 0
            and float(final_evaluation["gui_coordinate_records"]) > 0
        )
    except (TypeError, ValueError):
        return False


def _accepted(
    path: Path,
    *,
    require_gui_recovery: bool = False,
    require_tail_protected_lineage: bool = False,
    require_gui_correctness: bool = False,
) -> bool:
    manifest = path / "distillation-manifest.json"
    if not manifest.is_file():
        return False
    value = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping) or not bool(value.get("accepted")):
        return False
    if require_tail_protected_lineage and not _tail_protected_lineage(path):
        return False
    if require_gui_correctness and not _gui_correctness_measured(value):
        return False
    if not require_gui_recovery:
        return True

    if str(value.get("stage") or "") != "recovery":
        return False
    final_evaluation = value.get("final_evaluation")
    if not isinstance(final_evaluation, Mapping):
        return False
    try:
        return float(final_evaluation.get("gui_records") or 0.0) > 0.0
    except (TypeError, ValueError):
        return False


def _run(command: Sequence[str], *, dry_run: bool) -> None:
    print("+ " + " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(list(command), check=True)


def _count_records(path: Path) -> int:
    """Count non-blank JSONL records without loading the file into memory."""

    total = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                total += 1
    return total


def _require_recovery_corpora(plan: Mapping[str, object]) -> None:
    value = plan.get("recovery_corpora")
    if not isinstance(value, Mapping) or not value:
        raise ValueError("recovery_corpora must be configured before progressive distillation")
    missing: list[str] = []
    thin: list[str] = []
    for name, raw in value.items():
        if not isinstance(raw, Mapping) or not bool(raw.get("required", True)):
            continue
        path = Path(str(raw.get("path") or ""))
        if not path.is_file():
            missing.append(f"{name}: {path}")
            continue
        minimum = int(raw.get("min_records", 0))
        if minimum <= 0:
            continue
        records = _count_records(path)
        if records < minimum:
            thin.append(f"{name}: {records} records < required {minimum} ({path})")
    if missing:
        raise FileNotFoundError(
            "required broad recovery corpus is missing; refusing to start paid compression:\n- "
            + "\n- ".join(missing)
        )
    if thin:
        raise ValueError(
            "recovery corpus is too small for representation recovery; "
            "refusing to start paid compression:\n- " + "\n- ".join(thin)
        )


def _calibration_path(plan: Mapping[str, object], override: str | None) -> Path:
    if override:
        return Path(override)
    value = plan.get("calibration")
    if isinstance(value, Mapping) and value.get("manifest"):
        return Path(str(value["manifest"]))
    return Path("artifacts/calibration/gui-owl-compression.json")


def run(args: argparse.Namespace) -> dict[str, object]:
    plan = load_student_plan(args.config)
    _require_recovery_corpora(plan)
    student_entry = _student_entry(plan, args.student)
    progressive = _progressive_entry(plan, args.student)
    teacher_model = args.teacher_model or str(plan.get("initialization_teacher") or "")
    if not teacher_model:
        raise ValueError("initialization teacher is missing")

    source_arg = str(progressive.get("source") or "")
    if not source_arg:
        raise ValueError("progressive compression source is missing")
    source = Path(source_arg)
    if source_arg.startswith("artifacts/") and not source.is_dir():
        raise FileNotFoundError(
            f"required predecessor checkpoint is missing: {source}. Run the preceding size first."
        )
    if source_arg.startswith("artifacts/") and args.student in {"1b", "0.5b"}:
        if not _accepted(source, require_gui_correctness=True):
            raise RuntimeError(
                f"refusing to prune {args.student} from predecessor {source}: its accepted "
                "manifest predates the grounding correctness gate or lacks measured action/coordinate "
                "correctness. Re-evaluate/recover the predecessor before structural pruning."
            )

    calibration = _calibration_path(plan, args.calibration_manifest)
    if not calibration.is_file():
        raise FileNotFoundError(
            f"calibration manifest is missing: {calibration}; run calibrate_gui_owl_compression.py first"
        )
    if not args.dry_run:
        normalized = normalize_manifest(calibration)
        if normalized.get("gqa_group_consistent") is not True:
            raise RuntimeError("calibration did not preserve grouped-query attention head groups")

    schedule_raw = progressive.get("text_depth_schedule") or ()
    if not isinstance(schedule_raw, Sequence) or isinstance(schedule_raw, (str, bytes)):
        raise ValueError("text_depth_schedule must be an array")
    schedule = tuple(int(value) for value in schedule_raw)
    work_root = Path(args.work_root or f"artifacts/students/v2/progressive/{args.student}")
    work_root.mkdir(parents=True, exist_ok=True)
    current_source = source_arg
    stages: list[dict[str, object]] = []

    if schedule:
        upstream_rebuilt = False
        for index, depth in enumerate(schedule):
            init_dir = work_root / f"depth-{depth}-init"
            recovered_dir = work_root / f"depth-{depth}-recovered"
            is_final_depth = index == len(schedule) - 1
            resume_ok = (
                args.resume
                and not upstream_rebuilt
                and _accepted(
                    recovered_dir,
                    require_gui_recovery=is_final_depth,
                    require_tail_protected_lineage=True,
                    require_gui_correctness=True,
                )
            )
            if not resume_ok:
                build_command = [
                    sys.executable,
                    "scripts/build_gui_owl_student.py",
                    "--student",
                    args.student,
                    "--source-model",
                    current_source,
                    "--target-text-layers",
                    str(depth),
                    "--calibration-manifest",
                    str(calibration),
                    "--output",
                    str(init_dir),
                    "--overwrite",
                ]
                _run(build_command, dry_run=args.dry_run)

                recover_command = [
                    sys.executable,
                    RECOVERY_TRAINER,
                    "--student",
                    args.student,
                    "--stage",
                    "recovery",
                    "--student-model",
                    str(init_dir),
                    "--teacher-model",
                    teacher_model,
                    "--output",
                    str(recovered_dir),
                    "--device",
                    args.device,
                ]
                if args.max_records_per_corpus is not None:
                    recover_command += [
                        "--max-records-per-corpus",
                        str(args.max_records_per_corpus),
                    ]
                if args.max_updates is not None:
                    recover_command += ["--max-updates", str(args.max_updates)]
                elif not is_final_depth:
                    recover_command += [
                        "--max-updates",
                        str(args.intermediate_recovery_updates),
                    ]
                _run(recover_command, dry_run=args.dry_run)
                upstream_rebuilt = True
            current_source = str(recovered_dir)
            stages.append({"depth": depth, "checkpoint": str(recovered_dir)})
    else:
        init_dir = Path(
            str(student_entry.get("initialized") or work_root / "width-init")
        )
        recovered_dir = work_root / "width-recovered"
        if not (
            args.resume
            and _accepted(
                recovered_dir,
                require_gui_recovery=True,
                require_tail_protected_lineage=True,
                require_gui_correctness=True,
            )
        ):
            build_command = [
                sys.executable,
                "scripts/build_gui_owl_student.py",
                "--student",
                args.student,
                "--source-model",
                current_source,
                "--calibration-manifest",
                str(calibration),
                "--output",
                str(init_dir),
                "--overwrite",
            ]
            _run(build_command, dry_run=args.dry_run)
            recover_command = [
                sys.executable,
                RECOVERY_TRAINER,
                "--student",
                args.student,
                "--stage",
                "recovery",
                "--student-model",
                str(init_dir),
                "--teacher-model",
                teacher_model,
                "--output",
                str(recovered_dir),
                "--device",
                args.device,
            ]
            if args.max_records_per_corpus is not None:
                recover_command += [
                    "--max-records-per-corpus",
                    str(args.max_records_per_corpus),
                ]
            if args.max_updates is not None:
                recover_command += ["--max-updates", str(args.max_updates)]
            _run(recover_command, dry_run=args.dry_run)
        current_source = str(recovered_dir)
        stages.append({"width_pruned": True, "checkpoint": str(recovered_dir)})

    if not args.skip_on_policy:
        on_policy_dir = work_root / "on-policy"
        if not (args.resume and _accepted(on_policy_dir, require_gui_correctness=True)):
            command = [
                sys.executable,
                POLICY_TRAINER,
                "--student",
                args.student,
                "--stage",
                "on_policy",
                "--student-model",
                current_source,
                "--teacher-model",
                teacher_model,
                "--output",
                str(on_policy_dir),
                "--device",
                args.device,
            ]
            if args.max_records_per_corpus is not None:
                command += [
                    "--max-records-per-corpus",
                    str(args.max_records_per_corpus),
                ]
            if args.max_updates is not None:
                command += ["--max-updates", str(args.max_updates)]
            _run(command, dry_run=args.dry_run)
        current_source = str(on_policy_dir)
        stages.append({"stage": "on_policy", "checkpoint": current_source})

    final = Path(
        str(
            student_entry.get("distilled")
            or f"artifacts/students/v2/gui-owl-{args.student}-distilled"
        )
    )
    if not args.skip_specialize:
        if not (args.resume and _accepted(final, require_gui_correctness=True)):
            command = [
                sys.executable,
                POLICY_TRAINER,
                "--student",
                args.student,
                "--stage",
                "specialize",
                "--student-model",
                current_source,
                "--teacher-model",
                teacher_model,
                "--output",
                str(final),
                "--device",
                args.device,
            ]
            if args.max_records_per_corpus is not None:
                command += [
                    "--max-records-per-corpus",
                    str(args.max_records_per_corpus),
                ]
            if args.max_updates is not None:
                command += ["--max-updates", str(args.max_updates)]
            _run(command, dry_run=args.dry_run)
        current_source = str(final)
        stages.append({"stage": "specialize", "checkpoint": current_source})

    manifest = {
        "student": args.student,
        "calibration_manifest": str(calibration),
        "gqa_group_consistent": True if not args.dry_run else None,
        "teacher_model": teacher_model,
        "stages": stages,
        "final_checkpoint": current_source,
        "dry_run": bool(args.dry_run),
    }
    path = work_root / "progressive-run-manifest.json"
    if not args.dry_run:
        path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", required=True, choices=("0.5b", "1b", "1.5b"))
    parser.add_argument("--config", default="config/gui_owl_students.json")
    parser.add_argument("--teacher-model")
    parser.add_argument("--calibration-manifest")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--work-root")
    parser.add_argument("--intermediate-recovery-updates", type=int, default=750)
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("--max-records-per-corpus", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-on-policy", action="store_true")
    parser.add_argument("--skip-specialize", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(run(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
