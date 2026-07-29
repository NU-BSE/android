#!/usr/bin/env python3
"""Run on-policy distillation from a local Ollama VLM teacher across AVDs.

The run is sequential by construction: one emulator profile is booted, used
and torn down before the next one starts, and a single student checkpoint
carries across all of them.

    python scripts/run_distillation.py \
        --profiles config/emulator_profiles.json \
        --tasks config/distillation_tasks.json \
        --teacher qwen2.5vl:7b \
        --allow-emulator-input \
        --output runs/2026-07-26

Input injection is refused unless --allow-emulator-input is passed AND adb
reports the target as an emulator. This script must never be pointed at a
personal device.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow `python scripts/run_distillation.py` from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from android_drlqn.core import AndroidBatchConfig  # noqa: E402
from android_drlqn.distillation import (  # noqa: E402
    DistillationConfig,
    DistillationTask,
    build_student,
    load_checkpoint,
    run_sequential_distillation,
    save_checkpoint,
)
from android_drlqn.emulator import EmulatorError, EmulatorFleet, load_profiles  # noqa: E402
from android_drlqn.ollama import (  # noqa: E402
    DEFAULT_OLLAMA_HOST,
    OllamaConfig,
    OllamaError,
    ollama_grounding_vlm,
)


def load_tasks(path: Path) -> tuple[DistillationTask, ...]:
    document = json.loads(path.read_text(encoding="utf-8"))
    entries = document.get("tasks") if isinstance(document, dict) else document
    if not entries:
        raise SystemExit(f"{path} does not define any task")
    return tuple(
        DistillationTask(
            goal=str(entry["goal"]),
            targets=tuple(str(item) for item in entry["targets"]),
            max_attempts_per_target=int(entry.get("max_attempts_per_target", 3)),
        )
        for entry in entries
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profiles", type=Path, required=True, help="ordered AVD profile JSON")
    parser.add_argument("--tasks", type=Path, required=True, help="distillation task JSON")
    parser.add_argument("--teacher", required=True, help="Ollama vision model tag")
    parser.add_argument("--ollama-host", default=DEFAULT_OLLAMA_HOST)
    parser.add_argument("--output", type=Path, required=True, help="run directory")
    parser.add_argument("--resume", type=Path, help="student checkpoint to continue from")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-steps-per-profile", type=int, default=128)
    parser.add_argument("--exploration", type=float, default=0.10)
    parser.add_argument("--min-teacher-confidence", type=float, default=0.60)
    parser.add_argument("--min-teacher-iou", type=float, default=0.35)
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--teacher-vlm-only",
        action="store_true",
        help="never accept the accessibility shortcut as teacher supervision",
    )
    parser.add_argument(
        "--allow-emulator-input",
        action="store_true",
        help="required to inject taps; adb must also report an emulator",
    )
    parser.add_argument("--adb-path", default="adb")
    parser.add_argument("--emulator-path", default="emulator")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate profiles, tasks and the Ollama model, then exit",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    profiles = load_profiles(args.profiles)
    tasks = load_tasks(args.tasks)

    teacher_config = OllamaConfig(model=args.teacher, host=args.ollama_host)
    teacher = ollama_grounding_vlm(teacher_config)

    config = DistillationConfig(
        learning_rate=args.learning_rate,
        temperature=args.temperature,
        batch_size=args.batch_size,
        max_steps_per_profile=args.max_steps_per_profile,
        exploration=args.exploration,
        min_teacher_confidence=args.min_teacher_confidence,
        min_teacher_iou=args.min_teacher_iou,
        teacher_vlm_only=args.teacher_vlm_only,
        seed=args.seed,
        batch_config=AndroidBatchConfig(history=args.history),
    )

    if args.dry_run:
        print(
            json.dumps(
                {
                    "profiles": [profile.name for profile in profiles],
                    "tasks": [task.goal for task in tasks],
                    "teacher": args.teacher,
                    "ollama_host": args.ollama_host,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    student = build_student(config, seed=args.seed)
    if args.resume:
        load_checkpoint(student, args.resume)

    fleet = EmulatorFleet(
        adb_path=args.adb_path,
        emulator_path=args.emulator_path,
        allow_input=args.allow_emulator_input,
    )
    try:
        report = run_sequential_distillation(
            student,
            teacher,
            fleet.sequential(profiles),
            tasks,
            config,
            checkpoint_dir=args.output / "checkpoints",
        )
    finally:
        fleet.stop_all()

    save_checkpoint(student, args.output / "checkpoints" / "student-final.npz")
    report.write(args.output / "report.json")
    for summary in report.profiles:
        print(
            f"{summary.profile:<28} steps={summary.steps:<4} "
            f"accepted={summary.acceptance_rate:.0%} "
            f"agreement={summary.agreement_rate:.0%} "
            f"node_loss={summary.mean_node_loss:.4f}"
        )
    print(f"updates={report.total_updates} report={args.output / 'report.json'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OllamaError, EmulatorError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from None
