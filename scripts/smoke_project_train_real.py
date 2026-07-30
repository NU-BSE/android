#!/usr/bin/env python3
"""Run one bounded LoRA update with the real project student model."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from android_drlqn.distillation import (  # noqa: E402
    DistillationSample,
    EnvironmentOutcome,
    OnPolicyConfig,
    ProposalKind,
    RolloutObservation,
    StudentProposal,
    TeacherJudgement,
    TeacherTarget,
)
from android_drlqn.document_curriculum import (  # noqa: E402
    CurriculumTask,
    CurriculumTaskKind,
)
from android_drlqn.emulator_lab import selected_emulator_matrix  # noqa: E402

import project_train  # noqa: E402


def _sample(policy_version: int) -> DistillationSample:
    task = CurriculumTask(
        "real-model-smoke",
        "real-model-smoke",
        "android_settings_api",
        "android-aosp",
        "smoke-snapshot",
        "root",
        CurriculumTaskKind.GROUND_UI,
        "Open Bluetooth settings",
        {"display_name": "Bluetooth"},
        {"snapshot": "smoke-snapshot"},
        setting_id="bluetooth",
    )
    profile = selected_emulator_matrix().profiles[0]
    observation = RolloutObservation(
        "fixture://smoke-screen",
        ("Network & internet", "Bluetooth", "Privacy"),
        {"node_count": 3},
        "com.android.settings",
        ".Settings",
        0,
    )
    proposal = StudentProposal(
        ProposalKind.TAP,
        "Bluetooth",
        None,
        "Tap Bluetooth",
        0.7,
        (0.7, 0.1, 0.05, 0.05, 0.05, 0.05),
        -0.35,
        policy_version,
    )
    judgement = TeacherJudgement(
        "real-model-smoke-teacher",
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        True,
        TeacherTarget(ProposalKind.TAP, None, None, "ok"),
    )
    return DistillationSample(
        task,
        profile,
        observation,
        proposal,
        EnvironmentOutcome(1.0, True, True, None, "verified"),
        judgement,
        1.0,
        0.5,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id",
        default=project_train.DEFAULT_MODEL_ID,
        help="Hugging Face model id or local model directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/project-train-real-smoke"),
    )
    parser.add_argument("--pack-len", type=int, default=64)
    args = parser.parse_args()

    started = time.monotonic()
    student, updater = project_train.create_student(
        {
            "model_id": args.model_id,
            "device": "cpu",
            "dtype": "float32",
            "seed": 0,
            "max_new_tokens": 1,
            "speedrun": {
                "pack_len": args.pack_len,
                "ce_chunk": args.pack_len,
                "total_steps": 1,
                "warmup_steps": 1,
            },
        },
        args.output,
    )
    sample = _sample(student.policy_version)
    encoded = student.encode_sample(sample)
    if len(encoded.completion_ids) >= args.pack_len:
        raise RuntimeError(
            f"completion has {len(encoded.completion_ids)} tokens, "
            f"but pack_len is only {args.pack_len}"
        )

    checkpoint_id = updater.update(
        [sample],
        OnPolicyConfig(
            supervised_weight=1.0,
            policy_weight=0.1,
            grounding_weight=0.0,
            language_weight=1.0,
            kl_weight=0.0,
        ),
    )
    checkpoint = args.output / "checkpoints" / "step-00000001"
    adapter = checkpoint / "adapter_model.safetensors"
    if student.policy_version != 1:
        raise RuntimeError(f"expected policy version 1, got {student.policy_version}")
    if len(checkpoint_id) != 24 or not adapter.is_file():
        raise RuntimeError("LoRA checkpoint was not written")

    print(
        json.dumps(
            {
                "model_id": args.model_id,
                "policy_version": student.policy_version,
                "completion_tokens": len(encoded.completion_ids),
                "checkpoint_id": checkpoint_id,
                "adapter_bytes": adapter.stat().st_size,
                "elapsed_seconds": round(time.monotonic() - started, 1),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
