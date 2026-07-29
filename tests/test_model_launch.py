from __future__ import annotations

import json
from pathlib import Path

import pytest

from android_drlqn.distillation import DistillationMetrics
from android_drlqn.document_curriculum import CurriculumTask, CurriculumTaskKind
from android_drlqn.model_launch import (
    ModelLaunchBundle,
    ModelLaunchConfig,
    aggregate_distillation_metrics,
)


def _write_documents(path: Path, families: tuple[tuple[str, str], ...]) -> Path:
    documents = []
    for index, (source_id, setting_id) in enumerate(families):
        documents.append(
            {
                "document": {
                    "source_id": source_id,
                    "snapshot_sha256": f"{'a' * 63}{index}",
                    "confidence_tier": 1,
                    "document_id": f"doc-{index}",
                    "document_type": ["settings_ui_path"],
                },
                "claims": [
                    {
                        "claim_kind": "setting_variant",
                        "canonical_setting_hint": setting_id,
                        "display_name": setting_id.replace("_", " ").title(),
                        "ui_path": ["Settings", setting_id],
                        "risk_level": "low",
                        "control_plane": "user_ui",
                        "required_role": "device_user",
                        "scope": {"release": "test"},
                    }
                ],
                "sections": [
                    {
                        "section_id": "root",
                        "document_class": "settings_ui_path",
                        "claim_indexes": [0],
                    }
                ],
            }
        )
    path.write_text(json.dumps(documents), encoding="utf-8")
    return path


def _all_os_documents(tmp_path: Path) -> Path:
    return _write_documents(
        tmp_path / "documents.json",
        (
            ("android_settings_api", "wifi_scanning"),
            ("grapheneos_official_docs", "exploit_protection"),
            ("calyxos_official_docs", "microg_location"),
            ("eos_official_docs", "advanced_privacy"),
        ),
    )


def test_launch_config_resolves_central_paths() -> None:
    config = ModelLaunchConfig.from_json("config/model_launch.json")

    assert config.ollama_config.name == "ollama_models.json"
    assert config.distillation_config.name == "distillation.json"
    assert config.require_all_os_settings is True
    assert {profile.os_family for profile in config.os_profiles} == {
        "android-aosp",
        "grapheneos",
        "calyxos",
        "eos",
    }


def test_bundle_loads_settings_and_runtime_overrides_for_every_os(tmp_path: Path) -> None:
    bundle = ModelLaunchBundle.from_json(
        "config/model_launch.json",
        _all_os_documents(tmp_path),
    )

    assert bundle.settings.count("android-aosp") == 1
    assert bundle.settings.count("grapheneos") == 1
    assert bundle.settings.count("calyxos") == 1
    assert bundle.settings.count("eos") == 1
    assert bundle.runtime_for("eos").config.qwen3_vl.num_ctx == 24576
    assert bundle.runtime_for("eos").config.ui_tars.num_ctx == 12288
    assert bundle.policy_for("grapheneos").min_teacher_confidence == 0.75
    assert [profile.profile_id for profile in bundle.matrix_for("grapheneos").profiles] == [
        "grapheneos17-sdk-phone64"
    ]


def test_bundle_rejects_missing_os_settings(tmp_path: Path) -> None:
    documents = _write_documents(
        tmp_path / "documents.json",
        (("android_settings_api", "wifi_scanning"),),
    )

    with pytest.raises(RuntimeError, match="missing for OS families"):
        ModelLaunchBundle.from_json("config/model_launch.json", documents)


def test_enrich_tasks_attaches_bounded_os_settings_context(tmp_path: Path) -> None:
    bundle = ModelLaunchBundle.from_json(
        "config/model_launch.json",
        _all_os_documents(tmp_path),
    )
    task = CurriculumTask(
        task_id="task-1",
        coverage_id="coverage-1",
        source_key="grapheneos_official_docs",
        os_family="grapheneos",
        snapshot_sha256="b" * 64,
        section_id="root",
        kind=CurriculumTaskKind.PLAN_WORKFLOW,
        prompt="Build a workflow.",
        expected_output={"display_name": "Exploit Protection"},
        evidence={"source_key": "grapheneos_official_docs"},
        setting_id="exploit_protection",
    )

    enriched = bundle.enrich_tasks((task,), "grapheneos")[0]
    context = enriched.evidence["os_settings_context"]

    assert context["os_family"] == "grapheneos"
    assert context["selected_count"] == 1
    assert context["settings"][0]["setting_id"] == "exploit_protection"
    assert task.evidence == {"source_key": "grapheneos_official_docs"}


def test_aggregate_distillation_metrics_is_rollout_weighted() -> None:
    metrics = aggregate_distillation_metrics(
        {
            "android-aosp": DistillationMetrics(4, 1, 3, 1.0, 0.5, "a"),
            "grapheneos": DistillationMetrics(2, 2, 1, 2.0, 1.0, "b"),
        }
    )

    assert metrics.rollouts == 6
    assert metrics.updates == 3
    assert metrics.verified_successes == 4
    assert metrics.mean_reward == pytest.approx(4 / 3)
    assert metrics.mean_teacher_score == pytest.approx(2 / 3)
    assert metrics.checkpoint_id == "b"
