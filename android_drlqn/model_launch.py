"""Unified local-model launch configuration for all supported Android OS families.

The launch bundle ties together the loopback Ollama runtime, normalized OS
settings, emulator profiles, and per-OS on-policy distillation settings. It does
not download models, crawl documentation, or silently replace missing custom OS
images with AOSP profiles.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .distillation import (
    DistillationMetrics,
    OnPolicyConfig,
    TeacherEndpointConfig,
    load_distillation_config,
)
from .document_curriculum import CurriculumTask, SourceDescriptor, load_source_registry
from .emulator_lab import EmulatorMatrix, load_emulator_matrix
from .ollama_runtime import (
    OllamaEnsembleConfig,
    OllamaModelSpec,
    OllamaRuntimeBundle,
    OllamaTransport,
)


class DistillationMode(str, Enum):
    REQUIRED = "required"
    OPTIONAL = "optional"
    DISABLED = "disabled"


_MODEL_OVERRIDE_FIELDS = frozenset(
    {
        "keep_alive",
        "num_ctx",
        "temperature",
        "top_p",
        "coordinate_scale",
    }
)
_POLICY_FIELDS = frozenset(asdict(OnPolicyConfig()))


@dataclass(frozen=True)
class OSLaunchProfile:
    os_family: str
    profile_ids: tuple[str, ...]
    distillation_mode: DistillationMode = DistillationMode.REQUIRED
    max_settings_context: int = 24
    qwen3_vl: Mapping[str, object] = field(default_factory=dict)
    ui_tars: Mapping[str, object] = field(default_factory=dict)
    on_policy: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.os_family:
            raise ValueError("os_family is required")
        if len(self.profile_ids) != len(set(self.profile_ids)):
            raise ValueError(f"duplicate emulator profile id for {self.os_family}")
        if self.max_settings_context < 1:
            raise ValueError("max_settings_context must be positive")
        unknown_models = (set(self.qwen3_vl) | set(self.ui_tars)) - _MODEL_OVERRIDE_FIELDS
        if unknown_models:
            raise ValueError(f"unknown Ollama model overrides: {sorted(unknown_models)}")
        unknown_policy = set(self.on_policy) - _POLICY_FIELDS
        if unknown_policy:
            raise ValueError(f"unknown on-policy overrides: {sorted(unknown_policy)}")
        if self.distillation_mode is DistillationMode.REQUIRED and not self.profile_ids:
            raise ValueError(f"required distillation needs profiles for {self.os_family}")


@dataclass(frozen=True)
class ModelLaunchConfig:
    config_path: Path
    ollama_config: Path
    distillation_config: Path
    source_registry: Path
    emulator_matrix: Path
    require_all_os_settings: bool
    max_evidence_tier: int
    os_profiles: tuple[OSLaunchProfile, ...]

    def __post_init__(self) -> None:
        families = [profile.os_family for profile in self.os_profiles]
        if not families:
            raise ValueError("at least one OS launch profile is required")
        if len(families) != len(set(families)):
            raise ValueError("OS launch profiles must have unique os_family values")
        if not 0 <= self.max_evidence_tier <= 6:
            raise ValueError("max_evidence_tier must be in [0, 6]")

    @classmethod
    def from_json(cls, path: str | Path) -> "ModelLaunchConfig":
        config_path = Path(path).expanduser().resolve()
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("model launch config must be a JSON object")
        if int(payload.get("schema_version", 0)) != 1:
            raise ValueError("unsupported model launch config schema_version")
        paths = _mapping(payload.get("paths"), "paths")
        settings = _mapping(payload.get("settings") or {}, "settings")
        raw_profiles = _mapping(payload.get("os_families"), "os_families")
        base = config_path.parent
        profiles = []
        for os_family, raw_value in raw_profiles.items():
            raw = _mapping(raw_value, f"os_families.{os_family}")
            profiles.append(
                OSLaunchProfile(
                    os_family=str(os_family),
                    profile_ids=tuple(str(value) for value in raw.get("profile_ids") or ()),
                    distillation_mode=DistillationMode(
                        str(raw.get("distillation_mode") or DistillationMode.REQUIRED.value)
                    ),
                    max_settings_context=int(raw.get("max_settings_context", 24)),
                    qwen3_vl=dict(_optional_mapping(raw.get("qwen3_vl"))),
                    ui_tars=dict(_optional_mapping(raw.get("ui_tars"))),
                    on_policy=dict(_optional_mapping(raw.get("on_policy"))),
                )
            )
        return cls(
            config_path=config_path,
            ollama_config=_resolve_path(base, paths.get("ollama_config")),
            distillation_config=_resolve_path(base, paths.get("distillation_config")),
            source_registry=_resolve_path(base, paths.get("source_registry")),
            emulator_matrix=_resolve_path(base, paths.get("emulator_matrix")),
            require_all_os_settings=bool(settings.get("require_all_os_settings", True)),
            max_evidence_tier=int(settings.get("max_evidence_tier", 2)),
            os_profiles=tuple(profiles),
        )

    def for_os(self, os_family: str) -> OSLaunchProfile:
        for profile in self.os_profiles:
            if profile.os_family == os_family:
                return profile
        raise KeyError(f"OS family is not configured for model launch: {os_family}")


@dataclass(frozen=True)
class OSSettingRecord:
    os_family: str
    source_key: str
    setting_id: str
    display_name: str
    ui_path: tuple[str, ...]
    risk_level: str
    control_plane: str | None
    required_role: str | None
    scope: Mapping[str, object]
    snapshot_sha256: str

    def to_prompt_dict(self) -> dict[str, object]:
        return {
            "setting_id": self.setting_id,
            "display_name": self.display_name,
            "ui_path": list(self.ui_path),
            "risk_level": self.risk_level,
            "control_plane": self.control_plane,
            "required_role": self.required_role,
            "scope": dict(self.scope),
            "evidence": {
                "source_key": self.source_key,
                "snapshot_sha256": self.snapshot_sha256,
            },
        }


class OSSettingsRegistry:
    def __init__(self, records: Iterable[OSSettingRecord]):
        grouped: dict[str, list[OSSettingRecord]] = {}
        seen: set[tuple[str, str, str, tuple[str, ...]]] = set()
        for record in records:
            scope_key = json.dumps(dict(record.scope), sort_keys=True, separators=(",", ":"))
            key = (record.os_family, record.setting_id, scope_key, record.ui_path)
            if key in seen:
                continue
            seen.add(key)
            grouped.setdefault(record.os_family, []).append(record)
        self._records = {
            family: tuple(sorted(values, key=lambda item: (item.setting_id, item.display_name, item.source_key)))
            for family, values in grouped.items()
        }

    @classmethod
    def from_documents(
        cls,
        documents: Iterable[Mapping[str, object]],
        sources: Sequence[SourceDescriptor],
        *,
        max_evidence_tier: int = 2,
    ) -> "OSSettingsRegistry":
        source_map = {source.source_key: source for source in sources if source.enabled}
        records: list[OSSettingRecord] = []
        for document in documents:
            meta = _optional_mapping(document.get("document"))
            source_key = str(meta.get("source_id") or "")
            source = source_map.get(source_key)
            if source is None:
                raise ValueError(f"normalized document uses an unregistered source: {source_key}")
            snapshot = str(meta.get("snapshot_sha256") or "")
            if not snapshot:
                raise ValueError("normalized documents need snapshot_sha256")
            tier = int(meta.get("confidence_tier", source.confidence_tier))
            if tier > max_evidence_tier:
                continue
            for claim in document.get("claims") or ():
                if not isinstance(claim, Mapping) or claim.get("claim_kind") != "setting_variant":
                    continue
                setting_id = str(
                    claim.get("canonical_setting_hint") or claim.get("display_name") or ""
                ).strip()
                if not setting_id:
                    continue
                display_name = str(claim.get("display_name") or setting_id)
                ui_path = tuple(str(value) for value in claim.get("ui_path") or ())
                scope = _optional_mapping(claim.get("scope"))
                records.append(
                    OSSettingRecord(
                        os_family=source.os_family,
                        source_key=source.source_key,
                        setting_id=setting_id,
                        display_name=display_name,
                        ui_path=ui_path,
                        risk_level=str(claim.get("risk_level") or "medium"),
                        control_plane=_optional_str(claim.get("control_plane")),
                        required_role=_optional_str(claim.get("required_role")),
                        scope=dict(scope),
                        snapshot_sha256=snapshot,
                    )
                )
        return cls(records)

    @property
    def families(self) -> tuple[str, ...]:
        return tuple(sorted(self._records))

    def count(self, os_family: str) -> int:
        return len(self._records.get(os_family, ()))

    def records_for(self, os_family: str) -> tuple[OSSettingRecord, ...]:
        return self._records.get(os_family, ())

    def require_families(self, os_families: Iterable[str]) -> None:
        missing = sorted(family for family in set(os_families) if not self._records.get(family))
        if missing:
            raise RuntimeError(f"normalized settings are missing for OS families: {missing}")

    def context(
        self,
        os_family: str,
        *,
        setting_id: str | None,
        limit: int,
    ) -> Mapping[str, object]:
        records = list(self.records_for(os_family))
        if limit < 1:
            raise ValueError("settings context limit must be positive")
        needle = (setting_id or "").casefold()
        exact = [
            record
            for record in records
            if needle
            and needle
            in {record.setting_id.casefold(), record.display_name.casefold()}
        ]
        exact_keys = {(item.setting_id, item.source_key, item.ui_path) for item in exact}
        remaining = [
            record
            for record in records
            if (record.setting_id, record.source_key, record.ui_path) not in exact_keys
        ]
        selected = (exact + remaining)[:limit]
        return {
            "os_family": os_family,
            "setting_count": len(records),
            "selected_count": len(selected),
            "settings": [record.to_prompt_dict() for record in selected],
        }


@dataclass(frozen=True)
class ModelLaunchBundle:
    config: ModelLaunchConfig
    sources: tuple[SourceDescriptor, ...]
    matrix: EmulatorMatrix
    teacher_endpoint: TeacherEndpointConfig
    base_policy: OnPolicyConfig
    student_config: Mapping[str, object]
    ollama_config: OllamaEnsembleConfig
    settings: OSSettingsRegistry
    transport: OllamaTransport | None = field(default=None, repr=False, compare=False)

    @classmethod
    def from_json(
        cls,
        config_path: str | Path,
        documents_path: str | Path,
        *,
        ollama_config: str | Path | None = None,
        distillation_config: str | Path | None = None,
        source_registry: str | Path | None = None,
        emulator_matrix: str | Path | None = None,
        transport: OllamaTransport | None = None,
    ) -> "ModelLaunchBundle":
        config = ModelLaunchConfig.from_json(config_path)
        config = replace(
            config,
            ollama_config=_override_path(config.ollama_config, ollama_config),
            distillation_config=_override_path(config.distillation_config, distillation_config),
            source_registry=_override_path(config.source_registry, source_registry),
            emulator_matrix=_override_path(config.emulator_matrix, emulator_matrix),
        )
        sources = load_source_registry(config.source_registry)
        matrix = load_emulator_matrix(config.emulator_matrix)
        matrix.require_complete()
        teacher, policy, student = load_distillation_config(config.distillation_config)
        ollama_runtime = OllamaEnsembleConfig.from_json(config.ollama_config)
        documents = json.loads(Path(documents_path).read_text(encoding="utf-8"))
        if not isinstance(documents, list):
            raise ValueError("normalized documents must contain a JSON array")
        settings = OSSettingsRegistry.from_documents(
            documents,
            sources,
            max_evidence_tier=config.max_evidence_tier,
        )
        bundle = cls(
            config=config,
            sources=tuple(sources),
            matrix=matrix,
            teacher_endpoint=teacher,
            base_policy=policy,
            student_config=dict(student),
            ollama_config=ollama_runtime,
            settings=settings,
            transport=transport,
        )
        bundle.validate()
        return bundle

    @property
    def os_families(self) -> tuple[str, ...]:
        return tuple(profile.os_family for profile in self.config.os_profiles)

    def validate(self) -> None:
        configured = set(self.os_families)
        matrix_families = {profile.os_family for profile in self.matrix.profiles}
        source_families = {source.os_family for source in self.sources if source.enabled}
        missing_config = sorted((matrix_families | source_families) - configured)
        if missing_config:
            raise RuntimeError(f"model launch config omits OS families: {missing_config}")
        if self.config.require_all_os_settings:
            self.settings.require_families(configured)
        profiles_by_id = {profile.profile_id: profile for profile in self.matrix.profiles}
        for launch_profile in self.config.os_profiles:
            for profile_id in launch_profile.profile_ids:
                profile = profiles_by_id.get(profile_id)
                if profile is None:
                    raise RuntimeError(
                        f"model launch references unknown emulator profile: {profile_id}"
                    )
                if profile.os_family != launch_profile.os_family:
                    raise RuntimeError(
                        f"emulator profile {profile_id} belongs to {profile.os_family}, "
                        f"not {launch_profile.os_family}"
                    )

    def runtime_for(self, os_family: str) -> OllamaRuntimeBundle:
        profile = self.config.for_os(os_family)
        runtime_config = replace(
            self.ollama_config,
            qwen3_vl=_override_model_spec(self.ollama_config.qwen3_vl, profile.qwen3_vl),
            ui_tars=_override_model_spec(self.ollama_config.ui_tars, profile.ui_tars),
        )
        return OllamaRuntimeBundle(runtime_config, transport=self.transport)

    def policy_for(self, os_family: str) -> OnPolicyConfig:
        profile = self.config.for_os(os_family)
        values = asdict(self.base_policy)
        values.update(profile.on_policy)
        return OnPolicyConfig(**values)

    def matrix_for(self, os_family: str) -> EmulatorMatrix:
        launch_profile = self.config.for_os(os_family)
        profiles_by_id = {profile.profile_id: profile for profile in self.matrix.profiles}
        return EmulatorMatrix(profiles_by_id[profile_id] for profile_id in launch_profile.profile_ids)

    def enrich_tasks(
        self,
        tasks: Sequence[CurriculumTask],
        os_family: str,
    ) -> tuple[CurriculumTask, ...]:
        launch_profile = self.config.for_os(os_family)
        output = []
        for task in tasks:
            if task.os_family != os_family:
                raise ValueError(
                    f"task {task.task_id} belongs to {task.os_family}, not {os_family}"
                )
            evidence = dict(task.evidence)
            evidence["os_settings_context"] = self.settings.context(
                os_family,
                setting_id=task.setting_id,
                limit=launch_profile.max_settings_context,
            )
            output.append(replace(task, evidence=evidence))
        return tuple(output)

    def plan(self) -> Mapping[str, object]:
        os_values: dict[str, object] = {}
        for profile in self.config.os_profiles:
            runtime = self.runtime_for(profile.os_family)
            os_values[profile.os_family] = {
                "distillation_mode": profile.distillation_mode.value,
                "settings_count": self.settings.count(profile.os_family),
                "max_settings_context": profile.max_settings_context,
                "profile_ids": list(profile.profile_ids),
                "qwen3_vl": asdict(runtime.config.qwen3_vl),
                "ui_tars": asdict(runtime.config.ui_tars),
                "on_policy": asdict(self.policy_for(profile.os_family)),
            }
        return {
            "schema_version": 1,
            "launch_config": str(self.config.config_path),
            "teacher_profile": "ollama_local_ensemble",
            "teacher_id": self.runtime_for(self.os_families[0]).teacher.teacher_id,
            "student": dict(self.student_config),
            "require_all_os_settings": self.config.require_all_os_settings,
            "os_families": os_values,
        }


def aggregate_distillation_metrics(
    metrics_by_os: Mapping[str, DistillationMetrics],
) -> DistillationMetrics:
    values = tuple(metrics_by_os.values())
    rollouts = sum(value.rollouts for value in values)
    updates = sum(value.updates for value in values)
    verified = sum(value.verified_successes for value in values)
    reward_total = sum(value.mean_reward * value.rollouts for value in values)
    teacher_total = sum(value.mean_teacher_score * value.rollouts for value in values)
    checkpoint = next(
        (value.checkpoint_id for value in reversed(values) if value.checkpoint_id),
        None,
    )
    return DistillationMetrics(
        rollouts=rollouts,
        updates=updates,
        verified_successes=verified,
        mean_reward=reward_total / rollouts if rollouts else 0.0,
        mean_teacher_score=teacher_total / rollouts if rollouts else 0.0,
        checkpoint_id=checkpoint,
    )


def _override_model_spec(
    base: OllamaModelSpec,
    overrides: Mapping[str, object],
) -> OllamaModelSpec:
    if not overrides:
        return base
    values = asdict(base)
    values.update(overrides)
    values["required_capabilities"] = tuple(values["required_capabilities"])
    return OllamaModelSpec(**values)


def _override_path(current: Path, override: str | Path | None) -> Path:
    if override is None:
        return current
    return Path(override).expanduser().resolve()


def _resolve_path(base: Path, value: object) -> Path:
    text = str(value or "").strip()
    if not text:
        raise ValueError("all model launch paths are required")
    path = Path(text).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _optional_mapping(value: object) -> Mapping[str, object]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("expected a JSON object")
    return value


def _optional_str(value: object) -> str | None:
    return None if value is None else str(value)
