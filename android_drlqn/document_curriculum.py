"""Snapshot-bounded documentation curriculum and coverage ledger.

The training process consumes normalized ORM exports. It never crawls, executes,
or treats source prose as instructions. "Complete" means that every section in
an immutable discovery snapshot has a terminal disposition; a source revision
change invalidates completion.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable, Mapping, Sequence


class DocumentClass(str, Enum):
    API_REFERENCE = "api_reference"
    USER_GUIDE = "user_guide"
    SETTINGS_UI_PATH = "settings_ui_path"
    ENTERPRISE_POLICY = "enterprise_policy"
    RELEASE_NOTES = "release_notes"
    DEVICE_SUPPORT = "device_support"
    PERMISSION_SECURITY = "permission_security"
    SOURCE_CODE = "source_code"
    RESOURCE_OVERLAY = "resource_overlay"
    LEGAL_LICENSE = "legal_license"
    INFORMATIONAL = "informational"


class CurriculumTaskKind(str, Enum):
    GROUND_UI = "ground_ui"
    PLAN_WORKFLOW = "plan_workflow"
    EXPLAIN_SETTING = "explain_setting"
    VERIFY_TRANSITION = "verify_transition"
    INFORMATIONAL_QA = "informational_qa"


class CoverageStatus(str, Enum):
    DISCOVERED = "discovered"
    NORMALIZED = "normalized"
    TASK_GENERATED = "task_generated"
    ROLLOUT_ATTEMPTED = "rollout_attempted"
    TEACHER_REVIEWED = "teacher_reviewed"
    STUDENT_UPDATED = "student_updated"
    VALIDATED = "validated"
    NON_ACTIONABLE = "non_actionable"
    BLOCKED = "blocked"


TERMINAL = {CoverageStatus.VALIDATED, CoverageStatus.NON_ACTIONABLE, CoverageStatus.BLOCKED}


@dataclass(frozen=True)
class SourceDescriptor:
    source_key: str
    os_family: str
    root_url: str
    confidence_tier: int
    license_identifier: str
    discovery_mode: str
    required_classes: tuple[DocumentClass, ...]
    repository_url: str | None = None
    allowed_hosts: tuple[str, ...] = ()
    enabled: bool = True

    def __post_init__(self) -> None:
        if not self.source_key or not self.os_family or not self.root_url:
            raise ValueError("source_key, os_family and root_url are required")
        if not 0 <= self.confidence_tier <= 6:
            raise ValueError("confidence_tier must be in [0, 6]")
        if not self.license_identifier or not self.allowed_hosts:
            raise ValueError("license and allowed_hosts are mandatory")


@dataclass(frozen=True)
class DocumentSection:
    source_key: str
    document_id: str
    snapshot_sha256: str
    section_id: str
    document_class: DocumentClass
    heading_path: tuple[str, ...]
    source_locator: Mapping[str, object]
    actionable: bool
    non_actionable_reason: str | None = None

    @property
    def coverage_id(self) -> str:
        value = "\0".join((self.source_key, self.document_id, self.snapshot_sha256, self.section_id))
        return hashlib.sha256(value.encode()).hexdigest()[:24]


@dataclass(frozen=True)
class CurriculumTask:
    task_id: str
    coverage_id: str
    source_key: str
    os_family: str
    snapshot_sha256: str
    section_id: str
    kind: CurriculumTaskKind
    prompt: str
    expected_output: Mapping[str, object]
    evidence: Mapping[str, object]
    setting_id: str | None = None
    risk_level: str = "low"
    target_selector: Mapping[str, object] = field(default_factory=dict)


@dataclass
class CoverageEntry:
    section: DocumentSection
    status: CoverageStatus = CoverageStatus.DISCOVERED
    task_ids: list[str] = field(default_factory=list)
    emulator_profile_ids: list[str] = field(default_factory=list)
    teacher_ids: list[str] = field(default_factory=list)
    checkpoint_ids: list[str] = field(default_factory=list)
    note: str | None = None

    def advance(self, status: CoverageStatus, note: str | None = None) -> None:
        if self.status in TERMINAL and status != self.status:
            raise ValueError("terminal coverage entries are immutable")
        order = list(CoverageStatus)
        if status not in TERMINAL and order.index(status) < order.index(self.status):
            return
        self.status = status
        self.note = note or self.note


@dataclass(frozen=True)
class DocumentationCoverageReport:
    total_sections: int
    terminal_sections: int
    validated_sections: int
    non_actionable_sections: int
    blocked_sections: int
    actionable_sections: int
    missing_required_classes: Mapping[str, tuple[str, ...]]
    stale_sources: tuple[str, ...]

    @property
    def completion_ratio(self) -> float:
        return 1.0 if not self.total_sections else self.terminal_sections / self.total_sections

    @property
    def complete(self) -> bool:
        return bool(self.total_sections) and self.terminal_sections == self.total_sections and not self.missing_required_classes and not self.stale_sources


class CoverageLedger:
    def __init__(self, sources: Sequence[SourceDescriptor]):
        self.sources = {source.source_key: source for source in sources if source.enabled}
        self.entries: dict[str, CoverageEntry] = {}
        self.pinned_revisions: dict[str, str] = {}
        self.current_revisions: dict[str, str] = {}

    def register_source_revision(self, source_key: str, pinned: str, current: str | None = None) -> None:
        if source_key not in self.sources:
            raise ValueError(f"unregistered source: {source_key}")
        self.pinned_revisions[source_key] = pinned
        self.current_revisions[source_key] = current or pinned

    def register_section(self, section: DocumentSection) -> CoverageEntry:
        if section.source_key not in self.sources:
            raise ValueError(f"unregistered source: {section.source_key}")
        entry = self.entries.setdefault(section.coverage_id, CoverageEntry(section))
        entry.advance(CoverageStatus.NORMALIZED)
        if not section.actionable:
            entry.advance(CoverageStatus.NON_ACTIONABLE, section.non_actionable_reason or "no executable claim")
        return entry

    def link_task(self, coverage_id: str, task_id: str) -> None:
        entry = self.entries[coverage_id]
        if task_id not in entry.task_ids:
            entry.task_ids.append(task_id)
        entry.advance(CoverageStatus.TASK_GENERATED)

    def mark_rollout(self, coverage_id: str, profile_id: str) -> None:
        entry = self.entries[coverage_id]
        if profile_id not in entry.emulator_profile_ids:
            entry.emulator_profile_ids.append(profile_id)
        entry.advance(CoverageStatus.ROLLOUT_ATTEMPTED)

    def mark_teacher_review(self, coverage_id: str, teacher_id: str) -> None:
        entry = self.entries[coverage_id]
        if teacher_id not in entry.teacher_ids:
            entry.teacher_ids.append(teacher_id)
        entry.advance(CoverageStatus.TEACHER_REVIEWED)

    def mark_student_update(self, coverage_id: str, checkpoint_id: str) -> None:
        entry = self.entries[coverage_id]
        if checkpoint_id not in entry.checkpoint_ids:
            entry.checkpoint_ids.append(checkpoint_id)
        entry.advance(CoverageStatus.STUDENT_UPDATED)

    def mark_validated(self, coverage_id: str, note: str | None = None) -> None:
        self.entries[coverage_id].advance(CoverageStatus.VALIDATED, note)

    def mark_blocked(self, coverage_id: str, reason: str) -> None:
        if not reason:
            raise ValueError("blocked entries need a reason")
        self.entries[coverage_id].advance(CoverageStatus.BLOCKED, reason)

    def report(self) -> DocumentationCoverageReport:
        observed: dict[str, set[DocumentClass]] = {key: set() for key in self.sources}
        values = tuple(self.entries.values())
        for entry in values:
            observed[entry.section.source_key].add(entry.section.document_class)
        missing = {}
        for key, source in self.sources.items():
            absent = tuple(sorted((item.value for item in set(source.required_classes) - observed[key])))
            if absent:
                missing[key] = absent
        stale = tuple(sorted(key for key, pinned in self.pinned_revisions.items() if self.current_revisions.get(key) != pinned))
        return DocumentationCoverageReport(
            total_sections=len(values),
            terminal_sections=sum(entry.status in TERMINAL for entry in values),
            validated_sections=sum(entry.status is CoverageStatus.VALIDATED for entry in values),
            non_actionable_sections=sum(entry.status is CoverageStatus.NON_ACTIONABLE for entry in values),
            blocked_sections=sum(entry.status is CoverageStatus.BLOCKED for entry in values),
            actionable_sections=sum(entry.section.actionable for entry in values),
            missing_required_classes=missing,
            stale_sources=stale,
        )

    def require_complete(self) -> DocumentationCoverageReport:
        report = self.report()
        if not report.complete:
            raise RuntimeError(f"documentation coverage incomplete: {asdict(report)}")
        return report

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "pinned_revisions": self.pinned_revisions,
            "current_revisions": self.current_revisions,
            "entries": [
                {
                    **asdict(entry),
                    "section": {**asdict(entry.section), "document_class": entry.section.document_class.value},
                    "status": entry.status.value,
                }
                for entry in self.entries.values()
            ],
            "report": asdict(self.report()),
        }

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


class DocumentationCurriculumBuilder:
    """Turns normalized snapshots into evidence-linked training tasks."""

    def __init__(self, sources: Sequence[SourceDescriptor], max_evidence_tier: int = 2):
        self.sources = {source.source_key: source for source in sources if source.enabled}
        self.max_evidence_tier = max_evidence_tier

    def build(self, documents: Iterable[Mapping[str, object]]) -> tuple[tuple[DocumentSection, ...], tuple[CurriculumTask, ...]]:
        sections: list[DocumentSection] = []
        tasks: list[CurriculumTask] = []
        for document in documents:
            meta = document.get("document") or {}
            source_key = str(meta.get("source_id") or "")
            source = self.sources.get(source_key)
            if source is None:
                raise ValueError(f"normalized document uses an unregistered source: {source_key}")
            snapshot = str(meta.get("snapshot_sha256") or "")
            if not snapshot:
                raise ValueError("normalized documents need snapshot_sha256")
            document_id = str(meta.get("document_id") or meta.get("canonical_url") or snapshot[:16])
            tier = int(meta.get("confidence_tier", source.confidence_tier))
            classes = tuple(_doc_class(value) for value in meta.get("document_type") or ("informational",))
            claims = tuple(value for value in document.get("claims") or () if isinstance(value, Mapping))
            raw_sections = tuple(document.get("sections") or ()) or ({"section_id": "root", "claim_indexes": list(range(len(claims)))},)
            for raw in raw_sections:
                section_id = str(raw.get("section_id") or "root")
                indexes = tuple(int(index) for index in raw.get("claim_indexes") or ())
                section_claims = tuple(claims[index] for index in indexes if 0 <= index < len(claims))
                actionable_claims = tuple(claim for claim in section_claims if claim.get("claim_kind") == "setting_variant" and tier <= self.max_evidence_tier)
                section = DocumentSection(
                    source_key=source_key,
                    document_id=document_id,
                    snapshot_sha256=snapshot,
                    section_id=section_id,
                    document_class=_doc_class(raw.get("document_class") or classes[0].value),
                    heading_path=tuple(str(value) for value in raw.get("heading_path") or ()),
                    source_locator=dict(raw.get("source_locator") or {}),
                    actionable=bool(actionable_claims),
                    non_actionable_reason=None if actionable_claims else ("source evidence tier exceeds threshold" if tier > self.max_evidence_tier else "no evidence-backed setting_variant claim"),
                )
                sections.append(section)
                if actionable_claims:
                    for claim in actionable_claims:
                        tasks.extend(self._tasks(source, section, claim))
                else:
                    tasks.append(self._informational(source, section, len(section_claims)))
        return tuple(sections), tuple(tasks)

    def _tasks(self, source: SourceDescriptor, section: DocumentSection, claim: Mapping[str, object]) -> list[CurriculumTask]:
        setting_id = str(claim.get("canonical_setting_hint") or claim.get("display_name") or "unknown_setting")
        display = str(claim.get("display_name") or setting_id)
        ui_path = tuple(str(value) for value in claim.get("ui_path") or ())
        evidence = {"source_key": source.source_key, "snapshot_sha256": section.snapshot_sha256, "section_id": section.section_id, "source_locator": dict(section.source_locator)}
        expected = {"setting_id": setting_id, "display_name": display, "ui_path": ui_path, "control_plane": claim.get("control_plane"), "required_role": claim.get("required_role")}
        kinds = [CurriculumTaskKind.PLAN_WORKFLOW, CurriculumTaskKind.EXPLAIN_SETTING, CurriculumTaskKind.VERIFY_TRANSITION]
        if ui_path:
            kinds.append(CurriculumTaskKind.GROUND_UI)
        output = []
        for kind in kinds:
            suffix = kind.value
            task_id = hashlib.sha256(f"{section.coverage_id}\0{setting_id}\0{suffix}".encode()).hexdigest()[:24]
            prompts = {
                CurriculumTaskKind.PLAN_WORKFLOW: f"Build an evidence-backed workflow for {display}.",
                CurriculumTaskKind.EXPLAIN_SETTING: f"Explain one safe next user step for {display}.",
                CurriculumTaskKind.VERIFY_TRANSITION: f"Verify the UI transition for {display} from an action receipt.",
                CurriculumTaskKind.GROUND_UI: f"Ground the visible UI element labelled {(ui_path[-1] if ui_path else 'Settings')!r}.",
            }
            output.append(CurriculumTask(task_id, section.coverage_id, source.source_key, source.os_family, section.snapshot_sha256, section.section_id, kind, prompts[kind], expected, evidence, setting_id, str(claim.get("risk_level") or "medium"), dict(claim.get("scope") or {})))
        return output

    @staticmethod
    def _informational(source: SourceDescriptor, section: DocumentSection, claim_count: int) -> CurriculumTask:
        task_id = hashlib.sha256(f"{section.coverage_id}\0informational".encode()).hexdigest()[:24]
        return CurriculumTask(task_id, section.coverage_id, source.source_key, source.os_family, section.snapshot_sha256, section.section_id, CurriculumTaskKind.INFORMATIONAL_QA, "Summarize normalized facts without inventing an executable action.", {"claim_count": claim_count, "must_not_execute": True}, {"source_key": source.source_key, "snapshot_sha256": section.snapshot_sha256, "section_id": section.section_id})


def load_source_registry(path: str | Path) -> tuple[SourceDescriptor, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    values = payload.get("sources") if isinstance(payload, Mapping) else payload
    if not isinstance(values, list):
        raise ValueError("source registry must contain a sources list")
    return tuple(
        SourceDescriptor(
            source_key=str(item["source_key"]),
            os_family=str(item["os_family"]),
            root_url=str(item["root_url"]),
            repository_url=item.get("repository_url"),
            confidence_tier=int(item.get("confidence_tier", 1)),
            license_identifier=str(item["license_identifier"]),
            discovery_mode=str(item["discovery_mode"]),
            required_classes=tuple(_doc_class(value) for value in item["required_classes"]),
            allowed_hosts=tuple(str(value) for value in item["allowed_hosts"]),
            enabled=bool(item.get("enabled", True)),
        )
        for item in values
    )


def _doc_class(value: object) -> DocumentClass:
    try:
        return DocumentClass(str(value))
    except ValueError:
        return DocumentClass.INFORMATIONAL
