"""Evidence-backed VLM + chat coordination for Android settings.

This module consumes normalized ORM exports rather than scraping documentation
or inventing settings at runtime. It resolves an exact OS/release/device scope,
creates a reviewable settings workflow, asks the VLM to ground only the next
semantic target, and lets the chat model explain the action. Execution remains
behind an explicit confirmation/receipt boundary.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Mapping, Protocol, Sequence

from .advisor import LocalChatModel, PlannerAdvisor
from .core import AndroidState, VerificationExpectation, verify_transition
from .vlm import GroundingDecision, HybridGrounder, LocalVisionLanguageModel
from .window_graph import PlannerSubgoal, SubgoalType


class ControlPlane(str, Enum):
    ANDROID_PUBLIC_API = "android_public_api"
    SETTINGS_PANEL = "settings_panel"
    DEVICE_POLICY_MANAGER = "device_policy_manager"
    ANDROID_MANAGEMENT_API = "android_management_api"
    OS_SPECIFIC_API = "os_specific_api"
    DOCUMENTED_SHELL = "shell_command"
    SETTINGS_PROVIDER = "settings_provider"
    UI_ONLY = "ui_only"
    MANUAL_ONLY = "manual_only"
    UNKNOWN = "unknown"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class WorkflowStepType(str, Enum):
    CHECK_TARGET = "CHECK_TARGET"
    CHECK_PERMISSION = "CHECK_PERMISSION"
    READ_SETTING = "READ_SETTING"
    OPEN_SETTINGS_PANEL = "OPEN_SETTINGS_PANEL"
    REQUEST_USER_ACTION = "REQUEST_USER_ACTION"
    VERIFY_VALUE = "VERIFY_VALUE"
    REPORT = "REPORT"


class SessionStatus(str, Enum):
    READY = "ready"
    WAITING_FOR_CONFIRMATION = "waiting_for_confirmation"
    WAITING_FOR_RECEIPT = "waiting_for_receipt"
    BLOCKED = "blocked"
    DONE = "done"


@dataclass(frozen=True)
class TargetScope:
    os_family: str
    os_release: str | None
    android_version: int
    api_level: int
    device_codename: str | None = None
    management_mode: str = "unmanaged"


@dataclass(frozen=True)
class VariantScope:
    os_family: str
    min_android_version: int | None = None
    max_android_version: int | None = None
    min_api_level: int | None = None
    max_api_level: int | None = None
    release_min: str | None = None
    release_max: str | None = None
    device_codenames: tuple[str, ...] = ()
    management_modes: tuple[str, ...] = ()

    def matches(self, target: TargetScope) -> bool:
        if self.os_family.casefold() != target.os_family.casefold():
            return False
        if self.min_android_version is not None and target.android_version < self.min_android_version:
            return False
        if self.max_android_version is not None and target.android_version > self.max_android_version:
            return False
        if self.min_api_level is not None and target.api_level < self.min_api_level:
            return False
        if self.max_api_level is not None and target.api_level > self.max_api_level:
            return False
        if self.device_codenames:
            if not target.device_codename or target.device_codename.casefold() not in {
                item.casefold() for item in self.device_codenames
            }:
                return False
        if self.management_modes and target.management_mode.casefold() not in {
            item.casefold() for item in self.management_modes
        }:
            return False
        if self.release_min is not None and target.os_release is None:
            return False
        if self.release_max is not None and target.os_release is None:
            return False
        if target.os_release is not None:
            if self.release_min is not None and _version_key(target.os_release) < _version_key(self.release_min):
                return False
            if self.release_max is not None and _version_key(target.os_release) > _version_key(self.release_max):
                return False
        return True

    @property
    def specificity(self) -> int:
        return 10 + 4 * bool(self.device_codenames) + 3 * bool(self.management_modes) + 2 * bool(
            self.release_min or self.release_max
        ) + bool(self.min_api_level or self.max_api_level)


@dataclass(frozen=True)
class EvidenceRef:
    source_id: str
    snapshot_sha256: str
    confidence_tier: int
    locator: Mapping[str, object] = field(default_factory=dict)
    excerpt_sha256: str | None = None

    def __post_init__(self) -> None:
        if not 0 <= int(self.confidence_tier) <= 6:
            raise ValueError("confidence_tier must be in [0, 6]")
        if not self.source_id or not self.snapshot_sha256:
            raise ValueError("evidence must identify its source and immutable snapshot")


@dataclass(frozen=True)
class SettingCapability:
    setting_id: str
    canonical_name: str
    display_name: str
    aliases: tuple[str, ...]
    scope: VariantScope
    operation: str
    control_plane: ControlPlane
    ui_path: tuple[str, ...]
    required_role: str | None
    requires_user_interaction: bool
    reversible: bool
    risk_level: RiskLevel
    verification_text: str | None
    evidence: tuple[EvidenceRef, ...]
    current_value: object | None = None
    proposed_value: object | None = None

    @property
    def best_evidence_tier(self) -> int:
        return min((item.confidence_tier for item in self.evidence), default=6)


class SettingsCatalog(Protocol):
    def resolve(self, goal: str, target: TargetScope) -> SettingCapability | None:
        ...


class InMemorySettingsCatalog:
    """Small adapter for normalized ORM exports and tests.

    Production code can implement SettingsCatalog over the project's ORM. This
    adapter deliberately accepts facts only; it does not parse arbitrary
    documentation or execute document-provided commands.
    """

    def __init__(self, capabilities: Iterable[SettingCapability], *, max_evidence_tier: int = 2):
        self.capabilities = tuple(capabilities)
        self.max_evidence_tier = int(max_evidence_tier)

    def resolve(self, goal: str, target: TargetScope) -> SettingCapability | None:
        goal_norm = _normalize(goal)
        candidates: list[tuple[float, int, int, SettingCapability]] = []
        for capability in self.capabilities:
            if capability.best_evidence_tier > self.max_evidence_tier:
                continue
            if not capability.scope.matches(target):
                continue
            names = (capability.canonical_name, capability.display_name, *capability.aliases)
            score = max((_goal_score(goal_norm, _normalize(name)) for name in names), default=0.0)
            if score <= 0:
                continue
            candidates.append(
                (score, capability.scope.specificity, -capability.best_evidence_tier, capability)
            )
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[:3], reverse=True)
        best = candidates[0]
        if best[0] < 0.45:
            return None
        if len(candidates) > 1 and candidates[1][:3] == best[:3] and candidates[1][3].setting_id != best[3].setting_id:
            return None
        return best[3]

    @classmethod
    def from_normalized_claims(
        cls,
        documents: Iterable[Mapping[str, object]],
        *,
        max_evidence_tier: int = 2,
    ) -> "InMemorySettingsCatalog":
        capabilities: list[SettingCapability] = []
        for document in documents:
            doc_meta = document.get("document") or {}
            source_id = str(doc_meta.get("source_id") or "")
            snapshot = str(doc_meta.get("snapshot_sha256") or "")
            tier = int(doc_meta.get("confidence_tier", 2))
            for claim in document.get("claims") or ():
                if not isinstance(claim, Mapping) or claim.get("claim_kind") != "setting_variant":
                    continue
                control_plane = _parse_control_plane(str(claim.get("control_plane") or "unknown"))
                scope = claim.get("scope") or {}
                evidence = claim.get("evidence") or {}
                capabilities.append(
                    SettingCapability(
                        setting_id=str(claim.get("canonical_setting_hint") or claim.get("display_name") or ""),
                        canonical_name=str(claim.get("canonical_setting_hint") or claim.get("display_name") or ""),
                        display_name=str(claim.get("display_name") or ""),
                        aliases=tuple(str(item) for item in claim.get("aliases") or ()),
                        scope=VariantScope(
                            os_family=str(scope.get("os_slug") or ""),
                            min_android_version=_optional_int(scope.get("android_version_min")),
                            max_android_version=_optional_int(scope.get("android_version_max")),
                            min_api_level=_optional_int(scope.get("api_level_min")),
                            max_api_level=_optional_int(scope.get("api_level_max")),
                            release_min=_optional_str(scope.get("release_min")),
                            release_max=_optional_str(scope.get("release_max")),
                            device_codenames=tuple(str(item) for item in scope.get("device_codenames") or ()),
                            management_modes=tuple(str(item) for item in scope.get("management_modes") or ()),
                        ),
                        operation=str(claim.get("operation") or "set"),
                        control_plane=control_plane,
                        ui_path=tuple(str(item) for item in claim.get("ui_path") or ()),
                        required_role=_optional_str(claim.get("required_role")),
                        requires_user_interaction=str(claim.get("write_capability") or "").casefold()
                        in {"user_interaction_required", "manual_only"},
                        reversible=bool(claim.get("reversible", False)),
                        risk_level=_parse_risk(str(claim.get("risk_level") or "medium")),
                        verification_text=_optional_str(claim.get("verification_text") or claim.get("display_name")),
                        evidence=(
                            EvidenceRef(
                                source_id=source_id,
                                snapshot_sha256=snapshot,
                                confidence_tier=tier,
                                locator=dict(evidence.get("source_locator") or {}),
                            ),
                        ),
                        proposed_value=claim.get("proposed_value"),
                    )
                )
        return cls(capabilities, max_evidence_tier=max_evidence_tier)


@dataclass(frozen=True)
class WorkflowStep:
    step_type: WorkflowStepType
    label: str
    subgoal: PlannerSubgoal | None = None
    confirmation_required: bool = False
    verification: VerificationExpectation | None = None


@dataclass(frozen=True)
class WorkflowManifest:
    workflow_id: str
    goal: str
    target: TargetScope
    capability: SettingCapability
    steps: tuple[WorkflowStep, ...]
    evidence_revision: tuple[str, ...]
    risk_level: RiskLevel
    reversible: bool


class EvidenceBackedWorkflowPlanner:
    """Creates only allowlisted, evidence-backed user-action workflows."""

    _UI_PLANES = {ControlPlane.SETTINGS_PANEL, ControlPlane.UI_ONLY, ControlPlane.MANUAL_ONLY}

    def build(self, goal: str, target: TargetScope, capability: SettingCapability) -> WorkflowManifest:
        if capability.best_evidence_tier > 2:
            raise ValueError("executable workflow requires Tier 0-2 evidence")
        if not capability.scope.matches(target):
            raise ValueError("setting capability does not match the exact target scope")
        if capability.control_plane not in self._UI_PLANES:
            raise ValueError(
                f"control plane {capability.control_plane.value} needs a separately approved executor"
            )
        if not capability.ui_path:
            raise ValueError("UI workflow requires a documented ui_path")

        confirmation = capability.requires_user_interaction or capability.risk_level in {
            RiskLevel.MEDIUM,
            RiskLevel.HIGH,
            RiskLevel.CRITICAL,
        }
        steps: list[WorkflowStep] = [
            WorkflowStep(WorkflowStepType.CHECK_TARGET, capability.display_name),
        ]
        if capability.required_role:
            steps.append(WorkflowStep(WorkflowStepType.CHECK_PERMISSION, capability.required_role))

        navigation = capability.ui_path[:-1]
        final_label = capability.ui_path[-1]
        for index, label in enumerate(navigation):
            expected_label = capability.ui_path[index + 1]
            steps.append(
                WorkflowStep(
                    WorkflowStepType.OPEN_SETTINGS_PANEL
                    if index == 0
                    else WorkflowStepType.REQUEST_USER_ACTION,
                    label,
                    PlannerSubgoal(SubgoalType.FIND_SETTING, label),
                    confirmation_required=False,
                    verification=VerificationExpectation(
                        expected_text=expected_label,
                        require_tree_change=True,
                        wait_for_idle=False,
                    ),
                )
            )
        steps.append(
            WorkflowStep(
                WorkflowStepType.REQUEST_USER_ACTION,
                final_label,
                PlannerSubgoal(SubgoalType.TAP_TARGET, final_label),
                confirmation_required=confirmation,
                verification=VerificationExpectation(
                    expected_text=capability.verification_text or final_label,
                    require_tree_change=True,
                    wait_for_idle=False,
                ),
            )
        )
        steps.append(
            WorkflowStep(
                WorkflowStepType.VERIFY_VALUE,
                capability.verification_text or capability.display_name,
                verification=VerificationExpectation(
                    expected_text=capability.verification_text or capability.display_name,
                    require_tree_change=True,
                ),
            )
        )
        steps.append(WorkflowStep(WorkflowStepType.REPORT, capability.display_name))

        evidence_revision = tuple(sorted({item.snapshot_sha256 for item in capability.evidence}))
        raw_id = json.dumps(
            {
                "goal": goal,
                "target": target.__dict__,
                "setting": capability.setting_id,
                "evidence": evidence_revision,
            },
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        workflow_id = hashlib.sha256(raw_id.encode()).hexdigest()[:20]
        return WorkflowManifest(
            workflow_id=workflow_id,
            goal=goal,
            target=target,
            capability=capability,
            steps=tuple(steps),
            evidence_revision=evidence_revision,
            risk_level=capability.risk_level,
            reversible=capability.reversible,
        )


@dataclass(frozen=True)
class ProposedAction:
    action_id: str
    workflow_id: str
    step_index: int
    target_label: str
    grounding: GroundingDecision
    requires_confirmation: bool


@dataclass(frozen=True)
class ActionReceipt:
    action_id: str
    success: bool
    before: AndroidState
    after: AndroidState
    user_confirmed: bool
    error: str | None = None


@dataclass
class CoordinatorSession:
    manifest: WorkflowManifest
    next_step_index: int = 0
    pending_action: ProposedAction | None = None
    status: SessionStatus = SessionStatus.READY
    receipts: list[ActionReceipt] = field(default_factory=list)


@dataclass(frozen=True)
class CoordinatorTurn:
    status: SessionStatus
    message: str
    session: CoordinatorSession | None
    proposed_action: ProposedAction | None = None
    grounding: GroundingDecision | None = None


class SettingsCoordinator:
    """Coordinates ORM resolution, VLM grounding and one-step LLM chat."""

    def __init__(
        self,
        catalog: SettingsCatalog,
        *,
        planner: EvidenceBackedWorkflowPlanner | None = None,
        grounder: HybridGrounder | None = None,
        advisor: PlannerAdvisor | None = None,
    ):
        self.catalog = catalog
        self.planner = planner or EvidenceBackedWorkflowPlanner()
        self.grounder = grounder or HybridGrounder()
        self.advisor = advisor or PlannerAdvisor()

    def start(
        self,
        goal: str,
        target: TargetScope,
        state: AndroidState,
        *,
        llm: LocalChatModel | None = None,
        vlm: LocalVisionLanguageModel | None = None,
        drlqn_node_scores: Sequence[float] | None = None,
    ) -> CoordinatorTurn:
        capability = self.catalog.resolve(goal, target)
        if capability is None:
            return CoordinatorTurn(
                SessionStatus.BLOCKED,
                "Нет точного evidence-backed варианта настройки для этой ОС, версии и модели устройства.",
                None,
            )
        try:
            manifest = self.planner.build(goal, target, capability)
        except ValueError as error:
            return CoordinatorTurn(SessionStatus.BLOCKED, str(error), None)
        session = CoordinatorSession(manifest)
        return self.next_turn(
            session,
            state,
            llm=llm,
            vlm=vlm,
            drlqn_node_scores=drlqn_node_scores,
        )

    def next_turn(
        self,
        session: CoordinatorSession,
        state: AndroidState,
        *,
        llm: LocalChatModel | None = None,
        vlm: LocalVisionLanguageModel | None = None,
        drlqn_node_scores: Sequence[float] | None = None,
    ) -> CoordinatorTurn:
        if session.status is SessionStatus.DONE:
            return CoordinatorTurn(SessionStatus.DONE, "Задача уже завершена.", session)
        if session.pending_action is not None:
            return CoordinatorTurn(
                session.status,
                "Сначала подтвердите или отклоните текущее действие.",
                session,
                session.pending_action,
                session.pending_action.grounding,
            )

        while session.next_step_index < len(session.manifest.steps):
            step = session.manifest.steps[session.next_step_index]
            if step.step_type in {WorkflowStepType.CHECK_TARGET, WorkflowStepType.CHECK_PERMISSION}:
                session.next_step_index += 1
                continue
            if step.step_type is WorkflowStepType.VERIFY_VALUE:
                return CoordinatorTurn(
                    SessionStatus.BLOCKED,
                    "Ожидается receipt предыдущего действия для проверки результата.",
                    session,
                )
            if step.step_type is WorkflowStepType.REPORT:
                session.next_step_index += 1
                session.status = SessionStatus.DONE
                return CoordinatorTurn(
                    SessionStatus.DONE,
                    f"Настройка «{session.manifest.capability.display_name}» подтверждена.",
                    session,
                )
            if step.subgoal is None:
                session.status = SessionStatus.BLOCKED
                return CoordinatorTurn(SessionStatus.BLOCKED, "Шаг не содержит семантической подцели.", session)

            grounding = self.grounder.ground(
                state,
                step.label,
                vlm=vlm,
                drlqn_node_scores=drlqn_node_scores,
            )
            if not grounding.grounded:
                session.status = SessionStatus.BLOCKED
                return CoordinatorTurn(
                    SessionStatus.BLOCKED,
                    f"Не удалось надежно найти элемент «{step.label}» на текущем экране.",
                    session,
                    grounding=grounding,
                )
            action_id = hashlib.sha256(
                f"{session.manifest.workflow_id}:{session.next_step_index}:{grounding.box}".encode()
            ).hexdigest()[:20]
            proposed = ProposedAction(
                action_id=action_id,
                workflow_id=session.manifest.workflow_id,
                step_index=session.next_step_index,
                target_label=step.label,
                grounding=grounding,
                requires_confirmation=step.confirmation_required,
            )
            session.pending_action = proposed
            session.status = (
                SessionStatus.WAITING_FOR_CONFIRMATION
                if proposed.requires_confirmation
                else SessionStatus.WAITING_FOR_RECEIPT
            )
            reply = self.advisor.advise((step.subgoal,), state, session.manifest.goal, llm=llm)
            message = reply.step.instruction if reply.step else reply.message
            if proposed.requires_confirmation:
                message += " Подтвердите выполнение этого изменения перед действием."
            return CoordinatorTurn(session.status, message, session, proposed, grounding)

        session.status = SessionStatus.DONE
        return CoordinatorTurn(SessionStatus.DONE, "Задача завершена.", session)

    def confirm_action(self, session: CoordinatorSession, *, approved: bool) -> CoordinatorTurn:
        pending = session.pending_action
        if pending is None:
            return CoordinatorTurn(SessionStatus.BLOCKED, "Нет ожидающего подтверждения действия.", session)
        if not pending.requires_confirmation:
            session.status = SessionStatus.WAITING_FOR_RECEIPT
            return CoordinatorTurn(session.status, "Действие уже готово к выполнению.", session, pending)
        if not approved:
            session.pending_action = None
            session.status = SessionStatus.BLOCKED
            return CoordinatorTurn(SessionStatus.BLOCKED, "Изменение отменено пользователем.", session)
        session.status = SessionStatus.WAITING_FOR_RECEIPT
        return CoordinatorTurn(
            SessionStatus.WAITING_FOR_RECEIPT,
            "Подтверждение получено. Выполните предложенное действие и верните receipt.",
            session,
            pending,
            pending.grounding,
        )

    def submit_receipt(
        self,
        session: CoordinatorSession,
        receipt: ActionReceipt,
        *,
        llm: LocalChatModel | None = None,
        vlm: LocalVisionLanguageModel | None = None,
        drlqn_node_scores: Sequence[float] | None = None,
    ) -> CoordinatorTurn:
        pending = session.pending_action
        if pending is None or receipt.action_id != pending.action_id:
            return CoordinatorTurn(SessionStatus.BLOCKED, "Receipt не соответствует ожидаемому действию.", session)
        if pending.requires_confirmation and not receipt.user_confirmed:
            return CoordinatorTurn(SessionStatus.BLOCKED, "Для этого действия нет подтверждения пользователя.", session)
        if not receipt.success:
            session.pending_action = None
            session.status = SessionStatus.BLOCKED
            return CoordinatorTurn(
                SessionStatus.BLOCKED,
                f"Действие не выполнено: {receipt.error or 'неизвестная ошибка'}.",
                session,
            )

        step = session.manifest.steps[pending.step_index]
        expected = step.verification or VerificationExpectation(
            expected_text=step.label,
            require_tree_change=True,
            wait_for_idle=False,
        )
        result = verify_transition(receipt.before, receipt.after, expected)
        if not result.success:
            session.status = SessionStatus.BLOCKED
            return CoordinatorTurn(
                SessionStatus.BLOCKED,
                "Интерфейс не подтвердил ожидаемое изменение: " + ", ".join(result.reasons),
                session,
            )

        session.receipts.append(receipt)
        session.pending_action = None
        session.next_step_index = pending.step_index + 1
        if (
            session.next_step_index < len(session.manifest.steps)
            and session.manifest.steps[session.next_step_index].step_type is WorkflowStepType.VERIFY_VALUE
        ):
            session.next_step_index += 1
        session.status = SessionStatus.READY
        return self.next_turn(
            session,
            receipt.after,
            llm=llm,
            vlm=vlm,
            drlqn_node_scores=drlqn_node_scores,
        )


def _normalize(value: str) -> str:
    return " ".join((value or "").casefold().replace("_", " ").split())


def _goal_score(goal: str, name: str) -> float:
    if not goal or not name:
        return 0.0
    if goal == name:
        return 1.0
    if name in goal:
        return 0.94
    goal_tokens = set(goal.split())
    name_tokens = set(name.split())
    if not name_tokens:
        return 0.0
    return len(goal_tokens & name_tokens) / len(name_tokens)


def _version_key(value: str) -> tuple[tuple[int, object], ...]:
    parts: list[tuple[int, object]] = []
    for part in str(value).replace("-", ".").split("."):
        parts.append((0, int(part)) if part.isdigit() else (1, part.casefold()))
    return tuple(parts)


def _optional_int(value: object) -> int | None:
    return None if value is None or value == "" else int(value)


def _optional_str(value: object) -> str | None:
    return None if value is None or value == "" else str(value)


def _parse_risk(value: str) -> RiskLevel:
    try:
        return RiskLevel(value.casefold())
    except ValueError:
        return RiskLevel.MEDIUM


def _parse_control_plane(value: str) -> ControlPlane:
    normalized = value.casefold()
    aliases = {
        "ui_only": ControlPlane.UI_ONLY,
        "settings_panel": ControlPlane.SETTINGS_PANEL,
        "manual_only": ControlPlane.MANUAL_ONLY,
        "android_public_api": ControlPlane.ANDROID_PUBLIC_API,
        "device_policy_manager": ControlPlane.DEVICE_POLICY_MANAGER,
        "android_management_api": ControlPlane.ANDROID_MANAGEMENT_API,
        "os_specific_api": ControlPlane.OS_SPECIFIC_API,
        "shell_command": ControlPlane.DOCUMENTED_SHELL,
        "settings_system": ControlPlane.SETTINGS_PROVIDER,
        "settings_secure": ControlPlane.SETTINGS_PROVIDER,
        "settings_global": ControlPlane.SETTINGS_PROVIDER,
    }
    return aliases.get(normalized, ControlPlane.UNKNOWN)
