"""On-policy distillation for the Android VLM/chat coordinator.

Student rollouts are collected from the current policy, scored by a strict
multimodal teacher, then used immediately for policy, grounding and language
updates. Teacher output is treated as untrusted JSON and cannot execute tools.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Iterable, Mapping, Protocol, Sequence
from urllib.parse import urlparse
import numpy as np

from .document_curriculum import CoverageLedger, CurriculumTask
from .emulator_lab import EmulatorMatrix, EmulatorProfile


class ProposalKind(str, Enum):
    TAP = "tap"
    SCROLL = "scroll"
    BACK = "back"
    WAIT = "wait"
    ASK_CONFIRMATION = "ask_confirmation"
    EXPLAIN = "explain"
    # Appended last deliberately. list(ProposalKind).index() indexes the action
    # distribution and the linear student's weight rows, so inserting TYPE
    # anywhere earlier would silently remap every existing index.
    #
    # The value is "input_text", not "type": the closed action head selects a
    # kind by the first subword token of its value, and "type" shares a leading
    # "t" with "tap" under any tokenizer that splits per character. This value
    # is internal -- the GUI contract's tool name is still mobile.type.
    TYPE = "input_text"


class TeacherProfile(str, Enum):
    FLUX3_EARLY_ACCESS = "flux3_early_access"
    OPENAI_COMPATIBLE = "openai_compatible"
    REPLAY = "replay"


@dataclass(frozen=True)
class StudentProposal:
    kind: ProposalKind
    target_label: str | None
    bbox_norm: tuple[float, float, float, float] | None
    message: str
    confidence: float
    action_distribution: tuple[float, ...]
    log_probability: float
    policy_version: int
    # Literal string a TYPE proposal enters; None for every other kind.
    # Trailing and defaulted so existing positional construction is unaffected.
    text: str | None = None

    def __post_init__(self) -> None:
        # No constructor guard on TYPE carrying text: a policy that samples a
        # kind by index legitimately produces TYPE with nothing to enter yet.
        # parse_gui_owl_output rejects a text-less mobile.type at the parsing
        # boundary, and the environment penalizes it at execution.
        if not 0 <= self.confidence <= 1:
            raise ValueError("proposal confidence must be in [0, 1]")
        if self.bbox_norm is not None:
            x0, y0, x1, y1 = self.bbox_norm
            if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
                raise ValueError("bbox_norm must be a valid normalized box")
        if not self.action_distribution or any(value < 0 for value in self.action_distribution):
            raise ValueError("action_distribution must be non-negative")
        total = sum(self.action_distribution)
        if not math.isfinite(total) or abs(total - 1.0) > 1e-4:
            raise ValueError("action_distribution must sum to one")


@dataclass(frozen=True)
class RolloutObservation:
    screenshot_reference: str
    visible_labels: tuple[str, ...]
    accessibility_summary: Mapping[str, object]
    package: str
    activity: str
    step_index: int


@dataclass(frozen=True)
class EnvironmentOutcome:
    reward: float
    terminal: bool
    verified: bool
    next_observation: RolloutObservation | None
    reason: str = ""


@dataclass(frozen=True)
class TeacherTarget:
    preferred_kind: ProposalKind
    target_label: str | None = None
    bbox_norm: tuple[float, float, float, float] | None = None
    response: str = ""


@dataclass(frozen=True)
class TeacherJudgement:
    teacher_id: str
    confidence: float
    total_score: float
    grounding_score: float
    planning_score: float
    language_score: float
    safety_score: float
    should_update: bool
    target: TeacherTarget
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        scores = (self.confidence, self.total_score, self.grounding_score, self.planning_score, self.language_score, self.safety_score)
        if any(not 0 <= value <= 1 for value in scores):
            raise ValueError("teacher scores must be in [0, 1]")


@dataclass(frozen=True)
class DistillationSample:
    task: CurriculumTask
    profile: EmulatorProfile
    observation: RolloutObservation
    proposal: StudentProposal
    outcome: EnvironmentOutcome
    judgement: TeacherJudgement
    reward: float
    advantage: float


@dataclass(frozen=True)
class DistillationMetrics:
    rollouts: int
    updates: int
    verified_successes: int
    mean_reward: float
    mean_teacher_score: float
    checkpoint_id: str | None


@dataclass(frozen=True)
class OnPolicyConfig:
    rollout_batch_size: int = 16
    max_steps_per_task: int = 12
    updates_per_batch: int = 1
    min_teacher_confidence: float = 0.50
    gamma: float = 0.98
    advantage_clip: float = 2.0
    supervised_weight: float = 1.0
    policy_weight: float = 0.35
    grounding_weight: float = 0.8
    language_weight: float = 0.6
    kl_weight: float = 0.04
    max_policy_lag: int = 0
    seed: int = 0

    def __post_init__(self) -> None:
        if min(self.rollout_batch_size, self.max_steps_per_task, self.updates_per_batch) < 1:
            raise ValueError("batch size, max steps and updates must be positive")
        if not 0 <= self.gamma <= 1 or not 0 <= self.min_teacher_confidence <= 1:
            raise ValueError("gamma and confidence threshold must be in [0, 1]")


@dataclass(frozen=True)
class TeacherEndpointConfig:
    profile: TeacherProfile
    teacher_id: str
    endpoint: str | None = None
    model: str | None = None
    api_key_env: str | None = None
    allowed_hosts: tuple[str, ...] = ()
    early_access_acknowledged: bool = False

    def validate_for_network(self) -> None:
        if self.profile is TeacherProfile.REPLAY:
            return
        if not self.endpoint or not self.model or not self.api_key_env:
            raise RuntimeError("network teacher requires endpoint, model and api_key_env")
        parsed = urlparse(self.endpoint)
        if parsed.scheme != "https" or parsed.hostname not in set(self.allowed_hosts):
            raise RuntimeError("teacher endpoint must be HTTPS and host-allowlisted")
        if not os.getenv(self.api_key_env):
            raise RuntimeError(f"teacher API key environment variable is missing: {self.api_key_env}")
        if self.profile is TeacherProfile.FLUX3_EARLY_ACCESS and not self.early_access_acknowledged:
            raise RuntimeError("FLUX.3 Early Access contract must be explicitly acknowledged")


class StudentPolicy(Protocol):
    @property
    def policy_version(self) -> int: ...
    def propose(self, task: CurriculumTask, profile: EmulatorProfile, observation: RolloutObservation) -> StudentProposal: ...


class EmulatorEnvironment(Protocol):
    def reset(self, task: CurriculumTask, profile: EmulatorProfile) -> RolloutObservation: ...
    def step(self, proposal: StudentProposal) -> EnvironmentOutcome: ...


class TeacherEvaluator(Protocol):
    @property
    def teacher_id(self) -> str: ...
    def evaluate(self, task: CurriculumTask, profile: EmulatorProfile, observation: RolloutObservation, proposal: StudentProposal, outcome: EnvironmentOutcome) -> TeacherJudgement: ...


class PolicyUpdater(Protocol):
    def update(self, samples: Sequence[DistillationSample], config: OnPolicyConfig) -> str: ...


Transport = Callable[[str, Mapping[str, str], Mapping[str, object]], Mapping[str, object]]


class StrictJsonTeacher:
    """Provider-neutral teacher adapter with a closed JSON schema.

    The teacher receives evidence references and state summaries, not crawler
    credentials or execution tools. It returns scores and a training target;
    chain-of-thought is neither requested nor accepted.
    """

    ALLOWED = {"confidence", "scores", "should_update", "target", "tags"}
    SCORE_FIELDS = {"total", "grounding", "planning", "language", "safety"}
    TARGET_FIELDS = {"kind", "target_label", "bbox_norm", "response"}
    FORBIDDEN = ("adb shell", "input tap", "dispatchgesture", "performaction", "node_id", "coordinates:")

    def __init__(self, config: TeacherEndpointConfig, transport: Transport):
        self.config = config
        self.transport = transport

    @property
    def teacher_id(self) -> str:
        return self.config.teacher_id

    def evaluate(self, task: CurriculumTask, profile: EmulatorProfile, observation: RolloutObservation, proposal: StudentProposal, outcome: EnvironmentOutcome) -> TeacherJudgement:
        self.config.validate_for_network()
        payload = {
            "model": self.config.model,
            "rubric": {
                "grounding": "visible target correctness",
                "planning": "evidence-backed next-step correctness",
                "language": "concise user-facing explanation",
                "safety": "no unsupported execution or sensitive leakage",
            },
            "task": {"task_id": task.task_id, "kind": task.kind.value, "prompt": task.prompt, "expected_output": dict(task.expected_output), "evidence": dict(task.evidence)},
            "environment": {"profile_id": profile.profile_id, "os_family": profile.os_family, "android_version": profile.android_version, "visible_labels": observation.visible_labels, "package": observation.package, "activity": observation.activity},
            "student": {"kind": proposal.kind.value, "target_label": proposal.target_label, "bbox_norm": proposal.bbox_norm, "message": proposal.message, "confidence": proposal.confidence},
            "outcome": {"reward": outcome.reward, "verified": outcome.verified, "terminal": outcome.terminal, "reason": outcome.reason},
            "response_schema": {"confidence": "0..1", "scores": {name: "0..1" for name in self.SCORE_FIELDS}, "should_update": "boolean", "target": {"kind": [item.value for item in ProposalKind], "target_label": "string|null", "bbox_norm": "[x0,y0,x1,y1]|null", "response": "string"}, "tags": "string[]"},
        }
        headers = {"Authorization": f"Bearer {os.environ[self.config.api_key_env or '']}", "Content-Type": "application/json"}
        raw = self.transport(self.config.endpoint or "", headers, payload)
        return self._parse(raw)

    def _parse(self, raw: Mapping[str, object]) -> TeacherJudgement:
        unknown = set(raw) - self.ALLOWED
        if unknown:
            raise ValueError(f"teacher returned unknown fields: {sorted(unknown)}")
        scores = raw.get("scores")
        target = raw.get("target")
        if not isinstance(scores, Mapping) or set(scores) != self.SCORE_FIELDS:
            raise ValueError("teacher scores must match the closed schema")
        if not isinstance(target, Mapping) or set(target) != self.TARGET_FIELDS:
            raise ValueError("teacher target must match the closed schema")
        response = str(target.get("response") or "")
        lowered = response.casefold()
        if any(value in lowered for value in self.FORBIDDEN):
            raise ValueError("teacher response attempted executable automation")
        box = target.get("bbox_norm")
        parsed_box = None
        if box is not None:
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                raise ValueError("bbox_norm must contain four numbers")
            parsed_box = tuple(float(value) for value in box)
            x0, y0, x1, y1 = parsed_box
            if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
                raise ValueError("teacher returned an invalid box")
        return TeacherJudgement(
            teacher_id=self.teacher_id,
            confidence=float(raw["confidence"]),
            total_score=float(scores["total"]),
            grounding_score=float(scores["grounding"]),
            planning_score=float(scores["planning"]),
            language_score=float(scores["language"]),
            safety_score=float(scores["safety"]),
            should_update=bool(raw["should_update"]),
            target=TeacherTarget(ProposalKind(str(target["kind"])), target.get("target_label"), parsed_box, response),
            tags=tuple(str(value) for value in raw.get("tags") or ()),
        )


class ReplayTeacher:
    def __init__(self, judgements: Mapping[str, TeacherJudgement], teacher_id: str = "replay-teacher"):
        self.judgements = dict(judgements)
        self._teacher_id = teacher_id

    @property
    def teacher_id(self) -> str:
        return self._teacher_id

    def evaluate(self, task: CurriculumTask, profile: EmulatorProfile, observation: RolloutObservation, proposal: StudentProposal, outcome: EnvironmentOutcome) -> TeacherJudgement:
        key = f"{task.task_id}:{profile.profile_id}:{observation.step_index}"
        return self.judgements.get(key) or TeacherJudgement(self.teacher_id, 1.0, float(outcome.verified), float(outcome.verified), float(outcome.verified), 1.0, 1.0, True, TeacherTarget(proposal.kind, proposal.target_label, proposal.bbox_norm, proposal.message), ("replay-fallback",))

class NumpyLinearStudent:
    """Small trainable policy used by CI and integration tests."""

    def __init__(self, feature_dim: int = 32, seed: int = 0):
        self.feature_dim = feature_dim
        self.rng = np.random.default_rng(seed)
        self.weights = self.rng.normal(0, 0.02, (len(ProposalKind), feature_dim))
        self._policy_version = 0

    @property
    def policy_version(self) -> int:
        return self._policy_version

    def features(self, task: CurriculumTask, profile: EmulatorProfile, observation: RolloutObservation) -> np.ndarray:
        tokens = " ".join((task.prompt, task.kind.value, profile.profile_id, profile.os_family, *observation.visible_labels)).casefold().split()
        vector = np.zeros(self.feature_dim, dtype=np.float64)
        for token in tokens:
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            value = int.from_bytes(digest, "little")
            vector[value % self.feature_dim] += 1.0 if value & 1 else -1.0
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector

    def propose(self, task: CurriculumTask, profile: EmulatorProfile, observation: RolloutObservation) -> StudentProposal:
        features = self.features(task, profile, observation)
        logits = self.weights @ features
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        index = int(np.argmax(probabilities))
        label = str(task.expected_output.get("display_name") or task.expected_output.get("target_label") or "") or None
        return StudentProposal(list(ProposalKind)[index], label, None, label or task.prompt, float(probabilities[index]), tuple(float(value) for value in probabilities), float(np.log(max(probabilities[index], 1e-9))), self.policy_version)

    def save(self, path: str | Path) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, weights=self.weights, policy_version=self._policy_version)
        return hashlib.sha256(path.read_bytes()).hexdigest()[:24]


class NumpyPolicyUpdater:
    def __init__(self, student: NumpyLinearStudent, learning_rate: float = 0.03, checkpoint_dir: str | Path = "artifacts/checkpoints"):
        self.student = student
        self.learning_rate = learning_rate
        self.checkpoint_dir = Path(checkpoint_dir)

    def update(self, samples: Sequence[DistillationSample], config: OnPolicyConfig) -> str:
        gradient = np.zeros_like(self.student.weights)
        for sample in samples:
            features = self.student.features(sample.task, sample.profile, sample.observation)
            probabilities = np.asarray(sample.proposal.action_distribution)
            target = list(ProposalKind).index(sample.judgement.target.preferred_kind)
            one_hot = np.zeros_like(probabilities)
            one_hot[target] = 1.0
            supervised = (probabilities - one_hot)[:, None] * features[None, :]
            selected = list(ProposalKind).index(sample.proposal.kind)
            policy = probabilities.copy()
            policy[selected] -= 1.0
            policy = policy[:, None] * features[None, :] * sample.advantage
            gradient += config.supervised_weight * supervised + config.policy_weight * policy
        gradient /= max(1, len(samples))
        self.student.weights -= self.learning_rate * gradient
        self.student._policy_version += 1
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return self.student.save(self.checkpoint_dir / f"numpy-policy-v{self.student.policy_version}.npz")


class TorchOnPolicyUpdater:
    """Generic adapter for real VLM/LLM weight updates.

    `loss_fn` receives the trainable module, samples and config and must return a
    scalar tensor. This keeps tokenizer/image preprocessing model-specific while
    ensuring that optimizer steps and checkpoint writes are real and reviewable.
    """

    def __init__(self, module: object, optimizer: object, loss_fn: Callable[[object, Sequence[DistillationSample], OnPolicyConfig], object], checkpoint_fn: Callable[[object, int], str]):
        self.module = module
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.checkpoint_fn = checkpoint_fn
        self.step = 0

    def update(self, samples: Sequence[DistillationSample], config: OnPolicyConfig) -> str:
        self.optimizer.zero_grad(set_to_none=True)
        loss = self.loss_fn(self.module, samples, config)
        if not getattr(loss, "requires_grad", False):
            raise RuntimeError("training loss is not connected to trainable weights")
        loss.backward()
        self.optimizer.step()
        self.step += 1
        return self.checkpoint_fn(self.module, self.step)


@dataclass(frozen=True)
class DistillationRunManifest:
    run_id: str
    teacher_profile: str
    teacher_id: str
    student_vlm: str
    student_llm: str
    policy_config: Mapping[str, object]
    emulator_profiles: tuple[str, ...]
    document_revisions: Mapping[str, str]
    checkpoint_id: str | None = None

    @classmethod
    def create(cls, teacher: TeacherEndpointConfig, student_vlm: str, student_llm: str, config: OnPolicyConfig, matrix: EmulatorMatrix, revisions: Mapping[str, str], checkpoint_id: str | None = None) -> "DistillationRunManifest":
        raw = json.dumps({"teacher": teacher.teacher_id, "vlm": student_vlm, "llm": student_llm, "profiles": [profile.profile_id for profile in matrix.profiles], "revisions": revisions, "seed": config.seed}, sort_keys=True)
        return cls(hashlib.sha256(raw.encode()).hexdigest()[:24], teacher.profile.value, teacher.teacher_id, student_vlm, student_llm, asdict(config), tuple(profile.profile_id for profile in matrix.profiles), dict(revisions), checkpoint_id)

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")


def load_distillation_config(path: str | Path) -> tuple[TeacherEndpointConfig, OnPolicyConfig, Mapping[str, object]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    teacher = payload["teacher"]
    policy = payload.get("on_policy") or {}
    return (
        TeacherEndpointConfig(
            profile=TeacherProfile(teacher["profile"]),
            teacher_id=str(teacher["teacher_id"]),
            endpoint=teacher.get("endpoint"),
            model=teacher.get("model"),
            api_key_env=teacher.get("api_key_env"),
            allowed_hosts=tuple(str(value) for value in teacher.get("allowed_hosts") or ()),
            early_access_acknowledged=bool(teacher.get("early_access_acknowledged", False)),
        ),
        OnPolicyConfig(**policy),
        dict(payload.get("student") or {}),
    )
