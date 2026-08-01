"""Android DRLQN with evidence-backed coordination and on-policy distillation."""

from .advisor import (
    AdviceStatus, AdvisorConfig, AdvisorPolicyError, AdvisorReply, ChatMessage,
    GenerationOptions, LocalChatModel, PlannerAdvisor, SensitiveTextRedactor,
    UserActionStep,
)
from .advisory_model import AdvisoryDRLQN, AdvisoryModelConfig, PerceptionDRLQN, PerceptionModelConfig
from .core import (
    AccessibilityNode,
    AccessibilitySnapshot,
    AlignmentError,
    AndroidBatchConfig,
    AndroidState,
    Bounds,
    Caption,
    GestureAction,
    GestureType,
    ScreenshotFrame,
    SynchronizedState,
    TimestampSynchronizer,
    UIEvent,
    VerificationExpectation,
    VerificationResult,
    ViewportMetadata,
    normalize_screenshot,
    prepare_model_inputs,
    tree_fingerprint,
    verify_transition,
)
from .device_profiles import (
    BackendDecision, BenchmarkSample, DeviceCapabilities, DeviceFamily,
    DeviceProfileSelector, DeviceTier, InferenceProfile, ModelFamily,
    Quantization, RuntimeBackend, SocVendor,
)
from .distillation import (
    DistillationMetrics, DistillationRunManifest, DistillationSample,
    EmulatorEnvironment, EnvironmentOutcome, NumpyLinearStudent,
    NumpyPolicyUpdater, OnPolicyConfig, PolicyUpdater, ProposalKind,
    ReplayTeacher, RolloutObservation, StrictJsonTeacher, StudentPolicy,
    StudentProposal, TeacherEndpointConfig, TeacherEvaluator,
    TeacherJudgement, TeacherProfile, TeacherTarget, TorchOnPolicyUpdater,
    load_distillation_config,
)
from .document_curriculum import (
    CoverageEntry, CoverageLedger, CoverageStatus, CurriculumTask,
    CurriculumTaskKind, DocumentClass, DocumentSection,
    DocumentationCoverageReport, DocumentationCurriculumBuilder,
    SourceDescriptor, load_source_registry,
)
from .emulator_lab import (
    DisplaySpec, EmulatorCoverageReport, EmulatorDeviceTier, EmulatorKind,
    EmulatorMatrix, EmulatorProfile, FormFactor, load_emulator_matrix,
    selected_emulator_matrix,
)
from .emulator_runtime import (
    AdbClient, BootStatus, CommandResult, EmulatorBootError,
    EmulatorBootTimeout, EmulatorExitedError, EmulatorLauncher, EmulatorLease,
    serial_for_port,
)
from .emulator import (
    AdbClient,
    AdbEmulatorEnvironment,
    AdbResearchActuator,
    EmulatorError,
    EmulatorFleet,
    EmulatorProfile,
    load_profiles,
    parse_ui_dump,
)
from .model import AndroidDRLQN, AndroidDRLQNConfig
from .model_launch import (
    DistillationMode, ModelLaunchBundle, ModelLaunchConfig, OSLaunchProfile,
    OSSettingRecord, OSSettingsRegistry, aggregate_distillation_metrics,
)
from .ollama import (
    GROUNDING_SCHEMA,
    SUGGESTED_TEACHERS,
    OllamaChatModel,
    OllamaClient,
    OllamaConfig,
    OllamaError,
    OllamaGroundingVLM,
    OllamaMultimodalGenerator,
    normalize_grounding_payload,
    ollama_grounding_vlm,
)
from .on_policy import OnPolicyDistiller
from .runtime_config import (
    BuildManifest,
    litert_export_manifest,
    onnxruntime_android_manifest,
    write_manifest,
)
from .ollama_runtime import (
    GROUNDING_SCHEMA, JUDGEMENT_SCHEMA, OllamaChatModel, OllamaClient,
    OllamaEnsembleConfig, OllamaModelSpec, OllamaModelStatus,
    OllamaMultimodalGenerator, OllamaQwenTeacher, OllamaRuntimeBundle,
    OllamaRuntimeError, OllamaServerConfig, OllamaTeacherEnsemble,
    OllamaUITarsCritic, UITarsAction, critic_agreement,
    load_image_reference, parse_teacher_judgement, parse_ui_tars_action,
    smoke_image_base64,
)
from .runtime_config import BuildManifest, litert_export_manifest, onnxruntime_android_manifest, write_manifest
from .settings_agent import (
    ActionReceipt, ControlPlane, CoordinatorSession, CoordinatorTurn,
    EvidenceBackedWorkflowPlanner, EvidenceRef, InMemorySettingsCatalog,
    ProposedAction, RiskLevel, SessionStatus, SettingCapability,
    SettingsCatalog, SettingsCoordinator, TargetScope, VariantScope,
    WorkflowManifest, WorkflowStep, WorkflowStepType,
)
from .torch_student import JointOutputNames, JointTorchVlmLlmStudent, apply_lora, make_joint_loss
from .vlm import (
    GrounderConfig, GroundingDecision, GroundingSource, HybridGrounder,
    JsonGroundingVLM, LocalVisionLanguageModel, MultimodalGenerator,
    NormalizedBox, VLMContractError, VLMRequest, VisibleElement,
)
from .window_graph import (
    PlannerSubgoal,
    SubgoalType,
    WindowEdge,
    WindowGraph,
    WindowNode,
    WindowSignature,
)

__all__ = [name for name in globals() if not name.startswith("_")]
