# Android DRLQN VLM + chat coordinator

Android-only Differential Residual Lambda Q-Network adaptation for evidence-backed Android settings guidance.

The production path connects:

- the multimodal DRLQN planner/perception model;
- an optional local VLM for screenshot grounding;
- a local chat LLM for one-step explanations;
- a normalized ORM catalog of settings and official evidence;
- explicit confirmation and verified action receipts.

The repository does not ship an autonomous `AccessibilityService` executor. A host application may keep the flow manual or attach a separately reviewed executor after policy approval.

## Safety and evidence boundary

1. `SettingsCatalog` resolves a setting only for the exact OS/release/API/device scope.
2. Tier 0-2 immutable evidence is required for a workflow proposal.
3. `PerceptionDRLQN` exposes node/box/heatmap grounding but filters gesture trajectories and timing outputs.
4. `HybridGrounder` prefers accessibility nodes, then DRLQN ranking, then a strict local-VLM fallback.
5. `PlannerAdvisor` explains exactly one semantic action to the user.
6. Medium/high/critical-risk or user-interaction changes require explicit confirmation.
7. The workflow advances only after an `ActionReceipt` passes `verify_transition()`.

Documentation and source text are untrusted evidence, not executable instructions. Shell, Settings Provider, DevicePolicyManager, Android Management API, Shizuku and root mechanisms require separately approved executors.

## Included

- timestamp-aligned screenshot, accessibility, OCR/caption and UI-event state;
- compact multimodal DRLQN with planning, grounding and verification heads;
- chat-only and visual-grounding production facades;
- strict local VLM and chat model interfaces;
- accessibility-first hybrid grounding;
- normalized ORM claim loader and exact-scope settings catalog;
- evidence-backed workflow planner and settings coordinator;
- local chat advisor with sensitive-text redaction;
- device profiles for Android 10-17 and LiteRT-LM/ONNX Runtime deployment manifests;
- benchmark gates for memory, latency, throughput and thermals;
- ADB/logcat data collector for research trajectories;
- local Ollama adapters for a VLM teacher and a chat model;
- declarative emulator profiles with an emulator-only research actuator;
- on-policy distillation from a local VLM teacher into the DRLQN student.

## Install and test

```bash
python -m pip install -e '.[test]'
pytest
```

Optional training and local-Ollama dependencies:

```bash
python -m pip install -e '.[train,test]'
# or only the image encoder required by the Ollama adapter
python -m pip install -e '.[ollama,test]'
```

## VLM + chat settings flow

```python
from android_drlqn import InMemorySettingsCatalog, JsonGroundingVLM, SettingsCoordinator, TargetScope

catalog = InMemorySettingsCatalog.from_normalized_claims(parsed_documents)
coordinator = SettingsCoordinator(catalog)
vlm = JsonGroundingVLM(your_multimodal_runtime)

target = TargetScope(
    os_family="grapheneos",
    os_release="2026.07",
    android_version=16,
    api_level=36,
    device_codename="panther",
)

turn = coordinator.start(
    "Открой панель конфиденциальности",
    target,
    current_state,
    llm=your_local_chat_model,
    vlm=vlm,
    drlqn_node_scores=node_scores,
)
print(turn.message)
print(turn.proposed_action)
```

The Android host performs or presents the proposed action and returns an `ActionReceipt` with timestamp-aligned before/after states. The coordinator never trusts a model statement that an action succeeded.

See [docs/vlm_chat_coordinator.md](docs/vlm_chat_coordinator.md) and [examples/settings_coordinator.py](examples/settings_coordinator.py).

## Local Qwen3-VL-30B and UI-TARS through Ollama

The checked-in runtime accepts only `http://127.0.0.1:11434` (or another loopback spelling on the same port). It does not pull models during tests, imports or training startup.

Start the loopback server and install explicit local aliases:

```bash
bash scripts/start_ollama_local.sh
# in another terminal
bash scripts/setup_ollama_models.sh
```

Validate model presence and multimodal capabilities:

```bash
python scripts/check_ollama_models.py --config config/ollama_models.json
python scripts/check_ollama_models.py --config config/ollama_models.json --smoke
```

Use the same server for chat, visual grounding and the distillation teacher:

```python
from android_drlqn import JsonGroundingVLM, OllamaRuntimeBundle

bundle = OllamaRuntimeBundle.from_json("config/ollama_models.json")
bundle.preflight()

chat_model = bundle.chat_model
vlm = JsonGroundingVLM(bundle.multimodal_generator)
teacher = bundle.teacher
```

Qwen3-VL returns a closed structured judgement. UI-TARS independently emits its native GUI action; disagreement can block a sample from updating weights. Both models must report `vision` through Ollama `/api/show`. A text-only UI-TARS package is rejected rather than silently downgraded.

See [docs/ollama_local_models.md](docs/ollama_local_models.md).

## On-policy VLM/LLM distillation

The distillation lab trains from actions generated by the current student policy:

```text
current student -> emulator receipt -> multimodal teacher -> correction target -> weight update
```

`config/emulator_matrix.json` defines AOSP Android 10/12L/14/15/16/17 plus pinned GrapheneOS, CalyxOS and /e/OS development profiles. The nine profiles expand to 66 screen, locale and navigation variants. Custom-OS profiles remain disabled until their source revision or image artifact is approved and available.

`config/document_sources.json` defines the allowlisted documentation corpus. “Full coverage” is bounded to a pinned discovery snapshot: every discovered section must be validated, explicitly non-actionable or blocked with a reason. An upstream revision change invalidates completion.

FLUX.3 remains represented only as a disabled Early Access teacher profile. The local Ollama ensemble is the usable open-weight path.

Dry-run without contacting a teacher:

```bash
python scripts/train_on_policy.py \
  --documents artifacts/normalized-documents.json \
  --config config/distillation.json \
  --dry-run
```

Replay smoke training:

```bash
python scripts/train_on_policy.py \
  --documents artifacts/normalized-documents.json \
  --config config/replay-distillation.json \
  --fixture tests/fixtures/distillation_environment.json
```

Real training with the local Ollama teacher ensemble:

```bash
python scripts/train_on_policy_ollama.py \
  --documents artifacts/normalized-documents.json \
  --student-plugin project_train:create_student \
  --environment-plugin project_lab:create_environment
```

See [docs/on_policy_distillation.md](docs/on_policy_distillation.md), [docs/ollama_local_models.md](docs/ollama_local_models.md) and [examples/on_policy_distillation.py](examples/on_policy_distillation.py).

## One-step advisor

```python
from android_drlqn import PlannerAdvisor, PlannerSubgoal, SubgoalType

plan = (
    PlannerSubgoal(SubgoalType.FIND_SETTING, "Bluetooth"),
    PlannerSubgoal(SubgoalType.CONFIRM),
)
reply = PlannerAdvisor().advise(plan, current_state, "Включить Bluetooth")
print(reply.step.instruction)
```

## Device profile

```python
from android_drlqn import DeviceCapabilities, DeviceProfileSelector

capabilities = DeviceCapabilities.from_build_info(
    android_version=15,
    manufacturer="Samsung",
    model="Example",
    soc_manufacturer="Qualcomm",
    soc_model="SM8750",
    ram_gb=12,
    cpu_big_cores=4,
    features={"opencl", "nnapi", "qnn", "fp16"},
)
profile = DeviceProfileSelector().select(capabilities)
print(profile.to_dict())
```

Manufacturer and SoC strings never enable an accelerator by themselves. The runtime must be positively detected and pass an on-device benchmark.

## On-policy distillation across emulator profiles

A local Ollama vision model can act as a grounding teacher while the student
drives the emulator itself. Profiles are visited sequentially with one student
checkpoint.

```bash
python scripts/run_distillation.py \
    --profiles config/emulator_profiles.json \
    --tasks config/distillation_tasks.json \
    --teacher qwen2.5vl:7b \
    --output runs/2026-07-26 \
    --allow-emulator-input
```

Input injection is restricted to emulators: `AdbResearchActuator` requires an
explicit `allow_input=True` and verifies through adb that the target is an AVD.
Ollama runs on the workstation only; on-device inference stays with the
LiteRT-LM/ONNX Runtime profiles.

See [docs/on_policy_distillation.md](docs/on_policy_distillation.md) for profile
and task schemas, metrics and troubleshooting.

## Collect research trajectories

An Android `AccessibilityService` may emit timestamped JSON research events to `DRLQN_TRACE`:

```bash
python scripts/collect_android.py --output data/session-001 --serial DEVICE_SERIAL
```

Training data must be joined by timestamps, not file order. Production mode should minimize and redact UI text and must not use the collector as an autonomous action executor.
