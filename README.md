# Android DRLQN VLM + chat coordinator

Android-only Differential Residual Lambda Q-Network adaptation for evidence-backed Android settings guidance.

The production path now connects:

- the existing multimodal DRLQN planner/perception model;
- an optional local VLM for screenshot grounding;
- a local chat LLM for one-step explanations;
- a normalized ORM catalog of settings and official evidence;
- explicit confirmation and verified action receipts.

The repository still does not ship an autonomous `AccessibilityService` executor. A host application may keep the flow manual or attach a separately reviewed executor after policy approval.

## Safety and evidence boundary

1. `SettingsCatalog` resolves a setting only for the exact OS/release/API/device scope.
2. Tier 0-2 immutable evidence is required for a workflow proposal.
3. `PerceptionDRLQN` exposes node/box/heatmap grounding but filters all gesture trajectories and timing outputs.
4. `HybridGrounder` prefers accessibility nodes, then DRLQN ranking, then a strict local-VLM fallback.
5. `PlannerAdvisor` explains exactly one semantic action to the user.
6. Medium/high/critical-risk or user-interaction changes require explicit confirmation.
7. The workflow advances only after an `ActionReceipt` passes `verify_transition()`.

Documentation and source text are treated as untrusted evidence, not executable instructions. Shell, Settings Provider, DevicePolicyManager, Android Management API, Shizuku and root mechanisms require separate approved executors.

## Included

- timestamp-aligned screenshot, accessibility, OCR/caption and UI-event state;
- compact multimodal DRLQN with semantic planning, grounding and verification heads;
- `AdvisoryDRLQN` chat-only facade;
- `PerceptionDRLQN` internal visual-grounding facade with gesture filtering;
- strict `LocalVisionLanguageModel` and `MultimodalGenerator` interfaces;
- `JsonGroundingVLM` prompt/JSON adapter;
- accessibility-first `HybridGrounder` with DRLQN and VLM fallback;
- normalized ORM claim loader and exact-scope settings catalog;
- evidence-backed workflow planner and settings coordinator;
- local chat advisor with sensitive-text redaction;
- device profiles for Android 10-17 and LiteRT-LM/ONNX Runtime deployment manifests;
- benchmark gates for memory, latency, throughput and thermals;
- ADB/logcat data collector for research trajectories.

## Install and test

```bash
python -m pip install -e '.[test]'
pytest
```

## VLM + chat settings flow

```python
from android_drlqn import (
    InMemorySettingsCatalog,
    JsonGroundingVLM,
    SettingsCoordinator,
    TargetScope,
)

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

The Android host performs or presents the proposed action and returns an `ActionReceipt` with timestamp-aligned before/after states. The coordinator never trusts a model's statement that an action succeeded.

See [docs/vlm_chat_coordinator.md](docs/vlm_chat_coordinator.md) for the contracts and [examples/settings_coordinator.py](examples/settings_coordinator.py) for a complete normalized-claim example.

## Existing one-step advisor

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

Manufacturer and SoC strings never enable an accelerator by themselves. The runtime must be positively detected and pass a short on-device benchmark.

See [docs/on_device_advisor.md](docs/on_device_advisor.md) for the Android 10-17 runtime matrix.

## Collect research trajectories

An Android `AccessibilityService` may emit timestamped JSON research events to the `DRLQN_TRACE` logcat tag:

```bash
python scripts/collect_android.py --output data/session-001 --serial DEVICE_SERIAL
```

Training data must be joined by timestamps, not file order. Production mode should minimize and redact UI text and must not use the collector as an autonomous action executor.
