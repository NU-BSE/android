# Android DRLQN chat advisor

Android-only Differential Residual Lambda Q-Network adaptation for
**human-operated navigation guidance**.

DRLQN predicts semantic subgoals, while a small local chat model such as
Gemma 3 1B IT or Qwen2.5-1.5B-Instruct explains one next action to the user.
The repository does not autonomously execute Android gestures.

## Safety boundary

Google Play policy prohibits AccessibilityService usage that lets an app
autonomously initiate, plan and execute actions or decisions. The production
path therefore uses:

1. `AdvisoryDRLQN` to expose planner, node ranking and verifier outputs only;
2. `PlannerAdvisor` to render one semantic subgoal as a chat instruction;
3. explicit user confirmation before advancing the plan;
4. deterministic fallback when a local LLM emits malformed, sensitive or
   executable content.

Coordinates, bounding boxes, gesture trajectories, ADB commands and
Accessibility API actions are filtered from the advisor API.

## Included

- timestamp-aligned screenshot, accessibility, OCR/caption and UI-event state;
- compact multimodal DRLQN with semantic subgoal planning;
- observed-window BFS planner;
- policy-safe chat advisor with sensitive-text redaction;
- safe model facade that removes gesture/coordinate outputs;
- device profiles for Android 10-17;
- Stock Android, Samsung, OPPO, realme and vivo deployment cases;
- Gemma 3 1B IT and Qwen2.5-1.5B-Instruct profiles;
- LiteRT-LM CPU/GPU/NPU and ONNX Runtime CPU/NNAPI/QNN fallback plans;
- device benchmark gates for memory, latency, throughput and thermals;
- ADB/logcat data collector for research trajectories.

## Install and test

```bash
python -m pip install -e '.[test]'
pytest
```

## One-step advisor

```python
from android_drlqn import PlannerAdvisor, PlannerSubgoal, SubgoalType

plan = (
    PlannerSubgoal(SubgoalType.FIND_SETTING, "Bluetooth"),
    PlannerSubgoal(SubgoalType.CONFIRM),
)
reply = PlannerAdvisor().advise(plan, current_state, "Включить Bluetooth")
print(reply.step.instruction)

# Advance only after the user confirms the expected result.
plan = PlannerAdvisor().confirm_step(
    plan,
    user_confirmed=True,
    observed_expected_result=True,
)
```

A LiteRT-LM or ONNX Runtime wrapper can implement the `LocalChatModel`
protocol. When no local LLM is available, the same API generates deterministic
instructions from the semantic subgoal and visible UI labels.

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

Manufacturer and SoC strings never enable an accelerator by themselves. The
runtime must be positively detected and pass a short on-device benchmark.

See [docs/on_device_advisor.md](docs/on_device_advisor.md) for the complete
Android 10-17 matrix, LiteRT/ONNX conversion plans, processor-specific cases,
quantization guidance and training-data format.

## Collect research trajectories

An Android `AccessibilityService` may emit timestamped JSON research events to
the `DRLQN_TRACE` logcat tag:

```bash
python scripts/collect_android.py --output data/session-001 --serial DEVICE_SERIAL
```

Training data must be joined by timestamps, not file order. Production chat
mode should minimize and redact UI text and must not use the collector as an
autonomous action executor.
