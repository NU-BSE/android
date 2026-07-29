# On-device DRLQN chat advisor

## Product boundary

The production role of DRLQN is **advisory planning**, not Android automation.
The planner predicts semantic subgoals, for example:

```text
OPEN_MENU
FIND_SETTING("Bluetooth")
CONFIRM
```

`AdvisoryDRLQN` removes coordinate, heatmap, bounding-box and gesture outputs
from the production API. `PlannerAdvisor` converts only the next subgoal into a
short chat message. The user performs the action and confirms the observed
result before the plan can advance.

This boundary follows the Google Play Accessibility API guidance: an app may
not use AccessibilityService to autonomously initiate, plan and execute actions
or decisions. Narrow deterministic automation and verified accessibility tools
have separate requirements, but they are outside this repository's default
product mode.

The advisor therefore has no `execute`, `tap`, `swipe`, `performAction`,
`dispatchGesture` or ADB integration.

## Chat contract

The local model receives only:

- the user's goal;
- one DRLQN semantic subgoal;
- package/activity when enabled;
- a deduplicated list of visible UI labels;
- no screenshots, coordinates, node IDs, passwords, OTPs or long numbers.

It returns one JSON object:

```json
{
  "instruction": "Найдите пункт «Bluetooth» и нажмите его.",
  "expected_result": "Откроется раздел Bluetooth.",
  "confirmation_question": "Раздел открылся?",
  "target_label": "Bluetooth",
  "caution": null
}
```

Malformed output, executable commands, coordinates or labels absent from the
visible UI cause a deterministic fallback. A step is removed from the plan only
when the user explicitly confirms completion and the expected result is
observed.

## Model choices

The repository supports configuration for:

- `google/gemma-3-1b-it` as the default for low- and mid-range devices;
- `Qwen/Qwen2.5-1.5B-Instruct` for high-range devices with at least 8 GB RAM.

LiteRT Torch HF export currently recognizes both Gemma 3 and Qwen 2/2.5 model
architectures and produces `.litertlm` artifacts for LiteRT-LM. Its documented
default quantization is per-channel INT8 weights with FP32 activations
(`dynamic_wi8_afp32`). A custom INT4 recipe can reduce memory further, but it
must be validated for quality and backend support before shipping.

Do not fine-tune a 1B-class model on the phone. Perform SFT/LoRA or distillation
off-device, then convert and benchmark the frozen artifact locally. On-device
adaptation should be limited to prompt templates, short conversation history,
retrieval, caches and device-specific compiled artifacts.

## Device matrix

Every recommendation is a starting point. Run the benchmark gate on the actual
device and keep CPU fallback.

| Device group | Initial model | Backend order | Context | Notes |
|---|---|---|---:|---|
| Android 10-12, 4-6 GB, Stock Android | Gemma 3 1B IT | LiteRT-LM CPU → ORT NNAPI → ORT CPU | 384-640 | CPU-first because transformer coverage and vendor NNAPI drivers vary |
| Android 13-15, Stock mid-range | Gemma 3 1B IT | LiteRT GPU if OpenCL benchmark passes → LiteRT CPU → ORT | 768 | 3-5 CPU threads; prefill chunks of 64 tokens |
| Pixel/Google Tensor high-range | Qwen 2.5 1.5B or Gemma | Google Tensor NPU only when matching SDK/runtime is detected → GPU → CPU | 1024-1280 | AOT artifact must match the Tensor generation |
| Samsung mid-range | Gemma 3 1B IT | GPU benchmark → LiteRT CPU → ORT NNAPI/CPU | 640-768 | Never infer LiteCore support from Samsung branding alone |
| Samsung high-range, Snapdragon | Qwen 2.5 1.5B | QNN/LiteRT Qualcomm when exact runtime is present → GPU → CPU | 1024-1280 | QNN package and graph must match the target SoC |
| Samsung high-range, Exynos 2500/2600 | Qwen 2.5 1.5B | Samsung LiteCore → GPU → CPU | 1024-1280 | Current Samsung LiteRT documentation lists Exynos 2500 and 2600 |
| OPPO/realme/vivo mid-range, MediaTek | Gemma 3 1B IT | GPU benchmark → CPU → NNAPI | 640-768 | Enable NeuroPilot only after positive runtime detection and AOT validation |
| OPPO/realme/vivo high-range, Snapdragon | Qwen 2.5 1.5B | QNN/LiteRT Qualcomm → GPU → CPU | 1024-1280 | Thermal throttling is common in sustained decode; use short responses |
| Android 16 | Tier-specific profile | NPU/GPU only after benchmark → CPU | up to 1280 | LiteRT CompiledModel/LiteRT-LM is preferred for new integrations |
| Android 17 / API 37 | Same as hardware tier | Same backend gate | capped at 1024 initially | Start conservatively because Android 17 introduces app-memory limits |

## Runtime selection

`DeviceProfileSelector` uses explicit capabilities rather than manufacturer
names alone:

```python
from android_drlqn.device_profiles import DeviceCapabilities, DeviceProfileSelector

capabilities = DeviceCapabilities.from_build_info(
    android_version=15,
    manufacturer="OPPO",
    model="Example",
    soc_manufacturer="Qualcomm",
    soc_model="SM8750",
    ram_gb=12,
    cpu_big_cores=4,
    features={"opencl", "nnapi", "qnn", "fp16"},
)
profile = DeviceProfileSelector().select(capabilities)
```

On Android, populate these fields from `Build`, `ActivityManager`, ABI checks,
feature probing and successful runtime initialization. Never enable NPU solely
from a model-name string.

Benchmark each available candidate with the same prompt and enforce:

- first-token latency threshold;
- minimum decode tokens/second;
- peak RSS below the profile memory budget;
- no severe thermal status;
- identical or acceptable output quality on a small golden set.

The selector rejects failing candidates and chooses the fastest accepted
backend. If none pass, it falls back to CPU.

## LiteRT-LM conversion

`litert_export_manifest()` creates a reviewable command plan rather than
executing downloads or conversion automatically.

```python
from android_drlqn.runtime_config import litert_export_manifest

manifest = litert_export_manifest(profile, capabilities)
print(manifest.commands[0])
```

The Google AI Edge HF export tool:

- supports Gemma 3 and Qwen 2/2.5 architectures;
- applies graph optimizations and quantization;
- exports `.litertlm` for CPU/GPU and supported NPU paths;
- supports AOT targets for Google Tensor, Qualcomm and MediaTek;
- requires vendor-specific setup for NPU compilation.

Important Qualcomm constraint: the LiteRT Qualcomm NPU path does not support
weight-only quantized models; use a supported full-float or SRQ artifact for
that NPU path. INT4 profiles therefore keep QNN/CPU/GPU fallbacks until a
compatible target artifact has been produced.

Samsung LiteCore is not a generic Samsung-device switch. The published guide
currently names Exynos 2500 (E9955) and Exynos 2600 (E9965), Android API 36 for
the development target, and NDK support down to API 28.

## ONNX Runtime fallback

`onnxruntime_android_manifest()` emits an arm64 Android AAR build command with
optional NNAPI and QNN execution providers. The generated plan includes the R8
rule required by the ONNX Runtime Android documentation:

```text
-keep class ai.onnxruntime.** { *; }
```

NNAPI works on Android 8.1+, but operator coverage and vendor drivers vary.
For transformer chat models it should be benchmarked against CPU rather than
assumed faster. QNN works on supported Snapdragon devices and requires a
matching Qualcomm AI Engine Direct SDK for Android source builds.

## Memory and latency controls

The default profiles apply the following optimizations:

- 4-bit weight-only candidate for low-memory CPU/GPU deployments, otherwise
  documented LiteRT dynamic INT8/FP32;
- memory-mapped weights;
- reusable KV cache;
- split prefill/decode;
- short output cap of 48-96 tokens;
- 384-1280 token context depending on OS/RAM/tier;
- 20-48 visible UI labels, not the full accessibility tree;
- two to four chat turns;
- foreground-only generation;
- immediate fallback or stop on severe thermal status.

The advisor task needs concise imperative language, not broad reasoning. Smaller
context and output limits usually improve latency without harming navigation
quality.

## Recommended offline adaptation dataset

For visual grounding, prefer the on-policy loop in
[on_policy_distillation.md](on_policy_distillation.md): it supervises states the
student actually reaches. The offline set below stays the right shape for the
*chat* advisor, which has no emulator rollout.

Train or distill the local LLM on one-step records:

```json
{
  "goal": "Включить Bluetooth",
  "subgoal": "FIND_SETTING",
  "argument": "Bluetooth",
  "visible_labels": ["Сеть и интернет", "Bluetooth", "Подключенные устройства"],
  "assistant": {
    "instruction": "Найдите пункт «Bluetooth» и нажмите его.",
    "expected_result": "Откроется раздел Bluetooth.",
    "confirmation_question": "Раздел открылся?"
  }
}
```

Include hard negatives:

- duplicate labels;
- disabled controls;
- vendor-renamed settings;
- search results versus actual settings pages;
- permission and confirmation dialogs;
- Android 10-17 wording differences;
- Samsung One UI, ColorOS, realme UI and Funtouch OS variants.

Do not train the student to emit coordinates, node IDs, ADB commands, hidden
reasoning or multi-step autonomous scripts.

## Sources

- Google Play AccessibilityService policy:
  https://support.google.com/googleplay/android-developer/answer/10964491
- LiteRT PyTorch GenAI conversion:
  https://developers.google.com/edge/litert/conversion/pytorch/genai
- LiteRT Android versions and CompiledModel API:
  https://developers.google.com/edge/litert/android
- Samsung Exynos LiteCore:
  https://developers.google.com/edge/litert/next/samsung
- ONNX Runtime Android build:
  https://onnxruntime.ai/docs/build/android.html
- ONNX Runtime NNAPI:
  https://onnxruntime.ai/docs/execution-providers/NNAPI-ExecutionProvider.html
- ONNX Runtime QNN:
  https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html
- Android 17 behavior changes:
  https://developer.android.com/about/versions/17/behavior-changes-all
