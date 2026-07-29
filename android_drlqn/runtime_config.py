"""Generate reproducible conversion/build manifests for on-device chat runtimes.

The module does not download models or execute toolchains. It produces an
explicit plan that an Android build system can review, pin and run in CI.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from .device_profiles import (
    DeviceCapabilities,
    InferenceProfile,
    Quantization,
    RuntimeBackend,
)


@dataclass(frozen=True)
class BuildManifest:
    runtime: str
    model_id: str
    artifact_format: str
    commands: tuple[str, ...]
    android_dependencies: tuple[str, ...]
    proguard_rules: tuple[str, ...] = ()
    caveats: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return asdict(self)


def litert_export_manifest(
    profile: InferenceProfile,
    device: DeviceCapabilities,
    *,
    output_dir: str = "build/models/advisor-litertlm",
    int4_recipe_path: str = "config/quantization/int4-weight-only.json",
) -> BuildManifest:
    """Create a LiteRT Torch HF-export and LiteRT-LM deployment plan."""
    command = [
        "litert-torch export_hf",
        f"--model={profile.model.value}",
        f"--output_dir={output_dir}",
    ]
    caveats: list[str] = []

    if profile.quantization is Quantization.DYNAMIC_INT8_FP32:
        command.append("--quantization_recipe=dynamic_wi8_afp32")
    elif profile.quantization is Quantization.WEIGHT_ONLY_INT4:
        command.append(f"--quantization_recipe={int4_recipe_path}")
        caveats.append("INT4 is a custom validated recipe; LiteRT HF export defaults to dynamic INT8/FP32")
    elif profile.quantization is Quantization.FP16:
        command.append("--quantization_recipe=fp16")
    elif profile.quantization in (Quantization.FULL_FLOAT, Quantization.SRQ):
        command.append("--quantization_recipe=''" if profile.quantization is Quantization.FULL_FLOAT else "--quantization_recipe=srq")

    command.extend([
        "--split_cache",
        "--externalize_embedder",
        f"--prefill_lengths={profile.prefill_chunk_tokens}",
        f"--cache_length={profile.context_tokens}",
    ])

    preferred = profile.backend_priority[0]
    if preferred is RuntimeBackend.LITERT_LM_NPU_QUALCOMM:
        if profile.quantization is Quantization.WEIGHT_ONLY_INT4:
            raise ValueError("Qualcomm LiteRT NPU artifacts cannot use weight-only quantization")
        command.extend(["--aot_backend=qualcomm", f"--aot_soc_model={device.soc_model}"])
        caveats.append("Requires a supported Qualcomm AI Runtime SDK and exact SoC target")
    elif preferred is RuntimeBackend.LITERT_LM_NPU_MEDIATEK:
        command.extend(["--aot_backend=mediatek", f"--aot_soc_model={device.soc_model}"])
        caveats.append("Requires the matching MediaTek NeuroPilot toolchain and exact SoC target")
    elif preferred is RuntimeBackend.LITERT_LM_NPU_GOOGLE:
        command.extend(["--aot_backend=GOOGLE", f"--aot_soc_model={device.soc_model}"])
        caveats.append("Requires the matching Google Tensor SDK")
    elif preferred is RuntimeBackend.LITERT_LM_NPU_SAMSUNG:
        caveats.append(
            "Samsung Exynos LiteCore AOT/JIT setup is vendor-specific; use the Samsung LiteRT guide "
            "for Exynos 2500/2600 instead of assuming generic export flags"
        )

    return BuildManifest(
        runtime="LiteRT-LM",
        model_id=profile.model.value,
        artifact_format=".litertlm",
        commands=(" \\\n  ".join(command),),
        android_dependencies=(
            "com.google.ai.edge.litert:litert:2.1.6",
            "LiteRT-LM runtime matching the exported artifact",
        ),
        caveats=tuple(caveats),
    )


def onnxruntime_android_manifest(
    profile: InferenceProfile,
    device: DeviceCapabilities,
    *,
    android_sdk_path: str = "$ANDROID_SDK_ROOT",
    android_ndk_path: str = "$ANDROID_NDK_ROOT",
    source_dir: str = "third_party/onnxruntime",
    qnn_home: str = "$QNN_SDK_ROOT",
) -> BuildManifest:
    """Create an ONNX Runtime Android AAR build plan with ordered EP fallback."""
    api = max(29, device.api_level)
    build = [
        f"cd {source_dir}",
        "./build.sh",
        "--android",
        f"--android_sdk_path {android_sdk_path}",
        f"--android_ndk_path {android_ndk_path}",
        "--android_abi arm64-v8a",
        f"--android_api {api}",
        "--build_java",
    ]
    caveats: list[str] = []

    if RuntimeBackend.ONNXRUNTIME_NNAPI in profile.backend_priority:
        build.append("--use_nnapi")
        caveats.append("NNAPI EP support is operator- and driver-dependent; retain CPU fallback")
    if RuntimeBackend.ONNXRUNTIME_QNN in profile.backend_priority:
        build.extend(["--use_qnn static_lib", f"--qnn_home {qnn_home}"])
        caveats.append("QNN Android builds require the matching Qualcomm AI Engine Direct SDK")

    return BuildManifest(
        runtime="ONNX Runtime Android",
        model_id=profile.model.value,
        artifact_format=".onnx + .aar",
        commands=(" \\\n  ".join(build),),
        android_dependencies=(
            "Generated onnxruntime-android AAR",
            "Tokenizer/native text generation layer compatible with the exported model",
        ),
        proguard_rules=("-keep class ai.onnxruntime.** { *; }",),
        caveats=tuple(caveats),
    )


def write_manifest(manifest: BuildManifest, path: str | Path) -> None:
    import json

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
