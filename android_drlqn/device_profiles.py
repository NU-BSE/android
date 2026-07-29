"""Device-aware local LLM profiles for Android 10 through Android 17.

The selector is deliberately conservative. Vendor/OS names are hints only;
accelerators are enabled when the application has positively detected the
runtime/delegate and a short on-device benchmark passes memory, latency and
thermal gates.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Iterable, Mapping, Sequence


ANDROID_API_BY_VERSION = {
    10: 29,
    11: 30,
    12: 31,
    13: 33,
    14: 34,
    15: 35,
    16: 36,
    17: 37,
}


class DeviceFamily(str, Enum):
    STOCK_ANDROID = "stock_android"
    SAMSUNG = "samsung"
    OPPO = "oppo"
    REALME = "realme"
    VIVO = "vivo"
    OTHER = "other"


class DeviceTier(str, Enum):
    LOW = "low"
    MID = "mid"
    HIGH = "high"


class SocVendor(str, Enum):
    QUALCOMM = "qualcomm"
    SAMSUNG = "samsung_exynos"
    MEDIATEK = "mediatek"
    GOOGLE = "google_tensor"
    UNISOC = "unisoc"
    UNKNOWN = "unknown"


class RuntimeBackend(str, Enum):
    LITERT_LM_CPU = "litert_lm_cpu"
    LITERT_LM_GPU = "litert_lm_gpu"
    LITERT_LM_NPU_QUALCOMM = "litert_lm_npu_qualcomm"
    LITERT_LM_NPU_SAMSUNG = "litert_lm_npu_samsung"
    LITERT_LM_NPU_MEDIATEK = "litert_lm_npu_mediatek"
    LITERT_LM_NPU_GOOGLE = "litert_lm_npu_google"
    ONNXRUNTIME_CPU = "onnxruntime_cpu"
    ONNXRUNTIME_NNAPI = "onnxruntime_nnapi"
    ONNXRUNTIME_QNN = "onnxruntime_qnn"


class ModelFamily(str, Enum):
    GEMMA_3_1B_IT = "google/gemma-3-1b-it"
    QWEN_2_5_1_5B_INSTRUCT = "Qwen/Qwen2.5-1.5B-Instruct"


class Quantization(str, Enum):
    DYNAMIC_INT8_FP32 = "dynamic_wi8_afp32"
    WEIGHT_ONLY_INT4 = "weight_only_int4"
    FP16 = "fp16"
    FULL_FLOAT = "full_float"
    SRQ = "srq"


@dataclass(frozen=True)
class DeviceCapabilities:
    android_version: int
    manufacturer: str = ""
    model: str = ""
    family: DeviceFamily = DeviceFamily.OTHER
    tier: DeviceTier = DeviceTier.MID
    soc_vendor: SocVendor = SocVendor.UNKNOWN
    soc_model: str = ""
    ram_gb: float = 6.0
    cpu_big_cores: int = 2
    cpu_total_cores: int = 8
    is_arm64: bool = True
    has_opencl: bool = False
    has_nnapi: bool = True
    has_qnn_runtime: bool = False
    has_samsung_litecore: bool = False
    has_mediatek_neuropilot: bool = False
    has_google_tensor_runtime: bool = False
    supports_fp16: bool = True
    low_ram_device: bool = False

    def __post_init__(self):
        if self.android_version not in ANDROID_API_BY_VERSION:
            raise ValueError("android_version must be an integer from 10 through 17")
        if self.ram_gb <= 0:
            raise ValueError("ram_gb must be positive")
        if self.cpu_total_cores < 1:
            raise ValueError("cpu_total_cores must be positive")

    @property
    def api_level(self) -> int:
        return ANDROID_API_BY_VERSION[self.android_version]

    @classmethod
    def from_build_info(
        cls,
        *,
        android_version: int,
        manufacturer: str,
        model: str,
        soc_manufacturer: str = "",
        soc_model: str = "",
        ram_gb: float = 6.0,
        cpu_big_cores: int = 2,
        cpu_total_cores: int = 8,
        is_arm64: bool = True,
        features: Iterable[str] = (),
    ) -> "DeviceCapabilities":
        maker = manufacturer.casefold()
        soc_name = f"{soc_manufacturer} {soc_model}".casefold()
        feature_set = {item.casefold() for item in features}

        family = DeviceFamily.OTHER
        for candidate in DeviceFamily:
            if candidate is not DeviceFamily.OTHER and candidate.value.split("_")[0] in maker:
                family = candidate
                break
        if any(name in maker for name in ("google", "pixel")):
            family = DeviceFamily.STOCK_ANDROID

        if any(name in soc_name for name in ("qualcomm", "snapdragon", "sm8", "sm7")):
            soc_vendor = SocVendor.QUALCOMM
        elif any(name in soc_name for name in ("exynos", "samsung")):
            soc_vendor = SocVendor.SAMSUNG
        elif any(name in soc_name for name in ("mediatek", "dimensity", "mt6", "mt7", "mt8")):
            soc_vendor = SocVendor.MEDIATEK
        elif any(name in soc_name for name in ("google", "tensor")):
            soc_vendor = SocVendor.GOOGLE
        elif "unisoc" in soc_name:
            soc_vendor = SocVendor.UNISOC
        else:
            soc_vendor = SocVendor.UNKNOWN

        low_ram = ram_gb < 4.0
        if low_ram or ram_gb < 4.5:
            tier = DeviceTier.LOW
        elif ram_gb >= 8.0 and cpu_big_cores >= 3:
            tier = DeviceTier.HIGH
        else:
            tier = DeviceTier.MID

        return cls(
            android_version=android_version,
            manufacturer=manufacturer,
            model=model,
            family=family,
            tier=tier,
            soc_vendor=soc_vendor,
            soc_model=soc_model,
            ram_gb=ram_gb,
            cpu_big_cores=cpu_big_cores,
            cpu_total_cores=cpu_total_cores,
            is_arm64=is_arm64,
            has_opencl="opencl" in feature_set,
            has_nnapi="nnapi" in feature_set or android_version >= 10,
            has_qnn_runtime="qnn" in feature_set,
            has_samsung_litecore="samsung_litecore" in feature_set,
            has_mediatek_neuropilot="mediatek_neuropilot" in feature_set,
            has_google_tensor_runtime="google_tensor" in feature_set,
            supports_fp16="fp16" in feature_set or android_version >= 10,
            low_ram_device=low_ram,
        )


@dataclass(frozen=True)
class InferenceProfile:
    name: str
    model: ModelFamily
    quantization: Quantization
    backend_priority: tuple[RuntimeBackend, ...]
    context_tokens: int
    max_new_tokens: int
    prefill_chunk_tokens: int
    cpu_threads: int
    max_ui_labels: int
    max_history_turns: int
    memory_budget_mb: int
    mmap_weights: bool = True
    reuse_kv_cache: bool = True
    split_prefill_decode: bool = True
    run_only_when_foreground: bool = True
    stop_on_thermal_severe: bool = True
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        value = asdict(self)
        value["model"] = self.model.value
        value["quantization"] = self.quantization.value
        value["backend_priority"] = [backend.value for backend in self.backend_priority]
        return value


@dataclass(frozen=True)
class BenchmarkSample:
    backend: RuntimeBackend
    first_token_ms: float
    decode_tokens_per_second: float
    peak_rss_mb: float
    thermal_severe: bool = False
    failed: bool = False


@dataclass(frozen=True)
class BackendDecision:
    selected: RuntimeBackend
    accepted: tuple[BenchmarkSample, ...]
    rejected: Mapping[RuntimeBackend, str] = field(default_factory=dict)


class DeviceProfileSelector:
    """Select a safe starting profile, then refine it with device benchmarks."""

    def select(self, device: DeviceCapabilities) -> InferenceProfile:
        if not device.is_arm64:
            raise ValueError("local 1B-class advisor models require an arm64 build")

        model = self._model(device)
        quant = self._quantization(device)
        backends = self._backends(device, quant)
        memory_budget = self._memory_budget_mb(device)

        if device.tier is DeviceTier.LOW or device.low_ram_device:
            context, output, chunk, labels, turns = 384, 48, 32, 20, 2
        elif device.tier is DeviceTier.MID:
            context, output, chunk, labels, turns = 768, 72, 64, 36, 3
        else:
            context, output, chunk, labels, turns = 1280, 96, 128, 48, 4

        notes: list[str] = []
        if device.android_version == 17:
            context = min(context, 1024)
            chunk = min(chunk, 96)
            notes.append("Android 17: conservative KV cache due to platform app-memory limits")

        if device.android_version <= 12:
            context = min(context, 640)
            output = min(output, 64)
            notes.append("Android 10-12: conservative context and CPU-first fallback")

        if device.family in (DeviceFamily.OPPO, DeviceFamily.REALME, DeviceFamily.VIVO):
            notes.append("ColorOS/realme UI/Funtouch OS: benchmark vendor drivers before enabling GPU/NPU")
        if device.family is DeviceFamily.SAMSUNG and device.tier is DeviceTier.MID:
            notes.append("Samsung mid-range: do not assume Exynos LiteCore availability")
        if RuntimeBackend.LITERT_LM_NPU_SAMSUNG in backends:
            notes.append("Samsung LiteCore path is limited to positively detected supported Exynos SoCs")
        if RuntimeBackend.ONNXRUNTIME_QNN in backends:
            notes.append("QNN must use a device-matched SDK/runtime and a compatible quantized graph")

        return InferenceProfile(
            name=f"android{device.android_version}-{device.family.value}-{device.tier.value}",
            model=model,
            quantization=quant,
            backend_priority=backends,
            context_tokens=context,
            max_new_tokens=output,
            prefill_chunk_tokens=chunk,
            cpu_threads=self._threads(device),
            max_ui_labels=labels,
            max_history_turns=turns,
            memory_budget_mb=memory_budget,
            notes=tuple(notes),
        )

    def choose_benchmark_winner(
        self,
        profile: InferenceProfile,
        samples: Sequence[BenchmarkSample],
        *,
        max_first_token_ms: float = 2500.0,
        min_decode_tps: float = 3.0,
    ) -> BackendDecision:
        by_backend = {sample.backend: sample for sample in samples}
        accepted: list[BenchmarkSample] = []
        rejected: dict[RuntimeBackend, str] = {}

        for backend in profile.backend_priority:
            sample = by_backend.get(backend)
            if sample is None:
                rejected[backend] = "not benchmarked"
                continue
            reason = self._rejection_reason(
                profile,
                sample,
                max_first_token_ms=max_first_token_ms,
                min_decode_tps=min_decode_tps,
            )
            if reason:
                rejected[backend] = reason
            else:
                accepted.append(sample)

        if not accepted:
            cpu = next(
                (backend for backend in profile.backend_priority if backend in (
                    RuntimeBackend.LITERT_LM_CPU,
                    RuntimeBackend.ONNXRUNTIME_CPU,
                )),
                profile.backend_priority[-1],
            )
            return BackendDecision(cpu, (), rejected)

        priority = {backend: index for index, backend in enumerate(profile.backend_priority)}
        accepted.sort(
            key=lambda sample: (
                -sample.decode_tokens_per_second,
                sample.first_token_ms,
                priority.get(sample.backend, 999),
            )
        )
        return BackendDecision(accepted[0].backend, tuple(accepted), rejected)

    @staticmethod
    def _rejection_reason(profile, sample, *, max_first_token_ms, min_decode_tps):
        if sample.failed:
            return "runtime failure"
        if sample.thermal_severe and profile.stop_on_thermal_severe:
            return "thermal status severe"
        if sample.peak_rss_mb > profile.memory_budget_mb:
            return "memory budget exceeded"
        if sample.first_token_ms > max_first_token_ms:
            return "first-token latency too high"
        if sample.decode_tokens_per_second < min_decode_tps:
            return "decode throughput too low"
        return ""

    @staticmethod
    def _model(device: DeviceCapabilities) -> ModelFamily:
        if device.tier is DeviceTier.HIGH and device.ram_gb >= 8.0:
            return ModelFamily.QWEN_2_5_1_5B_INSTRUCT
        return ModelFamily.GEMMA_3_1B_IT

    @staticmethod
    def _quantization(device: DeviceCapabilities) -> Quantization:
        if device.tier is DeviceTier.LOW or device.ram_gb < 6.0:
            return Quantization.WEIGHT_ONLY_INT4
        return Quantization.DYNAMIC_INT8_FP32

    @staticmethod
    def _threads(device: DeviceCapabilities) -> int:
        if device.tier is DeviceTier.LOW:
            return min(3, device.cpu_total_cores)
        if device.tier is DeviceTier.MID:
            return min(max(3, device.cpu_big_cores + 1), 5, device.cpu_total_cores)
        return min(max(4, device.cpu_big_cores + 1), 6, device.cpu_total_cores)

    @staticmethod
    def _memory_budget_mb(device: DeviceCapabilities) -> int:
        ratio = 0.20 if device.android_version == 17 else 0.24
        if device.low_ram_device:
            ratio = 0.18
        return max(512, min(3072, int(device.ram_gb * 1024 * ratio)))

    @staticmethod
    def _backends(device: DeviceCapabilities, quant: Quantization) -> tuple[RuntimeBackend, ...]:
        candidates: list[RuntimeBackend] = []

        if (
            device.soc_vendor is SocVendor.SAMSUNG
            and device.has_samsung_litecore
            and device.soc_model.casefold() in {"exynos 2500", "exynos 2600", "e9955", "e9965"}
            and device.api_level >= 36
        ):
            candidates.append(RuntimeBackend.LITERT_LM_NPU_SAMSUNG)

        if device.soc_vendor is SocVendor.GOOGLE and device.has_google_tensor_runtime:
            candidates.append(RuntimeBackend.LITERT_LM_NPU_GOOGLE)

        if device.soc_vendor is SocVendor.MEDIATEK and device.has_mediatek_neuropilot:
            candidates.append(RuntimeBackend.LITERT_LM_NPU_MEDIATEK)

        if device.soc_vendor is SocVendor.QUALCOMM and device.has_qnn_runtime:
            if quant is not Quantization.WEIGHT_ONLY_INT4:
                candidates.append(RuntimeBackend.LITERT_LM_NPU_QUALCOMM)
            candidates.append(RuntimeBackend.ONNXRUNTIME_QNN)

        if device.has_opencl and device.tier is not DeviceTier.LOW:
            candidates.append(RuntimeBackend.LITERT_LM_GPU)

        candidates.append(RuntimeBackend.LITERT_LM_CPU)
        if device.has_nnapi:
            candidates.append(RuntimeBackend.ONNXRUNTIME_NNAPI)
        candidates.append(RuntimeBackend.ONNXRUNTIME_CPU)

        output: list[RuntimeBackend] = []
        for candidate in candidates:
            if candidate not in output:
                output.append(candidate)
        return tuple(output)
