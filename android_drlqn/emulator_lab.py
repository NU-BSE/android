"""Reproducible emulator matrix for Android on-policy distillation.

Profiles describe build/run inputs but never download or execute arbitrary URLs.
Only pre-approved SDK packages, source build targets, and image artifacts may be
materialized by a separate lab runner.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable, Mapping


ANDROID_API_LEVELS_BY_VERSION = {
    10: (29,),
    11: (30,),
    12: (31, 32),
    13: (33,),
    14: (34,),
    15: (35,),
    16: (36,),
    17: (37,),
}


class EmulatorKind(str, Enum):
    ANDROID_AVD = "android_avd"
    SOURCE_BUILD = "source_build"
    PREBUILT_IMAGE = "prebuilt_image"


class FormFactor(str, Enum):
    PHONE = "phone"
    TABLET = "tablet"
    FOLDABLE = "foldable"
    RESIZABLE = "resizable"


class EmulatorDeviceTier(str, Enum):
    LOW = "low"
    MID = "mid"
    HIGH = "high"


@dataclass(frozen=True)
class DisplaySpec:
    width_px: int
    height_px: int
    density_dpi: int
    font_scale: float = 1.0

    def __post_init__(self) -> None:
        if min(self.width_px, self.height_px, self.density_dpi) <= 0:
            raise ValueError("display dimensions and density must be positive")
        if not 0.5 <= self.font_scale <= 2.5:
            raise ValueError("font_scale must be in [0.5, 2.5]")


@dataclass(frozen=True)
class EmulatorProfile:
    profile_id: str
    os_family: str
    os_release: str
    android_version: int
    api_level: int
    kind: EmulatorKind
    device_class: str
    tier: EmulatorDeviceTier
    form_factor: FormFactor
    architecture: str
    ram_mb: int
    displays: tuple[DisplaySpec, ...]
    locales: tuple[str, ...]
    navigation_modes: tuple[str, ...]
    system_image: str | None = None
    avd_device: str | None = None
    build_target: str | None = None
    source_branch: str | None = None
    artifact_reference: str | None = None
    enabled_by_default: bool = True
    requires_kvm: bool = True
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.android_version not in ANDROID_API_LEVELS_BY_VERSION:
            raise ValueError("android_version must be in the supported 10-17 range")
        if self.api_level not in ANDROID_API_LEVELS_BY_VERSION[self.android_version]:
            raise ValueError(
                f"api_level {self.api_level} does not match Android {self.android_version}"
            )
        if self.ram_mb < 2048:
            raise ValueError("emulator profile RAM must be at least 2048 MB")
        if not self.displays or not self.locales or not self.navigation_modes:
            raise ValueError("display, locale and navigation coverage must be explicit")
        if self.kind is EmulatorKind.ANDROID_AVD and not self.system_image:
            raise ValueError("AVD profile requires system_image")
        if self.kind is EmulatorKind.SOURCE_BUILD and not self.build_target:
            raise ValueError("source-build profile requires build_target")
        if self.kind is EmulatorKind.PREBUILT_IMAGE and not self.artifact_reference:
            raise ValueError("prebuilt-image profile requires artifact_reference")

    @property
    def variant_count(self) -> int:
        return len(self.displays) * len(self.locales) * len(self.navigation_modes)

    def expanded_variants(self) -> tuple[dict[str, object], ...]:
        output = []
        for display in self.displays:
            for locale in self.locales:
                for navigation in self.navigation_modes:
                    output.append(
                        {
                            "profile_id": self.profile_id,
                            "os_family": self.os_family,
                            "os_release": self.os_release,
                            "android_version": self.android_version,
                            "api_level": self.api_level,
                            "device_class": self.device_class,
                            "tier": self.tier.value,
                            "form_factor": self.form_factor.value,
                            "architecture": self.architecture,
                            "ram_mb": self.ram_mb,
                            "display": asdict(display),
                            "locale": locale,
                            "navigation_mode": navigation,
                        }
                    )
        return tuple(output)


@dataclass(frozen=True)
class EmulatorCoverageReport:
    profile_count: int
    expanded_variant_count: int
    android_versions: tuple[int, ...]
    api_levels: tuple[int, ...]
    os_families: tuple[str, ...]
    tiers: tuple[str, ...]
    form_factors: tuple[str, ...]
    locales: tuple[str, ...]
    navigation_modes: tuple[str, ...]
    missing_axes: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return not self.missing_axes


class EmulatorMatrix:
    def __init__(self, profiles: Iterable[EmulatorProfile]):
        self.profiles = tuple(profiles)
        ids = [profile.profile_id for profile in self.profiles]
        if len(ids) != len(set(ids)):
            raise ValueError("emulator profile ids must be unique")

    def enabled(self) -> tuple[EmulatorProfile, ...]:
        return tuple(profile for profile in self.profiles if profile.enabled_by_default)

    def report(self) -> EmulatorCoverageReport:
        versions = {profile.android_version for profile in self.profiles}
        api_levels = {profile.api_level for profile in self.profiles}
        os_families = {profile.os_family for profile in self.profiles}
        tiers = {profile.tier.value for profile in self.profiles}
        form_factors = {profile.form_factor.value for profile in self.profiles}
        locales = {locale for profile in self.profiles for locale in profile.locales}
        navigation = {mode for profile in self.profiles for mode in profile.navigation_modes}
        missing = []
        for version in (10, 12, 14, 15, 16, 17):
            if version not in versions:
                missing.append(f"android_version:{version}")
        for tier in EmulatorDeviceTier:
            if tier.value not in tiers:
                missing.append(f"tier:{tier.value}")
        for factor in (FormFactor.PHONE, FormFactor.TABLET, FormFactor.FOLDABLE):
            if factor.value not in form_factors:
                missing.append(f"form_factor:{factor.value}")
        for locale in ("en-US", "ru-RU", "ar-SA"):
            if locale not in locales:
                missing.append(f"locale:{locale}")
        for mode in ("gesture", "three_button"):
            if mode not in navigation:
                missing.append(f"navigation:{mode}")
        for family in ("android-aosp", "grapheneos", "calyxos", "eos"):
            if family not in os_families:
                missing.append(f"os_family:{family}")
        return EmulatorCoverageReport(
            profile_count=len(self.profiles),
            expanded_variant_count=sum(profile.variant_count for profile in self.profiles),
            android_versions=tuple(sorted(versions)),
            api_levels=tuple(sorted(api_levels)),
            os_families=tuple(sorted(os_families)),
            tiers=tuple(sorted(tiers)),
            form_factors=tuple(sorted(form_factors)),
            locales=tuple(sorted(locales)),
            navigation_modes=tuple(sorted(navigation)),
            missing_axes=tuple(missing),
        )

    def require_complete(self) -> EmulatorCoverageReport:
        report = self.report()
        if not report.complete:
            raise RuntimeError(f"emulator matrix misses required axes: {report.missing_axes}")
        return report

    def select_for_os(self, os_family: str) -> tuple[EmulatorProfile, ...]:
        needle = os_family.casefold()
        return tuple(profile for profile in self.profiles if profile.os_family.casefold() == needle)

    def shard(self, shard_index: int, shard_count: int) -> tuple[EmulatorProfile, ...]:
        if shard_count < 1 or not 0 <= shard_index < shard_count:
            raise ValueError("invalid shard index/count")
        return tuple(
            profile for index, profile in enumerate(self.profiles)
            if index % shard_count == shard_index
        )

    def to_dict(self) -> dict[str, object]:
        output = []
        for profile in self.profiles:
            value = asdict(profile)
            value["kind"] = profile.kind.value
            value["tier"] = profile.tier.value
            value["form_factor"] = profile.form_factor.value
            output.append(value)
        return {"schema_version": 1, "profiles": output}

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def selected_emulator_matrix() -> EmulatorMatrix:
    """Return the repository's default train/evaluation matrix.

    AOSP profiles are directly materializable with Android SDK packages. Custom
    OS profiles intentionally require pinned source builds or an approved image
    artifact; they are not silently replaced by AOSP when unavailable.
    """

    common_locales = ("en-US", "ru-RU", "ar-SA")
    common_navigation = ("gesture", "three_button")
    return EmulatorMatrix(
        (
            EmulatorProfile(
                profile_id="aosp10-low-phone",
                os_family="android-aosp",
                os_release="10",
                android_version=10,
                api_level=29,
                kind=EmulatorKind.ANDROID_AVD,
                device_class="720p-low-phone",
                tier=EmulatorDeviceTier.LOW,
                form_factor=FormFactor.PHONE,
                architecture="x86_64",
                ram_mb=3072,
                displays=(DisplaySpec(720, 1280, 280),),
                locales=common_locales,
                navigation_modes=common_navigation,
                system_image="system-images;android-29;google_apis;x86_64",
                avd_device="pixel_2",
                notes=("Legacy modern baseline", "CPU-first inference gate"),
            ),
            EmulatorProfile(
                profile_id="aosp12l-mid-resizable",
                os_family="android-aosp",
                os_release="12L",
                android_version=12,
                api_level=32,
                kind=EmulatorKind.ANDROID_AVD,
                device_class="1080p-mid-resizable",
                tier=EmulatorDeviceTier.MID,
                form_factor=FormFactor.RESIZABLE,
                architecture="x86_64",
                ram_mb=4096,
                displays=(
                    DisplaySpec(1080, 2400, 420),
                    DisplaySpec(1600, 2560, 320, font_scale=1.3),
                ),
                locales=common_locales,
                navigation_modes=common_navigation,
                system_image="system-images;android-32;google_apis;x86_64",
                avd_device="resizable",
                notes=("Material You baseline", "Phone/tablet resize transitions"),
            ),
            EmulatorProfile(
                profile_id="aosp14-main-phone",
                os_family="android-aosp",
                os_release="14",
                android_version=14,
                api_level=34,
                kind=EmulatorKind.ANDROID_AVD,
                device_class="1080p-mainstream-phone",
                tier=EmulatorDeviceTier.MID,
                form_factor=FormFactor.PHONE,
                architecture="x86_64",
                ram_mb=6144,
                displays=(DisplaySpec(1080, 2400, 420),),
                locales=common_locales,
                navigation_modes=common_navigation,
                system_image="system-images;android-34;google_apis;x86_64",
                avd_device="pixel_6",
            ),
            EmulatorProfile(
                profile_id="aosp15-high-phone",
                os_family="android-aosp",
                os_release="15",
                android_version=15,
                api_level=35,
                kind=EmulatorKind.ANDROID_AVD,
                device_class="1080p-high-phone",
                tier=EmulatorDeviceTier.HIGH,
                form_factor=FormFactor.PHONE,
                architecture="x86_64",
                ram_mb=8192,
                displays=(DisplaySpec(1080, 2424, 420),),
                locales=common_locales,
                navigation_modes=common_navigation,
                system_image="system-images;android-35;google_apis;x86_64",
                avd_device="pixel_8",
            ),
            EmulatorProfile(
                profile_id="aosp16-tablet",
                os_family="android-aosp",
                os_release="16",
                android_version=16,
                api_level=36,
                kind=EmulatorKind.ANDROID_AVD,
                device_class="large-screen-tablet",
                tier=EmulatorDeviceTier.HIGH,
                form_factor=FormFactor.TABLET,
                architecture="x86_64",
                ram_mb=8192,
                displays=(DisplaySpec(1600, 2560, 320),),
                locales=common_locales,
                navigation_modes=common_navigation,
                system_image="system-images;android-36;google_apis;x86_64",
                avd_device="pixel_tablet",
                notes=("sw600dp", "Landscape and split-screen curriculum"),
            ),
            EmulatorProfile(
                profile_id="aosp17-foldable",
                os_family="android-aosp",
                os_release="17",
                android_version=17,
                api_level=37,
                kind=EmulatorKind.ANDROID_AVD,
                device_class="foldable-current",
                tier=EmulatorDeviceTier.HIGH,
                form_factor=FormFactor.FOLDABLE,
                architecture="x86_64",
                ram_mb=8192,
                displays=(DisplaySpec(1080, 2092, 420), DisplaySpec(1840, 2208, 420)),
                locales=common_locales,
                navigation_modes=common_navigation,
                system_image="system-images;android-37;google_apis;x86_64",
                avd_device="pixel_fold",
                notes=("Fold/unfold state transition", "Forward-compatibility profile"),
            ),
            EmulatorProfile(
                profile_id="grapheneos17-sdk-phone64",
                os_family="grapheneos",
                os_release="17-pinned",
                android_version=17,
                api_level=37,
                kind=EmulatorKind.SOURCE_BUILD,
                device_class="grapheneos-sdk-emulator",
                tier=EmulatorDeviceTier.HIGH,
                form_factor=FormFactor.PHONE,
                architecture="x86_64",
                ram_mb=8192,
                displays=(DisplaySpec(1080, 2400, 420),),
                locales=common_locales,
                navigation_modes=common_navigation,
                build_target="sdk_phone64_x86_64-userdebug",
                source_branch="17",
                enabled_by_default=False,
                notes=(
                    "Build from pinned official source revision",
                    "Development-only emulator; hardware security features are not represented",
                ),
            ),
            EmulatorProfile(
                profile_id="calyxos-emulator-current",
                os_family="calyxos",
                os_release="pinned-current",
                android_version=16,
                api_level=36,
                kind=EmulatorKind.SOURCE_BUILD,
                device_class="calyxos-sdk-emulator",
                tier=EmulatorDeviceTier.HIGH,
                form_factor=FormFactor.PHONE,
                architecture="x86_64",
                ram_mb=8192,
                displays=(DisplaySpec(1080, 2400, 420),),
                locales=common_locales,
                navigation_modes=common_navigation,
                build_target="sdk_phone64_x86_64-userdebug",
                source_branch="pinned-manifest",
                enabled_by_default=False,
                notes=("Requires the official CalyxOS emulator build instructions and pinned manifest",),
            ),
            EmulatorProfile(
                profile_id="eos13-nexus5x-emulator",
                os_family="eos",
                os_release="android-13-emulator",
                android_version=13,
                api_level=33,
                kind=EmulatorKind.PREBUILT_IMAGE,
                device_class="eos-nexus5x-emulator",
                tier=EmulatorDeviceTier.MID,
                form_factor=FormFactor.PHONE,
                architecture="x86_64",
                ram_mb=4096,
                displays=(DisplaySpec(1080, 1920, 420),),
                locales=common_locales,
                navigation_modes=common_navigation,
                artifact_reference="approved://eos/sdk_phone_x86_64/android-33",
                avd_device="Nexus 5X",
                enabled_by_default=False,
                notes=("Artifact hash and page-level CC BY-SA attribution are mandatory",),
            ),
        )
    )


def load_emulator_matrix(path: str | Path) -> EmulatorMatrix:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    values = payload.get("profiles") if isinstance(payload, Mapping) else payload
    if not isinstance(values, list):
        raise ValueError("emulator matrix must be a list or {'profiles': [...]} object")
    profiles = []
    for item in values:
        profiles.append(
            EmulatorProfile(
                profile_id=str(item["profile_id"]),
                os_family=str(item["os_family"]),
                os_release=str(item["os_release"]),
                android_version=int(item["android_version"]),
                api_level=int(item["api_level"]),
                kind=EmulatorKind(item["kind"]),
                device_class=str(item["device_class"]),
                tier=EmulatorDeviceTier(item["tier"]),
                form_factor=FormFactor(item["form_factor"]),
                architecture=str(item["architecture"]),
                ram_mb=int(item["ram_mb"]),
                displays=tuple(DisplaySpec(**display) for display in item["displays"]),
                locales=tuple(str(value) for value in item["locales"]),
                navigation_modes=tuple(str(value) for value in item["navigation_modes"]),
                system_image=item.get("system_image"),
                avd_device=item.get("avd_device"),
                build_target=item.get("build_target"),
                source_branch=item.get("source_branch"),
                artifact_reference=item.get("artifact_reference"),
                enabled_by_default=bool(item.get("enabled_by_default", True)),
                requires_kvm=bool(item.get("requires_kvm", True)),
                notes=tuple(str(value) for value in item.get("notes") or ()),
            )
        )
    return EmulatorMatrix(profiles)
