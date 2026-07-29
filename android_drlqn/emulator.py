"""Android emulator profiles and an adb-backed research environment.

Distillation runs against *emulators*, never against a user's phone. Every
input-injecting call in this module refuses a serial that adb does not report
as an emulator, and the actuator must be constructed with an explicit
``allow_input=True``. The package still ships no `AccessibilityService`
executor for production devices; this is a research harness for AVDs.

Profiles are declarative so a run can iterate Android 10-17, OEM skins and
screen geometries sequentially with one student checkpoint.
"""
from __future__ import annotations

import json
import re
import subprocess
import time
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np

from .core import (
    AccessibilityNode,
    AccessibilitySnapshot,
    AndroidState,
    Bounds,
    GestureAction,
    GestureType,
    ScreenshotFrame,
    TimestampSynchronizer,
    ViewportMetadata,
)
from .device_profiles import ANDROID_API_BY_VERSION, DeviceCapabilities
from .settings_agent import TargetScope
from .vlm import NormalizedBox

_BOUNDS = re.compile(r"\[(-?\d+),(-?\d+)\]\[(-?\d+),(-?\d+)\]")
_FOCUS = re.compile(r"mCurrentFocus=.*?\s([A-Za-z0-9_.]+)/(\S+?)}")
_EMULATOR_SERIAL = re.compile(r"^emulator-\d+$")

CommandRunner = Callable[[Sequence[str], bool], bytes]


class EmulatorError(RuntimeError):
    """Raised when adb or an emulator profile is not usable for a run."""


def _now_ms() -> int:
    return time.time_ns() // 1_000_000


@dataclass(frozen=True)
class EmulatorProfile:
    """One reproducible AVD configuration in a sequential distillation run."""

    name: str
    avd_name: str
    android_version: int
    os_family: str = "aosp"
    os_release: str | None = None
    device_codename: str | None = None
    manufacturer: str = "Google"
    model: str = "sdk_gphone64"
    width_px: int = 1080
    height_px: int = 2400
    density: float = 2.75
    ram_gb: float = 6.0
    cpu_big_cores: int = 2
    cpu_total_cores: int = 8
    features: tuple[str, ...] = ("nnapi",)
    serial: str | None = None
    boot_timeout_s: float = 300.0
    launch_component: str = "com.android.settings/.Settings"
    extra_emulator_args: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.android_version not in ANDROID_API_BY_VERSION:
            raise ValueError("android_version must be an integer from 10 through 17")
        if self.width_px < 1 or self.height_px < 1:
            raise ValueError("emulator viewport must be positive")
        if self.serial is not None and not _EMULATOR_SERIAL.match(self.serial):
            raise ValueError(
                f"serial {self.serial!r} is not an emulator serial (expected emulator-NNNN)"
            )

    @property
    def api_level(self) -> int:
        return ANDROID_API_BY_VERSION[self.android_version]

    def target_scope(self, *, management_mode: str = "unmanaged") -> TargetScope:
        return TargetScope(
            os_family=self.os_family,
            os_release=self.os_release,
            android_version=self.android_version,
            api_level=self.api_level,
            device_codename=self.device_codename,
            management_mode=management_mode,
        )

    def device_capabilities(self) -> DeviceCapabilities:
        return DeviceCapabilities.from_build_info(
            android_version=self.android_version,
            manufacturer=self.manufacturer,
            model=self.model,
            ram_gb=self.ram_gb,
            cpu_big_cores=self.cpu_big_cores,
            cpu_total_cores=self.cpu_total_cores,
            features=self.features,
        )

    def viewport(self) -> ViewportMetadata:
        return ViewportMetadata(self.width_px, self.height_px, density=self.density)

    def to_dict(self) -> dict:
        payload = {
            key: (list(value) if isinstance(value, tuple) else value)
            for key, value in self.__dict__.items()
        }
        payload["api_level"] = self.api_level
        return payload


def load_profiles(path: str | Path) -> tuple[EmulatorProfile, ...]:
    """Load an ordered profile list from JSON; order defines the run order."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = document.get("profiles") if isinstance(document, Mapping) else document
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        raise ValueError("profile file must contain a list under 'profiles'")
    allowed = set(EmulatorProfile.__dataclass_fields__) - {"api_level"}
    profiles: list[EmulatorProfile] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("each profile must be an object")
        unknown = set(entry) - allowed - {"api_level"}
        if unknown:
            raise ValueError(f"unknown profile fields: {sorted(unknown)}")
        fields = {key: value for key, value in entry.items() if key in allowed}
        for key in ("features", "extra_emulator_args", "notes"):
            if key in fields:
                fields[key] = tuple(fields[key])
        profile = EmulatorProfile(**fields)
        # api_level is derived from android_version; a stated one must agree
        # rather than being silently discarded.
        if "api_level" in entry and int(entry["api_level"]) != profile.api_level:
            raise ValueError(
                f"profile {profile.name}: api_level {entry['api_level']} contradicts "
                f"android_version {profile.android_version} (API {profile.api_level})"
            )
        profiles.append(profile)
    if not profiles:
        raise ValueError("profile file does not define any profile")
    names = [profile.name for profile in profiles]
    if len(set(names)) != len(names):
        raise ValueError("profile names must be unique")
    return tuple(profiles)


def _default_runner(command: Sequence[str], binary: bool) -> bytes:
    result = subprocess.run(
        list(command), stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
    )
    if result.returncode:
        raise EmulatorError(
            f"{' '.join(command)} failed: {result.stderr.decode('utf-8', 'replace').strip()}"
        )
    return result.stdout


class AdbClient:
    """Small adb wrapper with an injectable runner so tests need no device."""

    def __init__(
        self,
        serial: str | None = None,
        *,
        adb_path: str = "adb",
        runner: CommandRunner | None = None,
    ):
        self.serial = serial
        self.adb_path = adb_path
        self._runner = runner or _default_runner

    def run(self, *args: str, binary: bool = False) -> bytes:
        command = [self.adb_path]
        if self.serial:
            command += ["-s", self.serial]
        command += list(args)
        return self._runner(command, binary)

    def text(self, *args: str) -> str:
        return self.run(*args).decode("utf-8", "replace")

    # ---------------------------------------------------------------- devices
    def devices(self) -> tuple[tuple[str, str], ...]:
        output = []
        for line in self.text("devices").splitlines()[1:]:
            parts = line.split()
            if len(parts) >= 2:
                output.append((parts[0], parts[1]))
        return tuple(output)

    def is_emulator(self) -> bool:
        """True only when adb itself reports the target as an emulator."""
        if not self.serial or not _EMULATOR_SERIAL.match(self.serial):
            return False
        if self.serial not in {name for name, _ in self.devices()}:
            return False
        qemu = self.text("shell", "getprop", "ro.kernel.qemu").strip()
        characteristics = self.text("shell", "getprop", "ro.build.characteristics").strip()
        product = self.text("shell", "getprop", "ro.product.model").strip().casefold()
        return qemu == "1" or "emulator" in characteristics or "sdk" in product

    def wait_for_boot(self, timeout_s: float = 300.0, poll_s: float = 2.0) -> None:
        self.run("wait-for-device")
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.text("shell", "getprop", "sys.boot_completed").strip() == "1":
                return
            time.sleep(poll_s)
        raise EmulatorError(f"{self.serial or 'device'} did not finish booting in {timeout_s}s")

    def build_info(self) -> dict:
        return {
            "android_version": self.text("shell", "getprop", "ro.build.version.release").strip(),
            "api_level": self.text("shell", "getprop", "ro.build.version.sdk").strip(),
            "manufacturer": self.text("shell", "getprop", "ro.product.manufacturer").strip(),
            "model": self.text("shell", "getprop", "ro.product.model").strip(),
        }

    # ------------------------------------------------------------ observation
    def screencap(self) -> np.ndarray:
        return decode_screencap(self.run("exec-out", "screencap", binary=True))

    def ui_dump(self) -> str:
        remote = "/sdcard/drlqn_window.xml"
        self.run("shell", "uiautomator", "dump", "--compressed", remote)
        return self.run("exec-out", "cat", remote).decode("utf-8", "replace")

    def focused_component(self) -> tuple[str, str]:
        match = _FOCUS.search(self.text("shell", "dumpsys", "window", "windows"))
        return (match.group(1), match.group(2)) if match else ("", "")

    def viewport(self) -> ViewportMetadata:
        size = re.findall(
            r"(?:Override|Physical) size: (\d+)x(\d+)", self.text("shell", "wm", "size")
        )
        density = re.findall(
            r"(?:Override|Physical) density: (\d+)", self.text("shell", "wm", "density")
        )
        width, height = (int(size[-1][0]), int(size[-1][1])) if size else (0, 0)
        return ViewportMetadata(width, height, density=int(density[-1]) / 160 if density else 1.0)


def decode_screencap(payload: bytes) -> np.ndarray:
    """Decode the raw `adb exec-out screencap` framebuffer into ``HxWx3`` uint8.

    The header is width/height/format, plus a colorspace word on newer
    platform versions. Raw output avoids a PNG decoder dependency.
    """
    if len(payload) < 12:
        raise EmulatorError("screencap payload is too short")
    width, height, pixel_format = struct_unpack_three(payload)
    expected = width * height * 4
    for header in (12, 16):
        if len(payload) - header == expected:
            break
    else:
        raise EmulatorError(
            f"screencap payload {len(payload)}B does not match {width}x{height} RGBA"
        )
    if pixel_format not in (1, 2, 5):  # RGBA_8888 / RGBX_8888 / BGRA_8888
        raise EmulatorError(f"unsupported screencap pixel format {pixel_format}")
    pixels = np.frombuffer(payload[header:], dtype=np.uint8).reshape(height, width, 4)
    pixels = pixels[..., :3]
    if pixel_format == 5:
        pixels = pixels[..., ::-1]
    return np.ascontiguousarray(pixels)


def struct_unpack_three(payload: bytes) -> tuple[int, int, int]:
    return (
        int.from_bytes(payload[0:4], "little"),
        int.from_bytes(payload[4:8], "little"),
        int.from_bytes(payload[8:12], "little"),
    )


def parse_ui_dump(xml: str) -> tuple[AccessibilityNode, ...]:
    """Convert a `uiautomator dump` document into accessibility nodes.

    uiautomator has no stable node identity, so the id is the deterministic
    child path of the node, which is stable for one screen layout.
    """
    try:
        root = ElementTree.fromstring(xml)
    except ElementTree.ParseError as error:
        raise EmulatorError(f"cannot parse uiautomator dump: {error}") from error

    nodes: list[AccessibilityNode] = []

    def walk(element, path: str, parent_id: str | None) -> None:
        for index, child in enumerate(element):
            if child.tag != "node":
                continue
            node_id = f"{path}.{index}" if path else str(index)
            attributes = child.attrib
            match = _BOUNDS.match(attributes.get("bounds", ""))
            bounds = (
                Bounds(*(float(value) for value in match.groups()))
                if match
                else Bounds(0, 0, 0, 0)
            )
            nodes.append(
                AccessibilityNode(
                    node_id=node_id,
                    parent_id=parent_id,
                    class_name=attributes.get("class", ""),
                    package=attributes.get("package", ""),
                    resource_id=attributes.get("resource-id", ""),
                    text=attributes.get("text", ""),
                    content_description=attributes.get("content-desc", ""),
                    bounds=bounds,
                    enabled=attributes.get("enabled") == "true",
                    clickable=attributes.get("clickable") == "true",
                    scrollable=attributes.get("scrollable") == "true",
                    editable=attributes.get("class", "").endswith("EditText"),
                    checked=attributes.get("checked") == "true",
                    selected=attributes.get("selected") == "true",
                    focused=attributes.get("focused") == "true",
                    visible=bounds.right > bounds.left and bounds.bottom > bounds.top,
                )
            )
            walk(child, node_id, node_id)

    walk(root, "", None)
    return tuple(nodes)


class AdbResearchActuator:
    """Input injection restricted to emulators and to explicit opt-in.

    This is the only place in the package that can change device state. It is
    deliberately limited to tap/back/home/scroll on an AVD, and it verifies the
    emulator claim through adb rather than trusting the caller.
    """

    def __init__(
        self,
        adb: AdbClient,
        viewport: ViewportMetadata,
        *,
        allow_input: bool = False,
        settle_ms: int = 700,
    ):
        if not allow_input:
            raise EmulatorError(
                "input injection requires allow_input=True; the package ships no "
                "production executor"
            )
        if not adb.is_emulator():
            raise EmulatorError(
                f"refusing to inject input into {adb.serial or 'the default device'}: "
                "adb does not report it as an emulator"
            )
        self.adb = adb
        self.viewport = viewport
        self.settle_ms = int(settle_ms)

    def _point(self, box: NormalizedBox) -> tuple[int, int]:
        x_norm, y_norm = box.center
        return (
            int(round(x_norm * self.viewport.width_px)),
            int(round(y_norm * self.viewport.height_px)),
        )

    def tap(self, box: NormalizedBox) -> GestureAction:
        x, y = self._point(box)
        self.adb.run("shell", "input", "tap", str(x), str(y))
        self.settle()
        return GestureAction(GestureType.TAP, _now_ms(), points=((float(x), float(y)),))

    def scroll(self, *, up: bool = False) -> GestureAction:
        width, height = self.viewport.width_px, self.viewport.height_px
        x = width // 2
        start_y, end_y = (int(height * 0.3), int(height * 0.7)) if up else (
            int(height * 0.7),
            int(height * 0.3),
        )
        self.adb.run(
            "shell", "input", "swipe", str(x), str(start_y), str(x), str(end_y), "250"
        )
        self.settle()
        return GestureAction(
            GestureType.SWIPE,
            _now_ms(),
            points=((float(x), float(start_y)), (float(x), float(end_y))),
            duration_ms=250,
        )

    def back(self) -> GestureAction:
        self.adb.run("shell", "input", "keyevent", "KEYCODE_BACK")
        self.settle()
        return GestureAction(GestureType.BACK, _now_ms())

    def home(self) -> GestureAction:
        self.adb.run("shell", "input", "keyevent", "KEYCODE_HOME")
        self.settle()
        return GestureAction(GestureType.HOME, _now_ms())

    def launch(self, component: str) -> GestureAction:
        self.adb.run("shell", "am", "start", "-n", component)
        self.settle()
        return GestureAction(GestureType.WAIT, _now_ms())

    def settle(self) -> None:
        time.sleep(self.settle_ms / 1000.0)


class AdbEmulatorEnvironment:
    """`RolloutEnvironment` over one booted emulator.

    ``capture()`` returns a timestamp-aligned `AndroidState`; the synchronizer
    rejects a frame whose accessibility dump drifted too far from the
    screenshot, which is common on a cold or throttled AVD.
    """

    def __init__(
        self,
        profile: EmulatorProfile,
        adb: AdbClient,
        actuator: AdbResearchActuator,
        *,
        synchronizer: TimestampSynchronizer | None = None,
        max_capture_retries: int = 3,
    ):
        self.profile = profile
        self.adb = adb
        self.actuator = actuator
        self.synchronizer = synchronizer or TimestampSynchronizer(max_tree_skew_ms=1500)
        self.max_capture_retries = max(1, int(max_capture_retries))
        self._history: list[GestureAction] = []

    def reset(self) -> AndroidState:
        self.actuator.home()
        if self.profile.launch_component:
            self.actuator.launch(self.profile.launch_component)
        self._history.clear()
        return self.capture()

    def capture(self) -> AndroidState:
        last_error: Exception | None = None
        for _ in range(self.max_capture_retries):
            try:
                screenshot_ts = _now_ms()
                pixels = self.adb.screencap()
                tree_ts = _now_ms()
                nodes = parse_ui_dump(self.adb.ui_dump())
                package, activity = self.adb.focused_component()
                synchronized = self.synchronizer.synchronize(
                    ScreenshotFrame(screenshot_ts, pixels),
                    (AccessibilitySnapshot(tree_ts, nodes, active_window_id=hash_window(nodes)),),
                    (),
                    package,
                    activity,
                    self.profile.viewport(),
                    previous_actions=tuple(self._history[-8:]),
                )
                return synchronized.state
            except (EmulatorError, ValueError) as error:
                last_error = error
        raise EmulatorError(f"could not capture a synchronized state: {last_error}")

    def act(self, box: NormalizedBox) -> AndroidState:
        self._history.append(self.actuator.tap(box))
        return self.capture()

    def scroll(self) -> AndroidState:
        self._history.append(self.actuator.scroll())
        return self.capture()

    def back(self) -> AndroidState:
        self._history.append(self.actuator.back())
        return self.capture()


def hash_window(nodes: Sequence[AccessibilityNode]) -> int:
    """Stable pseudo window id from the root package/class of a dump."""
    if not nodes:
        return 0
    root = nodes[0]
    return abs(hash((root.package, root.class_name))) % 100_000


@dataclass
class EmulatorFleet:
    """Boot AVDs one at a time so a run never holds two emulators at once."""

    adb_path: str = "adb"
    emulator_path: str = "emulator"
    runner: CommandRunner | None = None
    allow_input: bool = False
    boot_launcher: Callable[[EmulatorProfile], subprocess.Popen] | None = None
    _started: list[subprocess.Popen] = field(default_factory=list, repr=False)

    def client(self, profile: EmulatorProfile) -> AdbClient:
        return AdbClient(profile.serial, adb_path=self.adb_path, runner=self.runner)

    def start(self, profile: EmulatorProfile) -> subprocess.Popen | None:
        """Launch the AVD unless the caller already booted it manually."""
        if profile.serial and profile.serial in {
            name for name, _ in self.client(profile).devices()
        }:
            return None
        launcher = self.boot_launcher or self._default_launcher
        process = launcher(profile)
        self._started.append(process)
        return process

    def _default_launcher(self, profile: EmulatorProfile) -> subprocess.Popen:
        command = [
            self.emulator_path,
            "-avd",
            profile.avd_name,
            "-no-snapshot-save",
            "-no-boot-anim",
            *profile.extra_emulator_args,
        ]
        if profile.serial:
            command += ["-port", profile.serial.split("-")[-1]]
        return subprocess.Popen(  # noqa: S603 - operator-supplied local AVD name
            command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

    def stop_all(self) -> None:
        for process in self._started:
            process.terminate()
        self._started.clear()

    def sequential(
        self, profiles: Iterable[EmulatorProfile]
    ) -> Iterator[tuple[EmulatorProfile, AdbEmulatorEnvironment]]:
        """Yield one booted environment per profile, in the given order."""
        for profile in profiles:
            process = self.start(profile)
            try:
                adb = self.client(profile)
                adb.wait_for_boot(profile.boot_timeout_s)
                actuator = AdbResearchActuator(
                    adb, profile.viewport(), allow_input=self.allow_input
                )
                yield profile, AdbEmulatorEnvironment(profile, adb, actuator)
            finally:
                if process is not None:
                    process.terminate()
                    if process in self._started:
                        self._started.remove(process)
