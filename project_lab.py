"""Project lab plugin: emulator environment for on-policy distillation.

``create_environment`` is the ``--environment-plugin`` entry point consumed by
``scripts/train_on_policy.py`` and ``scripts/train_on_policy_ollama.py``::

    python scripts/train_on_policy_ollama.py \
      --documents artifacts/normalized-documents.json \
      --student-plugin project_train:create_student \
      --environment-plugin project_lab:create_environment

Each environment drives one Android emulator over adb: it launches the
Settings app, captures timestamp-aligned screenshots plus the uiautomator
hierarchy, executes the student's proposal (tap/scroll/back/wait), and
verifies transitions against the task's expected setting.

Safety contract (mirrors ``android_drlqn.emulator.AdbResearchActuator``):

- input injection (tap/scroll/back/wake) requires an explicit opt-in --
  ``allow_input=True`` or ``ANDROID_DRLQN_ALLOW_EMULATOR_INPUT=1``;
- the target must positively identify as an emulator
  (``ro.kernel.qemu=1`` or emulator/sdk build characteristics);
- screenshots stay under ``artifacts/lab-screenshots`` (or another directory
  listed in ``server.allowed_image_roots``) so the Ollama teacher accepts
  them; HTTP URLs are never produced.

Configuration (environment variables):

- ``ANDROID_DRLQN_ADB_SERIAL`` -- optional global fallback serial; it is used
  only when it matches the requested profile
- ``ANDROID_DRLQN_ADB_SERIAL_<PROFILE_ID>`` -- preferred profile-specific serial
- ``ANDROID_DRLQN_ALLOW_EMULATOR_INPUT`` -- "1" to permit input injection
- ``ANDROID_DRLQN_LAB_SCREENSHOTS`` -- screenshot directory
  (default ``artifacts/lab-screenshots``)
- ``ANDROID_DRLQN_LAB_AUTOBOOT`` -- "1" to manage matching AVDs sequentially:
  the plugin stops the previous plugin-owned AVD before booting the next one
  (uses ``EmulatorLauncher``; default off)
- ``ANDROID_DRLQN_ADB`` / ``ANDROID_DRLQN_EMULATOR`` -- tool paths

The module is importable with only the repository dependencies; adb is only
invoked through the bounded, serial-scoped ``AdbClient``.
"""
from __future__ import annotations

import atexit
import hashlib
import os
import re
import subprocess
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence

from android_drlqn.distillation import (
    EnvironmentOutcome,
    ProposalKind,
    RolloutObservation,
    StudentProposal,
)
from android_drlqn.document_curriculum import CurriculumTask
from android_drlqn.emulator_lab import EmulatorProfile
from android_drlqn.emulator_runtime import (
    AdbClient,
    CommandResult,
    EmulatorLauncher,
)

ENV_SERIAL = "ANDROID_DRLQN_ADB_SERIAL"
ENV_ALLOW_INPUT = "ANDROID_DRLQN_ALLOW_EMULATOR_INPUT"
ENV_SCREENSHOTS = "ANDROID_DRLQN_LAB_SCREENSHOTS"
ENV_AUTOBOOT = "ANDROID_DRLQN_LAB_AUTOBOOT"
ENV_ADB = "ANDROID_DRLQN_ADB"
ENV_EMULATOR = "ANDROID_DRLQN_EMULATOR"
ENV_MANUAL_NAVIGATION = "ANDROID_DRLQN_MANUAL_NAVIGATION_ONLY"

_SEARCH_MARKERS = (
    "search",
    "search settings",
    "settings search",
    "поиск",
    "поиск настроек",
    "настройки поиска",
)

DEFAULT_SCREENSHOT_DIR = "artifacts/lab-screenshots"
_MAX_IMAGE_BYTES = 20 * 1024 * 1024  # matches config/ollama_models.json
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_FOCUS_PATTERNS = (
    re.compile(r"mCurrentFocus=\S+\s+\S+\s+([\w.]+)/([\w.$]+)"),
    re.compile(
        r"(?:mFocusedApp|topResumedActivity)[^\n]*?(?:u\d+\s+)?"
        r"([A-Za-z0-9_.]+)/([A-Za-z0-9_.$]+)"
    ),
)
_BOUNDS_RE = re.compile(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]")
_FIRST_PORT = 5554

# (adb_path, serial, args, timeout_s) -> raw stdout bytes; used for screencap,
# whose PNG payload must not pass through text-mode decoding.
BinaryRunner = Callable[[str, str, Sequence[str], float], bytes]


def _subprocess_binary_runner(
    adb_path: str, serial: str, args: Sequence[str], timeout_s: float
) -> bytes:
    completed = subprocess.run(
        [adb_path, "-s", serial, *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=timeout_s,
    )
    if completed.returncode != 0:
        raise LabEnvironmentError(
            f"{serial}: adb {' '.join(args)} failed: "
            f"{completed.stderr.decode(errors='replace').strip()}"
        )
    return completed.stdout


# ---------------------------------------------------------------------------
# Observation parsing helpers (pure, unit-testable)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UiNode:
    text: str
    content_desc: str
    bounds: tuple[int, int, int, int] | None
    clickable: bool
    focused: bool

    @property
    def label(self) -> str:
        return self.text or self.content_desc

    @property
    def center(self) -> tuple[int, int] | None:
        if self.bounds is None:
            return None
        x0, y0, x1, y1 = self.bounds
        return ((x0 + x1) // 2, (y0 + y1) // 2)


def parse_ui_hierarchy(payload: str) -> tuple[UiNode, ...]:
    """Parse a ``uiautomator dump`` document into accessibility nodes."""
    try:
        root = ET.fromstring(payload)
    except ET.ParseError:
        return ()
    nodes = []
    for element in root.iter("node"):
        bounds_match = _BOUNDS_RE.fullmatch(element.get("bounds") or "")
        bounds = None
        if bounds_match:
            bounds = tuple(int(value) for value in bounds_match.groups())
        nodes.append(
            UiNode(
                text=(element.get("text") or "").strip(),
                content_desc=(element.get("content-desc") or "").strip(),
                bounds=bounds,  # type: ignore[arg-type]
                clickable=(element.get("clickable") or "false") == "true",
                focused=(element.get("focused") or "false") == "true",
            )
        )
    return tuple(nodes)


def visible_labels(nodes: Sequence[UiNode], limit: int = 40) -> tuple[str, ...]:
    seen: list[str] = []
    for node in nodes:
        label = node.label
        if label and label not in seen:
            seen.append(label)
        if len(seen) >= limit:
            break
    return tuple(seen)


def parse_current_focus(dumpsys_window: str) -> tuple[str, str]:
    """Extract ``(package, activity)`` from common Android window formats."""
    for line in dumpsys_window.splitlines():
        for pattern in _FOCUS_PATTERNS:
            match = pattern.search(line)
            if match:
                return match.group(1), match.group(2)
    return "", ""


def _normalize(value: str) -> str:
    return " ".join(value.casefold().split())


def label_matches(expected: str, labels: Sequence[str]) -> bool:
    needle = _normalize(expected)
    if not needle:
        return False
    for label in labels:
        normalized = _normalize(label)
        if normalized == needle or needle in normalized:
            return True
    return False


def expected_labels(task: CurriculumTask) -> tuple[str, ...]:
    output = task.expected_output if isinstance(task.expected_output, Mapping) else {}
    values = []
    for key in ("display_name", "target_label", "setting_id"):
        value = output.get(key)
        if value and str(value).strip():
            values.append(str(value).strip())
    if task.setting_id and str(task.setting_id).strip():
        values.append(str(task.setting_id).strip())
    return tuple(dict.fromkeys(values))


def _looks_like_settings_search_label(value: object) -> bool:
    text = str(value or "").strip().casefold()
    return bool(text) and any(marker in text for marker in _SEARCH_MARKERS)


def _looks_like_settings_search_screen(
    observation: RolloutObservation,
) -> bool:
    values = [
        observation.activity,
        *observation.visible_labels,
    ]
    return any(_looks_like_settings_search_label(value) for value in values)

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class LabEnvironmentError(RuntimeError):
    """Raised for unsafe or unusable lab environment configurations."""


@dataclass
class AdbSettingsEnvironment:
    """One Settings-app episode on one emulator, driven over adb."""

    adb: AdbClient
    serial: str
    screenshots_dir: Path = Path(DEFAULT_SCREENSHOT_DIR)
    allow_input: bool = False
    settle_s: float = 0.75
    binary_runner: BinaryRunner = _subprocess_binary_runner
    _task: CurriculumTask | None = field(default=None, init=False)
    _profile: EmulatorProfile | None = field(default=None, init=False)
    _step_index: int = field(default=0, init=False)
    _nodes: tuple[UiNode, ...] = field(default=(), init=False)
    _verified: bool = field(default=False, init=False)
    _last_observation: RolloutObservation | None = field(default=None, init=False)
    _close_callback: Callable[[], None] | None = field(
        default=None, repr=False, compare=False
    )
    _closed: bool = field(default=False, init=False, repr=False)

    # -- EmulatorEnvironment protocol --------------------------------------

    def reset(self, task: CurriculumTask, profile: EmulatorProfile) -> RolloutObservation:
        self._task = task
        self._profile = profile
        self._step_index = 0
        self._verified = False
        self._last_observation = None
        self._require_safe_target()
        self._input("keyevent", "KEYCODE_WAKEUP")
        self._shell_checked("wm", "dismiss-keyguard")

        self._shell_checked(
            "am",
            "force-stop",
            "com.android.settings",
        )
        self._shell_checked(
            "am",
            "start",
            "-W",
            "-a",
            "android.settings.SETTINGS",
        )
        time.sleep(max(self.settle_s, 1.5))
        observation = self._observe()
        self._last_observation = observation
        return observation

    def step(self, proposal: StudentProposal) -> EnvironmentOutcome:
        if self._task is None or self._last_observation is None:
            raise LabEnvironmentError("reset() must be called before step()")
        before = self._last_observation
        kind = proposal.kind
        manual_navigation = os.environ.get(ENV_MANUAL_NAVIGATION, "1") == "1"
        if kind in (ProposalKind.EXPLAIN, ProposalKind.ASK_CONFIRMATION):
            return EnvironmentOutcome(
                reward=0.0,
                terminal=True,
                verified=False,
                next_observation=None,
                reason="language turn; verification delegated to the teacher",
            )
        if kind is ProposalKind.WAIT:
            time.sleep(1.0)
        elif kind is ProposalKind.BACK:
            self._input("keyevent", "KEYCODE_BACK")
        elif kind is ProposalKind.SCROLL:
            self._scroll()
        elif kind is ProposalKind.TYPE:
            typed = (proposal.text or "").strip()
            if not typed:
                observation = self._observe()
                self._last_observation = observation
                return EnvironmentOutcome(
                    reward=-0.05,
                    terminal=False,
                    verified=False,
                    next_observation=observation,
                    reason="type action carried no text",
                )
            # `adb shell input text` splits on spaces; %s is the documented
            # escape for a literal space.
            self._input("text", typed.replace(" ", "%s"))
        elif kind is ProposalKind.TAP:
            if (
                manual_navigation
                and kind is ProposalKind.TAP
                and _looks_like_settings_search_label(proposal.target_label)
            ):
                return EnvironmentOutcome(
                    reward=-0.25,
                    terminal=False,
                    verified=False,
                    next_observation=before,
                    reason=(
                        "Settings Search is forbidden; navigate through visible "
                        "settings categories manually"
                    ),
                )
            point = self._resolve_tap(proposal)
            if point is None:
                observation = self._observe()
                self._last_observation = observation
                return EnvironmentOutcome(
                    reward=-0.05,
                    terminal=False,
                    verified=False,
                    next_observation=observation,
                    reason="tap target could not be resolved on screen",
                )
            self._input("tap", str(point[0]), str(point[1]))
        else:  # pragma: no cover - ProposalKind is exhaustive
            observation = self._observe()
            self._last_observation = observation
            return EnvironmentOutcome(
                -0.05, False, False, observation, "unsupported action"
            )
        time.sleep(self.settle_s)
        observation = self._observe()
        self._last_observation = observation

        if manual_navigation and _looks_like_settings_search_screen(observation):
            self._input("keyevent", "KEYCODE_BACK")
            time.sleep(self.settle_s)

            recovered = self._observe()
            self._last_observation = recovered

            return EnvironmentOutcome(
                reward=-0.25,
                terminal=False,
                verified=False,
                next_observation=recovered,
                reason=(
                    "Settings Search was opened; action reverted because "
                    "manual navigation is required"
                ),
            )

        verified = self._check_verified(observation, kind, before)
        reward = 1.0 if verified else 0.0
        return EnvironmentOutcome(
            reward=reward,
            terminal=verified,
            verified=verified,
            next_observation=None if verified else observation,
            reason=(
                "expected setting verified after a UI transition"
                if verified
                else "transition unverified"
            ),
        )

    def close(self) -> None:
        """Release a plugin-owned emulator after this profile finishes.

        Manually attached emulators are never stopped. The callback is installed
        only when ``create_environment`` autobooted the matching AVD.
        """
        if self._closed:
            return
        self._closed = True
        if self._close_callback is not None:
            self._close_callback()

    def __enter__(self) -> "AdbSettingsEnvironment":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    # -- internals ----------------------------------------------------------

    def _require_safe_target(self) -> None:
        if not _is_emulator(self.adb, self.serial):
            raise LabEnvironmentError(
                f"{self.serial}: target is not an emulator; the lab environment "
                "never drives personal devices"
            )
        if not self.allow_input:
            raise LabEnvironmentError(
                "input injection requires allow_input=True or "
                f"{ENV_ALLOW_INPUT}=1 (emulator-only research actuator)"
            )
        if self._profile is not None:
            sdk = self.adb.run(
                self.serial,
                ("shell", "getprop", "ro.build.version.sdk"),
                timeout_s=5.0,
            )
            actual = sdk.stdout.strip() if sdk.returncode == 0 else ""
            if actual.isdigit() and int(actual) != int(self._profile.api_level):
                raise LabEnvironmentError(
                    f"{self.serial}: API {actual} does not match profile "
                    f"{self._profile.profile_id} API {self._profile.api_level}"
                )

    def _observe(self) -> RolloutObservation:
        self._step_index += 1
        index = self._step_index - 1
        screenshot = self._capture_screenshot(index)
        self._nodes = self._dump_nodes()
        package, activity = self._current_focus()
        summary = {
            "node_count": len(self._nodes),
            "clickable_count": sum(1 for node in self._nodes if node.clickable),
            "focused_label": next((node.label for node in self._nodes if node.focused), ""),
            "captured_at": round(time.time(), 3),
        }
        return RolloutObservation(
            screenshot_reference=screenshot,
            visible_labels=visible_labels(self._nodes),
            accessibility_summary=summary,
            package=package,
            activity=activity,
            step_index=index,
        )

    def _capture_screenshot(self, index: int) -> str:
        payload = self.binary_runner(
            self.adb.adb_path, self.serial, ("exec-out", "screencap", "-p"), 10.0
        )
        if not payload.startswith(_PNG_MAGIC):
            raise LabEnvironmentError(
                f"{self.serial}: screencap did not return a PNG payload"
            )
        if len(payload) > _MAX_IMAGE_BYTES:
            raise LabEnvironmentError(
                f"{self.serial}: screenshot exceeds the teacher image size limit"
            )
        directory = self.screenshots_dir / (self._profile.profile_id if self._profile else "unknown")
        directory.mkdir(parents=True, exist_ok=True)
        task_id = self._task.task_id if self._task else "task"
        safe_task = re.sub(r"[^A-Za-z0-9_.-]+", "-", task_id).strip("-")[:48] or "task"
        task_hash = hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:8]
        path = directory / f"{safe_task}-{task_hash}-s{index:02d}.png"
        path.write_bytes(payload)
        return path.as_posix()

    def _dump_nodes(self) -> tuple[UiNode, ...]:
        remote = "/sdcard/window_dump.xml"
        result = self._shell(
            "uiautomator", "dump", "--compressed", remote, timeout_s=10.0
        )
        if result.returncode != 0:  # --compressed needs API 30+
            self._shell("uiautomator", "dump", remote, timeout_s=10.0)
        result = self.adb.run(self.serial, ("exec-out", "cat", remote), timeout_s=10.0)
        if result.returncode != 0 or "<hierarchy" not in result.stdout:
            return ()
        return parse_ui_hierarchy(result.stdout)

    def _current_focus(self) -> tuple[str, str]:
        result = self._shell("dumpsys", "window", "displays", timeout_s=8.0)
        return parse_current_focus(result.stdout)

    def _resolve_tap(self, proposal: StudentProposal) -> tuple[int, int] | None:
        width, height = self._display_size()
        if proposal.bbox_norm is not None:
            x0, y0, x1, y1 = proposal.bbox_norm
            return (int((x0 + x1) / 2 * width), int((y0 + y1) / 2 * height))
        if proposal.target_label:
            needle = _normalize(proposal.target_label)
            for node in self._nodes:
                if node.center is None:
                    continue
                label = _normalize(node.label)
                if label and (label == needle or needle in label):
                    return node.center
        return None

    def _scroll(self) -> None:
        width, height = self._display_size()
        x = width // 2
        self._input(
            "swipe",
            str(x),
            str(int(height * 0.7)),
            str(x),
            str(int(height * 0.3)),
            "300",
        )

    def _display_size(self) -> tuple[int, int]:
        result = self._shell("wm", "size")
        match = re.search(r"(\d+)x(\d+)", result.stdout)
        if not match:
            return (1080, 2400)
        return int(match.group(1)), int(match.group(2))

    def _check_verified(
        self,
        observation: RolloutObservation,
        kind: ProposalKind,
        before: RolloutObservation,
    ) -> bool:
        if self._verified or self._task is None:
            return self._verified
        expected = expected_labels(self._task)
        after_match = any(
            label_matches(value, observation.visible_labels) for value in expected
        )
        if not after_match:
            return False
        before_match = any(
            label_matches(value, before.visible_labels) for value in expected
        )
        focus_changed = (
            observation.package,
            observation.activity,
        ) != (before.package, before.activity)
        labels_changed = observation.visible_labels != before.visible_labels
        transitioned = focus_changed or labels_changed

        # A label that was already visible cannot turn WAIT/BACK/SCROLL into a
        # false success. A TAP must cause a real UI transition; other actions
        # may succeed only when they newly reveal the expected setting.
        self._verified = (
            transitioned if kind is ProposalKind.TAP else (not before_match)
        )
        return self._verified

    def _shell(self, *args: str, timeout_s: float = 5.0) -> CommandResult:
        return self.adb.run(self.serial, ("shell", *args), timeout_s=timeout_s)

    def _shell_checked(self, *args: str, timeout_s: float = 5.0) -> CommandResult:
        result = self._shell(*args, timeout_s=timeout_s)
        if result.returncode != 0:
            raise LabEnvironmentError(
                f"{self.serial}: adb shell {' '.join(args)} failed: "
                f"{result.stderr.strip()}"
            )
        return result

    def _input(self, *args: str) -> CommandResult:
        if not self.allow_input:
            raise LabEnvironmentError("input injection is not enabled for this target")
        result = self.adb.run(
            self.serial, ("shell", "input", *args), timeout_s=5.0
        )
        if result.returncode != 0:
            raise LabEnvironmentError(
                f"{self.serial}: input {' '.join(args)} failed: "
                f"{result.stderr.strip()}"
            )
        return result


# ---------------------------------------------------------------------------
# Emulator discovery / safety
# ---------------------------------------------------------------------------


def _is_emulator(adb: AdbClient, serial: str) -> bool:
    qemu_result = adb.run(
        serial, ("shell", "getprop", "ro.kernel.qemu"), timeout_s=5.0
    )
    characteristics_result = adb.run(
        serial,
        ("shell", "getprop", "ro.build.characteristics"),
        timeout_s=5.0,
    )
    if qemu_result.returncode != 0 or characteristics_result.returncode != 0:
        return False
    qemu = qemu_result.stdout.strip()
    characteristics = characteristics_result.stdout.casefold()
    return qemu == "1" or (
        serial.startswith("emulator-") and "emulator" in characteristics
    )


def _attached_emulator_serials(adb: AdbClient) -> tuple[str, ...]:
    result = adb._runner((adb.adb_path, "devices"), 5.0)
    if result.returncode != 0:
        return ()
    serials: list[str] = []
    for line in result.stdout.splitlines()[1:]:
        parts = line.split()
        if (
            len(parts) >= 2
            and parts[1] == "device"
            and parts[0].startswith("emulator-")
        ):
            serials.append(parts[0])
    return tuple(serials)


def _emulator_api(adb: AdbClient, serial: str) -> int | None:
    result = adb.run(
        serial,
        ("shell", "getprop", "ro.build.version.sdk"),
        timeout_s=5.0,
    )
    value = result.stdout.strip() if result.returncode == 0 else ""
    return int(value) if value.isdigit() else None


def _emulator_avd_name(adb: AdbClient, serial: str) -> str:
    result = adb.run(serial, ("emu", "avd", "name"), timeout_s=5.0)
    if result.returncode != 0:
        return ""
    for line in result.stdout.splitlines():
        value = line.strip()
        if value and value != "OK":
            return value
    return ""


def _normalize_avd_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _expected_avd_name(profile: EmulatorProfile) -> str:
    return str(profile.avd_device or profile.profile_id)


def _serial_mismatch_reason(
    adb: AdbClient,
    serial: str,
    profile: EmulatorProfile,
    *,
    require_avd_match: bool,
) -> str | None:
    if not _is_emulator(adb, serial):
        return "target is not an emulator"

    actual_api = _emulator_api(adb, serial)
    if actual_api is None:
        return "could not read ro.build.version.sdk"
    if actual_api != int(profile.api_level):
        return (
            f"API {actual_api} does not match profile {profile.profile_id} "
            f"API {profile.api_level}"
        )

    expected_avd = _expected_avd_name(profile)
    actual_avd = _emulator_avd_name(adb, serial)
    if actual_avd and (
        _normalize_avd_name(actual_avd) != _normalize_avd_name(expected_avd)
    ):
        return f"AVD {actual_avd!r} does not match expected {expected_avd!r}"
    elif not actual_avd and (require_avd_match or profile.os_family != "android-aosp"):
        return f"could not verify expected AVD {expected_avd!r}"
    return None


def _require_serial_for_profile(
    adb: AdbClient,
    serial: str,
    profile: EmulatorProfile,
    *,
    source: str,
) -> str:
    reason = _serial_mismatch_reason(
        adb,
        serial,
        profile,
        require_avd_match=profile.os_family != "android-aosp",
    )
    if reason is not None:
        raise LabEnvironmentError(f"{source}={serial}: {reason}")
    return serial


def _matching_emulator_serial(
    adb: AdbClient,
    profile: EmulatorProfile,
) -> str | None:
    """Return an attached emulator matching this profile, never just the first one."""
    expected_avd = _normalize_avd_name(_expected_avd_name(profile))
    api_only_matches: list[str] = []

    for serial in _attached_emulator_serials(adb):
        if not _is_emulator(adb, serial):
            continue
        if _emulator_api(adb, serial) != int(profile.api_level):
            continue

        actual_avd = _normalize_avd_name(_emulator_avd_name(adb, serial))
        if actual_avd:
            if actual_avd == expected_avd:
                return serial
            # A known, different AVD is never an API-only match.
            continue

        # An API-only fallback is acceptable only when adb cannot report the
        # AVD name and only for the stock AOSP lane. Custom OS profiles must
        # prove their AVD identity to avoid mislabeled rollouts.
        if profile.os_family == "android-aosp":
            api_only_matches.append(serial)

    # Avoid picking arbitrarily when two attached AOSP emulators share an API.
    return api_only_matches[0] if len(api_only_matches) == 1 else None


def _attached_emulator_summary(adb: AdbClient) -> str:
    rows: list[str] = []
    for serial in _attached_emulator_serials(adb):
        api = _emulator_api(adb, serial)
        avd = _emulator_avd_name(adb, serial) or "unknown-avd"
        rows.append(f"{serial}(API {api if api is not None else '?'}, AVD {avd})")
    return ", ".join(rows) if rows else "none"


# Only emulators started by this module are lifecycle-managed. A user-started
# emulator is never killed by the plugin.
_ACTIVE_LEASE: tuple[str, object, str] | None = None


def _stop_lease(lease: object, adb_path: str) -> None:
    serial = str(getattr(lease, "serial", ""))
    if not serial:
        return
    try:
        AdbClient(adb_path=adb_path).run(
            serial,
            ("emu", "kill"),
            timeout_s=5.0,
        )
    except Exception:
        return

    # Free the emulator console/adb port before the next sequential profile.
    deadline = time.monotonic() + 20.0
    adb = AdbClient(adb_path=adb_path)
    while time.monotonic() < deadline:
        if serial not in _attached_emulator_serials(adb):
            break
        time.sleep(0.25)


def _stop_active_autoboot() -> None:
    global _ACTIVE_LEASE
    active = _ACTIVE_LEASE
    _ACTIVE_LEASE = None
    if active is None:
        return
    _, lease, adb_path = active
    _stop_lease(lease, adb_path)


def _stop_active_if_different(serial: str) -> None:
    active = _ACTIVE_LEASE
    if active is None:
        return
    active_serial = str(getattr(active[1], "serial", ""))
    if active_serial and active_serial != serial:
        _stop_active_autoboot()


def _autoboot(profile: EmulatorProfile, adb_path: str) -> str:
    """Boot exactly one matching AVD for sequential profile-major training."""
    global _ACTIVE_LEASE

    avd = _expected_avd_name(profile)
    if _ACTIVE_LEASE is not None:
        active_avd, lease, active_adb_path = _ACTIVE_LEASE
        active_serial = str(getattr(lease, "serial", ""))
        if (
            active_avd == avd
            and active_adb_path == adb_path
            and active_serial
            and active_serial in _attached_emulator_serials(
                AdbClient(adb_path=adb_path)
            )
        ):
            return active_serial
        _stop_active_autoboot()

    launcher = EmulatorLauncher(
        AdbClient(adb_path=adb_path),
        emulator_path=os.environ.get(ENV_EMULATOR, "emulator"),
    )
    # Reuse one fixed port because only one plugin-owned AVD is alive at once.
    lease = launcher.start_with_cold_boot_retry(
        profile,
        port=_FIRST_PORT,
        avd_name=avd,
    )
    _ACTIVE_LEASE = (avd, lease, adb_path)
    return str(lease.serial)


def _release_managed_serial(serial: str) -> None:
    active = _ACTIVE_LEASE
    if active is None:
        return
    active_serial = str(getattr(active[1], "serial", ""))
    if active_serial == serial:
        _stop_active_autoboot()


@atexit.register
def _stop_booted_emulators() -> None:
    _stop_active_autoboot()


def _profile_serial_env(profile: EmulatorProfile) -> str:
    suffix = re.sub(r"[^A-Za-z0-9]+", "_", profile.profile_id).strip("_").upper()
    return f"{ENV_SERIAL}_{suffix}"


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------


def create_environment(
    profile: EmulatorProfile,
    os_settings_context: Mapping[str, object] | None = None,
    *,
    serial: str | None = None,
    adb_path: str | None = None,
    screenshots_dir: str | Path | None = None,
    allow_input: bool | None = None,
    adb: AdbClient | None = None,
) -> AdbSettingsEnvironment:
    """``--environment-plugin`` factory for emulator-backed training.

    ``os_settings_context`` is accepted for signature compatibility with the
    training scripts; the student consumes it through the enriched task
    evidence, so the environment only needs the profile.
    """
    del os_settings_context  # settings evidence travels inside task.evidence
    resolved_adb = adb or AdbClient(
        adb_path=adb_path or os.environ.get(ENV_ADB, "adb")
    )
    profile_env = _profile_serial_env(profile)
    profile_serial = os.environ.get(profile_env)
    global_serial = os.environ.get(ENV_SERIAL)
    autoboot_enabled = os.environ.get(ENV_AUTOBOOT) == "1"

    resolved_serial: str | None = None
    managed_serial = False
    binding_errors: list[str] = []

    # A function argument is an explicit caller contract and therefore strict.
    if serial:
        resolved_serial = _require_serial_for_profile(
            resolved_adb, serial, profile, source="serial"
        )

    # Environment bindings may be stale after the previous sequential AVD was
    # stopped. With autoboot enabled, ignore stale bindings and recover by
    # discovering/booting the correct profile. Without autoboot, report them.
    if resolved_serial is None and profile_serial:
        reason = _serial_mismatch_reason(
            resolved_adb,
            profile_serial,
            profile,
            require_avd_match=True,
        )
        if reason is None:
            resolved_serial = profile_serial
        else:
            binding_errors.append(f"{profile_env}={profile_serial}: {reason}")

    if resolved_serial is None and global_serial:
        reason = _serial_mismatch_reason(
            resolved_adb,
            global_serial,
            profile,
            require_avd_match=True,
        )
        if reason is None:
            resolved_serial = global_serial
        else:
            binding_errors.append(f"{ENV_SERIAL}={global_serial}: {reason}")

    if resolved_serial is None:
        resolved_serial = _matching_emulator_serial(resolved_adb, profile)

    if resolved_serial is None and autoboot_enabled:
        resolved_serial = _autoboot(profile, resolved_adb.adb_path)
        managed_serial = True

    if resolved_serial is None:
        stale = f" Stale bindings: {'; '.join(binding_errors)}." if binding_errors else ""
        raise LabEnvironmentError(
            f"no emulator matching profile {profile.profile_id} "
            f"(API {profile.api_level}, AVD {_expected_avd_name(profile)!r}); "
            f"attached emulators: {_attached_emulator_summary(resolved_adb)}."
            f"{stale} Set {profile_env}, or enable {ENV_AUTOBOOT}=1"
        )

    _stop_active_if_different(resolved_serial)
    if _ACTIVE_LEASE is not None:
        managed_serial = (
            str(getattr(_ACTIVE_LEASE[1], "serial", "")) == resolved_serial
        )

    if allow_input is None:
        allow_input = os.environ.get(ENV_ALLOW_INPUT) == "1"
    return AdbSettingsEnvironment(
        adb=resolved_adb,
        serial=resolved_serial,
        screenshots_dir=Path(
            screenshots_dir
            or os.environ.get(ENV_SCREENSHOTS, DEFAULT_SCREENSHOT_DIR)
        ),
        allow_input=bool(allow_input),
        _close_callback=(
            (lambda serial=resolved_serial: _release_managed_serial(serial))
            if managed_serial
            else None
        ),
    )


__all__ = [
    "AdbSettingsEnvironment",
    "LabEnvironmentError",
    "UiNode",
    "create_environment",
    "expected_labels",
    "label_matches",
    "parse_current_focus",
    "parse_ui_hierarchy",
    "visible_labels",
]
