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

- ``ANDROID_DRLQN_ADB_SERIAL`` -- target serial (default: first ``emulator-*``)
- ``ANDROID_DRLQN_ALLOW_EMULATOR_INPUT`` -- "1" to permit input injection
- ``ANDROID_DRLQN_LAB_SCREENSHOTS`` -- screenshot directory
  (default ``artifacts/lab-screenshots``)
- ``ANDROID_DRLQN_LAB_AUTOBOOT`` -- "1" to boot a matching AVD when no
  emulator is attached (uses ``EmulatorLauncher``; default off)
- ``ANDROID_DRLQN_ADB`` / ``ANDROID_DRLQN_EMULATOR`` -- tool paths

The module is importable with only the repository dependencies; adb is only
invoked through the bounded, serial-scoped ``AdbClient``.
"""
from __future__ import annotations

import atexit
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
    serial_for_port,
)

ENV_SERIAL = "ANDROID_DRLQN_ADB_SERIAL"
ENV_ALLOW_INPUT = "ANDROID_DRLQN_ALLOW_EMULATOR_INPUT"
ENV_SCREENSHOTS = "ANDROID_DRLQN_LAB_SCREENSHOTS"
ENV_AUTOBOOT = "ANDROID_DRLQN_LAB_AUTOBOOT"
ENV_ADB = "ANDROID_DRLQN_ADB"
ENV_EMULATOR = "ANDROID_DRLQN_EMULATOR"

DEFAULT_SCREENSHOT_DIR = "artifacts/lab-screenshots"
_MAX_IMAGE_BYTES = 20 * 1024 * 1024  # matches config/ollama_models.json
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_FOCUS_RE = re.compile(r"mCurrentFocus=\S+\s+\S+\s+([\w.]+)/([\w.$]+)")
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
    """Extract ``(package, activity)`` from ``dumpsys window`` output."""
    for line in dumpsys_window.splitlines():
        match = _FOCUS_RE.search(line)
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

    # -- EmulatorEnvironment protocol --------------------------------------

    def reset(self, task: CurriculumTask, profile: EmulatorProfile) -> RolloutObservation:
        self._require_safe_target()
        self._task = task
        self._profile = profile
        self._step_index = 0
        self._verified = False
        self._input("keyevent", "KEYCODE_WAKEUP")
        self._shell("wm", "dismiss-keyguard")
        self._shell("am", "start", "-a", "android.settings.SETTINGS")
        time.sleep(max(self.settle_s, 1.5))
        return self._observe()

    def step(self, proposal: StudentProposal) -> EnvironmentOutcome:
        if self._task is None:
            raise LabEnvironmentError("reset() must be called before step()")
        kind = proposal.kind
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
        elif kind is ProposalKind.TAP:
            point = self._resolve_tap(proposal)
            if point is None:
                return EnvironmentOutcome(
                    reward=-0.05,
                    terminal=False,
                    verified=False,
                    next_observation=self._observe(),
                    reason="tap target could not be resolved on screen",
                )
            self._input("tap", str(point[0]), str(point[1]))
        else:  # pragma: no cover - ProposalKind is exhaustive
            return EnvironmentOutcome(-0.05, False, False, self._observe(), "unsupported action")
        time.sleep(self.settle_s)
        observation = self._observe()
        verified = self._check_verified(observation)
        reward = 1.0 if verified else 0.0
        return EnvironmentOutcome(
            reward=reward,
            terminal=verified,
            verified=verified,
            next_observation=None if verified else observation,
            reason="expected setting verified on screen" if verified else "transition unverified",
        )

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
        task_id = (self._task.task_id if self._task else "task")[:12]
        path = directory / f"{task_id}-s{index:02d}.png"
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

    def _check_verified(self, observation: RolloutObservation) -> bool:
        if self._verified or self._task is None:
            return self._verified
        expected = expected_labels(self._task)
        self._verified = any(
            label_matches(value, observation.visible_labels) for value in expected
        )
        return self._verified

    def _shell(self, *args: str, timeout_s: float = 5.0) -> CommandResult:
        result = self.adb.run(self.serial, ("shell", *args), timeout_s=timeout_s)
        return result

    def _input(self, *args: str) -> CommandResult:
        if not self.allow_input:
            raise LabEnvironmentError("input injection is not enabled for this target")
        return self.adb.run(self.serial, ("shell", "input", *args), timeout_s=5.0)


# ---------------------------------------------------------------------------
# Emulator discovery / safety
# ---------------------------------------------------------------------------


def _is_emulator(adb: AdbClient, serial: str) -> bool:
    qemu = adb.run(serial, ("shell", "getprop", "ro.kernel.qemu")).stdout.strip()
    characteristics = adb.run(
        serial, ("shell", "getprop", "ro.build.characteristics")
    ).stdout.strip()
    product = adb.run(serial, ("shell", "getprop", "ro.build.product")).stdout.strip()
    return qemu == "1" or "emulator" in characteristics or "sdk" in product


def _first_emulator_serial(adb: AdbClient) -> str | None:
    result = adb._runner((adb.adb_path, "devices"), 5.0)
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines()[1:]:
        parts = line.split()
        if len(parts) == 2 and parts[1] == "device" and parts[0].startswith("emulator-"):
            return parts[0]
    return None


_LEASES: dict[str, object] = {}


def _autoboot(profile: EmulatorProfile, adb_path: str) -> str:
    """Boot a matching AVD and return its serial (process-wide registry)."""
    avd = profile.avd_device or profile.profile_id
    if avd in _LEASES:
        return _LEASES[avd].serial  # type: ignore[attr-defined]
    launcher = EmulatorLauncher(
        AdbClient(adb_path=adb_path),
        emulator_path=os.environ.get(ENV_EMULATOR, "emulator"),
    )
    port = _FIRST_PORT + 2 * len(_LEASES)
    lease = launcher.start_with_cold_boot_retry(profile, port=port, avd_name=avd)
    _LEASES[avd] = lease
    return lease.serial


@atexit.register
def _stop_booted_emulators() -> None:
    for lease in _LEASES.values():
        try:
            AdbClient().run(lease.serial, ("emu", "kill"), timeout_s=3.0)  # type: ignore[attr-defined]
        except Exception:
            pass


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
    resolved_adb = adb or AdbClient(adb_path=adb_path or os.environ.get(ENV_ADB, "adb"))
    resolved_serial = serial or os.environ.get(ENV_SERIAL)
    if not resolved_serial:
        resolved_serial = _first_emulator_serial(resolved_adb)
    if not resolved_serial and os.environ.get(ENV_AUTOBOOT) == "1":
        resolved_serial = _autoboot(profile, resolved_adb.adb_path)
    if not resolved_serial:
        raise LabEnvironmentError(
            "no emulator attached; boot an AVD first (or set "
            f"{ENV_AUTOBOOT}=1 / {ENV_SERIAL})"
        )
    if allow_input is None:
        allow_input = os.environ.get(ENV_ALLOW_INPUT) == "1"
    return AdbSettingsEnvironment(
        adb=resolved_adb,
        serial=resolved_serial,
        screenshots_dir=Path(screenshots_dir or os.environ.get(ENV_SCREENSHOTS, DEFAULT_SCREENSHOT_DIR)),
        allow_input=bool(allow_input),
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
