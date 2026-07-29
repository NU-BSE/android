from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

from android_drlqn.emulator_runtime import (
    AdbClient,
    BootStatus,
    CommandResult,
    EmulatorBootTimeout,
    EmulatorExitedError,
    EmulatorLauncher,
    serial_for_port,
)


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.value += seconds


class FakeProcess:
    def __init__(self, returncode=None) -> None:
        self.returncode = returncode
        self.wait_calls: list[float | None] = []
        self.terminated = False
        self.killed = False

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.wait_calls.append(timeout)
        if self.returncode is None:
            raise subprocess.TimeoutExpired("emulator", timeout)
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = 143

    def kill(self):
        self.killed = True
        self.returncode = 137


def test_wait_for_boot_uses_serial_scoped_bounded_probes():
    clock = FakeClock()
    get_state_calls = 0
    calls = []

    def runner(command, timeout_s):
        nonlocal get_state_calls
        calls.append((tuple(command), timeout_s))
        suffix = tuple(command[3:])
        if suffix == ("get-state",):
            get_state_calls += 1
            return CommandResult(
                0,
                "offline" if get_state_calls == 1 else "device",
            )
        if suffix == ("shell", "getprop", "sys.boot_completed"):
            return CommandResult(0, "1")
        if suffix == ("shell", "getprop", "dev.bootcomplete"):
            return CommandResult(0, "1")
        if suffix == ("shell", "getprop", "init.svc.bootanim"):
            return CommandResult(0, "stopped")
        if suffix == ("shell", "cmd", "package", "path", "android"):
            return CommandResult(
                0,
                "package:/system/framework/framework-res.apk",
            )
        raise AssertionError(suffix)

    adb = AdbClient(
        runner=runner,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )
    status = adb.wait_for_boot(
        "emulator-5554",
        10,
        emulator_process=FakeProcess(),
        poll_interval_s=0.5,
        probe_timeout_s=2,
    )

    assert status.ready
    assert clock.sleeps == [0.5]
    assert all(
        command[:3] == ("adb", "-s", "emulator-5554")
        for command, _ in calls
    )
    assert all(timeout <= 5.0 for _, timeout in calls)
    assert not any("wait-for-device" in command for command, _ in calls)


def test_wait_for_boot_stops_when_owned_process_exits():
    adb = AdbClient(
        runner=lambda *_: (_ for _ in ()).throw(
            AssertionError("ADB must not run")
        )
    )
    with pytest.raises(EmulatorExitedError, match="code 17"):
        adb.wait_for_boot(
            "emulator-5556",
            30,
            emulator_process=FakeProcess(17),
        )


def test_wait_for_boot_honors_monotonic_deadline():
    clock = FakeClock()

    def runner(command, timeout_s):
        del command, timeout_s
        return CommandResult(1, "", "device not found")

    adb = AdbClient(
        runner=runner,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )
    with pytest.raises(EmulatorBootTimeout) as raised:
        adb.wait_for_boot(
            "emulator-5558",
            1.2,
            emulator_process=FakeProcess(),
            poll_interval_s=0.5,
            probe_timeout_s=0.2,
        )

    assert clock.value == pytest.approx(1.2)
    assert raised.value.status.adb_state == "unavailable"
    assert "device not found" in raised.value.status.last_error


@dataclass
class Profile:
    profile_id: str = "aosp16-tablet"
    kind: str = "android_avd"
    boot_timeout_s: float = 42.0


class FakeAdb:
    def __init__(self, fail_first=False):
        self.fail_first = fail_first
        self.waits = []
        self.runs = []

    def wait_for_boot(self, serial, timeout_s, *, emulator_process):
        self.waits.append((serial, timeout_s, emulator_process))
        if self.fail_first and len(self.waits) == 1:
            raise EmulatorBootTimeout(timeout_s, BootStatus(serial))
        return BootStatus(
            serial,
            "device",
            "1",
            "1",
            "stopped",
            True,
        )

    def run(self, serial, args, *, timeout_s):
        self.runs.append((serial, tuple(args), timeout_s))
        return CommandResult(0)


def test_launcher_assigns_known_serial_and_profile_timeout(tmp_path: Path):
    adb = FakeAdb()
    commands = []

    def popen(command, **kwargs):
        commands.append((tuple(command), kwargs))
        return FakeProcess(0)

    launcher = EmulatorLauncher(
        adb=adb,
        emulator_path="/sdk/emulator",
        logs_dir=tmp_path,
        popen_factory=popen,
    )
    lease = launcher.start(Profile(), port=5554)

    assert lease.serial == "emulator-5554"
    assert adb.waits[0][:2] == ("emulator-5554", 42.0)
    assert commands[0][0][:4] == (
        "/sdk/emulator",
        "@aosp16-tablet",
        "-port",
        "5554",
    )
    assert "-no-window" in commands[0][0]
    assert "-no-audio" in commands[0][0]


def test_launcher_retries_once_without_snapshot(tmp_path: Path):
    adb = FakeAdb(fail_first=True)
    commands = []

    def popen(command, **kwargs):
        del kwargs
        commands.append(tuple(command))
        return FakeProcess(0)

    launcher = EmulatorLauncher(
        adb=adb,
        logs_dir=tmp_path,
        popen_factory=popen,
    )
    lease = launcher.start_with_cold_boot_retry(Profile(), port=5560)

    assert lease.cold_boot
    assert len(commands) == 2
    assert "-no-snapshot-load" not in commands[0]
    assert "-no-snapshot-load" in commands[1]
    assert "-no-snapshot-save" in commands[1]
    assert adb.runs[0][:2] == (
        "emulator-5560",
        ("emu", "kill"),
    )


def test_serial_for_port_rejects_ambiguous_or_odd_ports():
    assert serial_for_port(5554) == "emulator-5554"
    with pytest.raises(ValueError):
        serial_for_port(5555)
