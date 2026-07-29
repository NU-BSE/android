"""Bounded Android Emulator startup and ADB boot readiness checks.

The runtime deliberately avoids ``adb wait-for-device``. That command only waits
for an ADB transport and can outlive a crashed emulator process. Instead, every
probe is scoped to one serial, has its own short subprocess timeout, and runs
under one monotonic deadline while the emulator process is monitored.
"""
from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol, Sequence


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""


@dataclass(frozen=True)
class BootStatus:
    serial: str
    adb_state: str = "unknown"
    sys_boot_completed: str = ""
    dev_bootcomplete: str = ""
    boot_animation: str = ""
    package_manager_ready: bool = False
    last_error: str = ""

    @property
    def ready(self) -> bool:
        return (
            self.adb_state == "device"
            and self.sys_boot_completed == "1"
            and self.dev_bootcomplete in {"", "1"}
            and self.boot_animation in {"", "stopped"}
            and self.package_manager_ready
        )


class EmulatorBootError(RuntimeError):
    """Base class for deterministic emulator startup failures."""


class EmulatorExitedError(EmulatorBootError):
    """Raised when the emulator process exits before Android becomes ready."""


class EmulatorBootTimeout(EmulatorBootError):
    """Raised when Android does not become ready before the monotonic deadline."""

    def __init__(self, timeout_s: float, status: BootStatus):
        self.timeout_s = float(timeout_s)
        self.status = status
        super().__init__(
            f"{status.serial}: Android did not become ready in {timeout_s:.1f}s; "
            f"last_status={status}"
        )


class ProcessHandle(Protocol):
    def poll(self) -> int | None: ...
    def wait(self, timeout: float | None = None) -> int: ...
    def terminate(self) -> None: ...
    def kill(self) -> None: ...


CommandRunner = Callable[[Sequence[str], float], CommandResult]
PopenFactory = Callable[..., ProcessHandle]


def _subprocess_runner(command: Sequence[str], timeout_s: float) -> CommandResult:
    try:
        completed = subprocess.run(
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = (
            exc.stdout.decode(errors="replace")
            if isinstance(exc.stdout, bytes)
            else (exc.stdout or "")
        )
        stderr = (
            exc.stderr.decode(errors="replace")
            if isinstance(exc.stderr, bytes)
            else (exc.stderr or "")
        )
        return CommandResult(
            124,
            stdout.strip(),
            (stderr or f"timeout after {timeout_s}s").strip(),
        )
    return CommandResult(
        completed.returncode,
        completed.stdout.strip(),
        completed.stderr.strip(),
    )


class AdbClient:
    """ADB client whose operations are serial-scoped and individually bounded."""

    def __init__(
        self,
        adb_path: str = "adb",
        *,
        runner: CommandRunner | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.adb_path = adb_path
        self._runner = runner or _subprocess_runner
        self._monotonic = monotonic
        self._sleep = sleep

    def run(
        self,
        serial: str,
        args: Sequence[str],
        *,
        timeout_s: float = 3.0,
    ) -> CommandResult:
        if not serial:
            raise ValueError("ADB serial is required")
        if timeout_s <= 0:
            raise ValueError("ADB command timeout must be positive")
        return self._runner(
            (self.adb_path, "-s", serial, *args),
            float(timeout_s),
        )

    def boot_status(
        self,
        serial: str,
        *,
        probe_timeout_s: float = 3.0,
    ) -> BootStatus:
        state = self.run(serial, ("get-state",), timeout_s=probe_timeout_s)
        adb_state = state.stdout.strip() if state.returncode == 0 else "unavailable"
        last_error = state.stderr.strip()
        if adb_state != "device":
            return BootStatus(
                serial=serial,
                adb_state=adb_state,
                last_error=last_error,
            )

        sys_boot = self.run(
            serial,
            ("shell", "getprop", "sys.boot_completed"),
            timeout_s=probe_timeout_s,
        )
        dev_boot = self.run(
            serial,
            ("shell", "getprop", "dev.bootcomplete"),
            timeout_s=probe_timeout_s,
        )
        bootanim = self.run(
            serial,
            ("shell", "getprop", "init.svc.bootanim"),
            timeout_s=probe_timeout_s,
        )
        package_manager = self.run(
            serial,
            ("shell", "cmd", "package", "path", "android"),
            timeout_s=max(5.0, probe_timeout_s),
        )
        errors = [
            result.stderr.strip()
            for result in (sys_boot, dev_boot, bootanim, package_manager)
            if result.returncode != 0 and result.stderr.strip()
        ]
        return BootStatus(
            serial=serial,
            adb_state=adb_state,
            sys_boot_completed=(
                sys_boot.stdout.strip() if sys_boot.returncode == 0 else ""
            ),
            dev_bootcomplete=(
                dev_boot.stdout.strip() if dev_boot.returncode == 0 else ""
            ),
            boot_animation=(
                bootanim.stdout.strip() if bootanim.returncode == 0 else ""
            ),
            package_manager_ready=(
                package_manager.returncode == 0
                and "package:" in package_manager.stdout
            ),
            last_error="; ".join(errors),
        )

    def wait_for_boot(
        self,
        serial: str,
        timeout_s: float,
        *,
        emulator_process: ProcessHandle | None = None,
        poll_interval_s: float = 0.75,
        probe_timeout_s: float = 3.0,
    ) -> BootStatus:
        """Wait until Android framework services are ready.

        Unlike ``adb wait-for-device``, this method cannot block past
        ``timeout_s`` and fails immediately if the owned emulator process exits.
        """
        if timeout_s <= 0:
            raise ValueError("boot timeout must be positive")
        if poll_interval_s <= 0:
            raise ValueError("poll interval must be positive")
        if probe_timeout_s <= 0:
            raise ValueError("probe timeout must be positive")

        deadline = self._monotonic() + float(timeout_s)
        last_status = BootStatus(serial=serial)

        while True:
            if emulator_process is not None:
                returncode = emulator_process.poll()
                if returncode is not None:
                    raise EmulatorExitedError(
                        f"{serial}: emulator exited with code {returncode}; "
                        f"last_status={last_status}"
                    )

            remaining = deadline - self._monotonic()
            if remaining <= 0:
                raise EmulatorBootTimeout(timeout_s, last_status)

            last_status = self.boot_status(
                serial,
                probe_timeout_s=min(probe_timeout_s, remaining),
            )
            if last_status.ready:
                return last_status

            remaining = deadline - self._monotonic()
            if remaining <= 0:
                raise EmulatorBootTimeout(timeout_s, last_status)
            self._sleep(min(poll_interval_s, remaining))


@dataclass(frozen=True)
class EmulatorLease:
    avd_name: str
    port: int
    serial: str
    process: ProcessHandle
    log_path: Path
    cold_boot: bool = False


def serial_for_port(port: int) -> str:
    if not 1024 <= int(port) <= 65534 or int(port) % 2:
        raise ValueError(
            "emulator console port must be an even number in 1024..65534"
        )
    return f"emulator-{int(port)}"


class EmulatorLauncher:
    """Own one emulator process, serial and log file for its full lifecycle."""

    def __init__(
        self,
        adb: AdbClient | None = None,
        *,
        emulator_path: str = "emulator",
        logs_dir: str | Path = "artifacts/emulator-logs",
        default_boot_timeout_s: float = 300.0,
        popen_factory: PopenFactory = subprocess.Popen,
    ) -> None:
        if default_boot_timeout_s <= 0:
            raise ValueError("default boot timeout must be positive")
        self.adb = adb or AdbClient()
        self.emulator_path = emulator_path
        self.logs_dir = Path(logs_dir)
        self.default_boot_timeout_s = float(default_boot_timeout_s)
        self._popen_factory = popen_factory

    def start(
        self,
        profile: object,
        *,
        port: int,
        avd_name: str | None = None,
        timeout_s: float | None = None,
        cold_boot: bool = False,
        extra_args: Sequence[str] = (),
    ) -> EmulatorLease:
        kind = getattr(
            getattr(profile, "kind", None),
            "value",
            getattr(profile, "kind", None),
        )
        if kind not in (None, "android_avd"):
            raise ValueError(
                "automatic launcher currently supports Android SDK AVD profiles only"
            )

        serial = serial_for_port(port)
        resolved_avd = avd_name or str(
            getattr(profile, "avd_name", "")
            or getattr(profile, "profile_id", "")
        )
        if not resolved_avd:
            raise ValueError("AVD name is required")

        resolved_timeout = (
            float(timeout_s)
            if timeout_s is not None
            else float(
                getattr(
                    profile,
                    "boot_timeout_s",
                    self.default_boot_timeout_s,
                )
            )
        )
        if resolved_timeout <= 0:
            raise ValueError("boot timeout must be positive")

        self.logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.logs_dir / f"{resolved_avd}-{port}.log"
        command = [
            self.emulator_path,
            f"@{resolved_avd}",
            "-port",
            str(port),
            "-no-window",
            "-no-audio",
            *extra_args,
        ]
        if cold_boot:
            command.extend(("-no-snapshot-load", "-no-snapshot-save"))

        log_file = log_path.open("w", encoding="utf-8")
        try:
            process = self._popen_factory(
                command,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
        finally:
            log_file.close()

        lease = EmulatorLease(
            avd_name=resolved_avd,
            port=port,
            serial=serial,
            process=process,
            log_path=log_path,
            cold_boot=cold_boot,
        )
        try:
            self.adb.wait_for_boot(
                serial,
                resolved_timeout,
                emulator_process=process,
            )
        except Exception:
            self.stop(lease)
            raise
        return lease

    def start_with_cold_boot_retry(
        self,
        profile: object,
        *,
        port: int,
        avd_name: str | None = None,
        timeout_s: float | None = None,
        extra_args: Sequence[str] = (),
    ) -> EmulatorLease:
        """Try Quick Boot once, then one clean cold boot after a boot failure."""
        try:
            return self.start(
                profile,
                port=port,
                avd_name=avd_name,
                timeout_s=timeout_s,
                cold_boot=False,
                extra_args=extra_args,
            )
        except (EmulatorBootTimeout, EmulatorExitedError):
            return self.start(
                profile,
                port=port,
                avd_name=avd_name,
                timeout_s=timeout_s,
                cold_boot=True,
                extra_args=extra_args,
            )

    def stop(self, lease: EmulatorLease) -> None:
        self.adb.run(lease.serial, ("emu", "kill"), timeout_s=3.0)
        try:
            lease.process.wait(timeout=8.0)
            return
        except subprocess.TimeoutExpired:
            lease.process.terminate()
        try:
            lease.process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            lease.process.kill()
            lease.process.wait(timeout=5.0)
