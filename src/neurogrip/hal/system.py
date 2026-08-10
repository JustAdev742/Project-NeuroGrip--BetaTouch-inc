"""Host platform probes: CPU, memory, temperature, storage, connectivity, battery.

Implemented against ``/proc`` and ``/sys`` directly rather than via ``psutil``, so
the diagnostics screen works on a freshly flashed SBC with nothing installed.
Every probe degrades to a "not available" result instead of raising: a missing
thermal zone must never take down the UI.
"""

from __future__ import annotations

import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..core.types import clamp
from .base import DeviceCapability, DeviceInfo, DeviceKind

__all__ = [
    "BatteryState",
    "ConnectivityProbe",
    "ConnectivityState",
    "PowerSource",
    "SimulatedPowerSource",
    "SysfsPowerSource",
    "SystemProbe",
    "SystemStats",
]


@dataclass(frozen=True, slots=True)
class SystemStats:
    """Host resource snapshot."""

    cpu_percent: float = 0.0
    cpu_temperature_c: float = 0.0
    memory_used_mb: float = 0.0
    memory_total_mb: float = 0.0
    disk_free_mb: float = 0.0
    disk_total_mb: float = 0.0
    uptime_s: float = 0.0
    load_average: tuple[float, float, float] = (0.0, 0.0, 0.0)
    process_rss_mb: float = 0.0
    available: bool = True

    @property
    def memory_percent(self) -> float:
        return 100.0 * self.memory_used_mb / self.memory_total_mb if self.memory_total_mb else 0.0

    @property
    def disk_percent_free(self) -> float:
        return 100.0 * self.disk_free_mb / self.disk_total_mb if self.disk_total_mb else 0.0


class SystemProbe:
    """Samples host resource usage.

    CPU utilisation needs two samples to compute a delta, so the first call
    returns ``0.0``; the diagnostics service polls at 2 Hz, which makes the
    warm-up invisible.
    """

    THERMAL_PATHS = (
        "/sys/class/thermal/thermal_zone0/temp",
        "/sys/devices/virtual/thermal/thermal_zone0/temp",
    )

    def __init__(self, *, disk_path: str = "/") -> None:
        self._disk_path = disk_path
        self._last_cpu: tuple[int, int] | None = None
        self._linux = os.path.exists("/proc/stat")

    def sample(self) -> SystemStats:
        if not self._linux:
            return SystemStats(available=False)
        return SystemStats(
            cpu_percent=self._cpu_percent(),
            cpu_temperature_c=self._temperature(),
            memory_used_mb=self._memory()[0],
            memory_total_mb=self._memory()[1],
            disk_free_mb=self._disk()[0],
            disk_total_mb=self._disk()[1],
            uptime_s=self._uptime(),
            load_average=self._load_average(),
            process_rss_mb=self._process_rss(),
        )

    def _cpu_percent(self) -> float:
        try:
            with open("/proc/stat", encoding="ascii") as handle:
                fields = handle.readline().split()
        except OSError:
            return 0.0
        if len(fields) < 5 or fields[0] != "cpu":
            return 0.0
        values = [int(v) for v in fields[1:8]]
        idle = values[3] + (values[4] if len(values) > 4 else 0)
        total = sum(values)
        if self._last_cpu is None:
            self._last_cpu = (idle, total)
            return 0.0
        last_idle, last_total = self._last_cpu
        self._last_cpu = (idle, total)
        delta_total = total - last_total
        delta_idle = idle - last_idle
        if delta_total <= 0:
            return 0.0
        return clamp(100.0 * (1.0 - delta_idle / delta_total), 0.0, 100.0)

    def _temperature(self) -> float:
        for path in self.THERMAL_PATHS:
            try:
                with open(path, encoding="ascii") as handle:
                    raw = int(handle.read().strip())
                return raw / 1000.0 if raw > 1000 else float(raw)
            except (OSError, ValueError):
                continue
        return 0.0

    def _memory(self) -> tuple[float, float]:
        try:
            values: dict[str, float] = {}
            with open("/proc/meminfo", encoding="ascii") as handle:
                for line in handle:
                    key, _, rest = line.partition(":")
                    parts = rest.split()
                    if parts:
                        values[key] = float(parts[0]) / 1024.0  # kB -> MB
        except OSError:
            return (0.0, 0.0)
        total = values.get("MemTotal", 0.0)
        available = values.get("MemAvailable", values.get("MemFree", 0.0))
        return (max(0.0, total - available), total)

    def _disk(self) -> tuple[float, float]:
        try:
            usage = shutil.disk_usage(self._disk_path)
        except OSError:
            return (0.0, 0.0)
        return (usage.free / 1e6, usage.total / 1e6)

    def _uptime(self) -> float:
        try:
            with open("/proc/uptime", encoding="ascii") as handle:
                return float(handle.read().split()[0])
        except (OSError, ValueError, IndexError):
            return 0.0

    def _load_average(self) -> tuple[float, float, float]:
        try:
            one, five, fifteen = os.getloadavg()
            return (one, five, fifteen)
        except (OSError, AttributeError):
            return (0.0, 0.0, 0.0)

    def _process_rss(self) -> float:
        try:
            with open("/proc/self/statm", encoding="ascii") as handle:
                pages = int(handle.read().split()[1])
            return pages * (os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096) / 1e6
        except (OSError, ValueError, IndexError):
            return 0.0

    def info(self) -> DeviceInfo:
        return DeviceInfo(
            name="system",
            kind=DeviceKind.SYSTEM,
            driver="procfs" if self._linux else "unavailable",
            connection=os.uname().nodename if hasattr(os, "uname") else "",
        )


@dataclass(frozen=True, slots=True)
class BatteryState:
    """Battery telemetry for the dashboard and the low-power safety rule."""

    percentage: float = 100.0
    voltage_v: float = 8.4
    current_ma: float = 0.0
    charging: bool = False
    temperature_c: float = 25.0
    time_remaining_s: float | None = None
    present: bool = True

    @property
    def is_critical(self) -> bool:
        """Below this, the safety layer forces a controlled shutdown."""
        return self.present and not self.charging and self.percentage <= 5.0

    @property
    def is_low(self) -> bool:
        """Below this, assistive features are disabled to conserve charge."""
        return self.present and not self.charging and self.percentage <= 15.0


class PowerSource:
    """Base class so both implementations share the derived-property logic."""

    def read(self) -> BatteryState:  # pragma: no cover - abstract
        raise NotImplementedError

    def info(self) -> DeviceInfo:  # pragma: no cover - abstract
        raise NotImplementedError


class SysfsPowerSource(PowerSource):
    """Reads a Linux ``power_supply`` node (a fuel gauge such as the MAX17048)."""

    def __init__(self, node: str = "/sys/class/power_supply/BAT0") -> None:
        self._node = Path(node)

    def read(self) -> BatteryState:
        if not self._node.exists():
            return BatteryState(present=False, percentage=0.0)
        return BatteryState(
            percentage=self._read_float("capacity", 0.0),
            voltage_v=self._read_float("voltage_now", 0.0) / 1e6,
            current_ma=self._read_float("current_now", 0.0) / 1000.0,
            charging=self._read_text("status").lower() in ("charging", "full"),
            temperature_c=self._read_float("temp", 250.0) / 10.0,
            present=True,
        )

    def _read_text(self, name: str) -> str:
        try:
            return (self._node / name).read_text(encoding="ascii").strip()
        except OSError:
            return ""

    def _read_float(self, name: str, default: float) -> float:
        try:
            return float(self._read_text(name))
        except ValueError:
            return default

    def info(self) -> DeviceInfo:
        return DeviceInfo(
            name="battery", kind=DeviceKind.POWER, driver="sysfs", connection=str(self._node)
        )


class SimulatedPowerSource(PowerSource):
    """Battery model that discharges over time, for demos and safety tests."""

    def __init__(
        self,
        *,
        capacity_mah: float = 2600.0,
        start_percentage: float = 92.0,
        base_draw_ma: float = 450.0,
        clock=None,
    ) -> None:
        from ..core.clock import RealClock

        self._clock = clock or RealClock()
        self._capacity = capacity_mah
        self._charge_mah = capacity_mah * start_percentage / 100.0
        self._base_draw = base_draw_ma
        self._extra_draw = 0.0
        self._charging = False
        self._last = self._clock.monotonic()

    def set_load(self, milliamps: float) -> None:
        """Add actuator load on top of the electronics baseline."""
        self._extra_draw = max(0.0, milliamps)

    def set_charging(self, charging: bool) -> None:
        self._charging = charging

    def read(self) -> BatteryState:
        now = self._clock.monotonic()
        dt_hours = max(0.0, now - self._last) / 3600.0
        self._last = now
        draw = self._base_draw + self._extra_draw
        delta = (-draw if not self._charging else 1200.0) * dt_hours
        self._charge_mah = max(0.0, min(self._capacity, self._charge_mah + delta))
        percentage = 100.0 * self._charge_mah / self._capacity
        remaining = (self._charge_mah / draw * 3600.0) if draw > 1 and not self._charging else None
        # Two 18650 cells in series: 6.0 V empty, 8.4 V full.
        return BatteryState(
            percentage=percentage,
            voltage_v=6.0 + 2.4 * (percentage / 100.0),
            current_ma=draw if not self._charging else -1200.0,
            charging=self._charging,
            temperature_c=26.0 + 0.004 * draw,
            time_remaining_s=remaining,
        )

    def info(self) -> DeviceInfo:
        return DeviceInfo(
            name="battery",
            kind=DeviceKind.POWER,
            driver="simulated",
            connection="sim",
            capabilities=frozenset({DeviceCapability.SIMULATED}),
        )


@dataclass(frozen=True, slots=True)
class ConnectivityState:
    """Wi-Fi and Bluetooth status shown in the dashboard status bar."""

    wifi_connected: bool = False
    wifi_ssid: str = ""
    wifi_signal_percent: float = 0.0
    bluetooth_available: bool = False
    bluetooth_connected: bool = False
    bluetooth_devices: tuple[str, ...] = field(default_factory=tuple)
    hostname: str = ""
    ip_address: str = ""


class ConnectivityProbe:
    """Best-effort network/Bluetooth status.

    Purely informational: nothing in the control path depends on it, so every
    failure path simply yields "not connected".
    """

    def __init__(self, *, simulated: bool = False) -> None:
        self._simulated = simulated

    def sample(self) -> ConnectivityState:
        if self._simulated:
            return ConnectivityState(
                wifi_connected=True,
                wifi_ssid="NeuroGrip-Lab",
                wifi_signal_percent=78.0,
                bluetooth_available=True,
                bluetooth_connected=True,
                bluetooth_devices=("Companion Phone",),
                hostname="neurogrip-sim",
                ip_address="10.0.0.42",
            )

        signal, connected = self._wifi()
        return ConnectivityState(
            wifi_connected=connected,
            wifi_ssid=self._ssid(),
            wifi_signal_percent=signal,
            bluetooth_available=Path("/sys/class/bluetooth").exists(),
            bluetooth_connected=self._bluetooth_connected(),
            hostname=os.uname().nodename if hasattr(os, "uname") else "",
            ip_address=self._ip_address(),
        )

    def _wifi(self) -> tuple[float, bool]:
        try:
            with open("/proc/net/wireless", encoding="ascii") as handle:
                lines = handle.readlines()
        except OSError:
            return (0.0, False)
        for line in lines[2:]:
            parts = line.split()
            if len(parts) >= 4:
                try:
                    # Column 3 is link quality out of 70 on most drivers.
                    quality = float(parts[2].rstrip("."))
                    return (clamp(quality / 70.0 * 100.0, 0.0, 100.0), quality > 0)
                except ValueError:
                    continue
        return (0.0, False)

    def _ssid(self) -> str:
        for path in Path("/sys/class/net").glob("*/wireless"):
            iface = path.parent.name
            candidate = Path(f"/run/wpa_supplicant/{iface}")
            if candidate.exists():
                return iface
        return ""

    def _bluetooth_connected(self) -> bool:
        root = Path("/sys/class/bluetooth")
        if not root.exists():
            return False
        return any(root.iterdir())

    def _ip_address(self) -> str:
        import socket

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.settimeout(0.05)
            try:
                sock.connect(("10.255.255.255", 1))
                return str(sock.getsockname()[0])
            finally:
                sock.close()
        except OSError:
            return ""


def boot_time() -> float:  # pragma: no cover - trivial
    """Unix timestamp at which this host booted, for the system-information screen."""
    try:
        with open("/proc/uptime", encoding="ascii") as handle:
            return time.time() - float(handle.read().split()[0])
    except (OSError, ValueError, IndexError):
        return time.time()
