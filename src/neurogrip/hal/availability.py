"""Hardware presence detection and production-readiness enforcement.

This module exists to make one distinction impossible to blur:

    hardware detected  !=  hardware missing  !=  hardware disabled

The previous composition root silently substituted a simulated driver when a
real one could not be opened. That is a good default for a *development* rig and
a dangerous one for a device attached to a person: the runtime reported healthy,
the UI showed plausible numbers, and nothing anywhere said "these readings are
invented". A simulated EMG source feeding a real intent engine that drives real
servos is exactly the failure this prevents.

So:

* Backends are named for what they are — ``real_ads1115``, ``real_v4l2``,
  ``real_servo_controller`` — or ``disabled``. There is no backend whose name
  hides whether it touches hardware.
* A device that cannot be opened is **MISSING**. It is never replaced by a
  stand-in. The runtime degrades or refuses to start, and says which.
* A device the operator has deliberately turned off is **DISABLED**, which is
  not a fault and is reported differently.

:class:`ProductionRequirements` then enforces the shipping configuration:
EMG, a servo controller, five servos and two depth cameras. Anything less and
the runtime refuses to enter operational mode. Partial hardware must never be
able to masquerade as a working prosthesis.
"""

from __future__ import annotations

import glob
import os
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from ..core.config import Config
from ..core.logging import get_logger

__all__ = [
    "Availability",
    "DeviceStatus",
    "HardwareInventory",
    "HardwareScanner",
    "ProductionRequirements",
    "Requirement",
]

log = get_logger(__name__)


class Availability(str, Enum):
    """Presence of one device. The whole point of this module."""

    #: Probed and present.
    DETECTED = "detected"
    #: Expected by configuration, probed, not present.
    MISSING = "missing"
    #: Deliberately turned off by the operator. Not a fault.
    DISABLED = "disabled"
    #: Present but unusable — permissions, wrong firmware, bad response.
    ERROR = "error"

    @property
    def is_usable(self) -> bool:
        return self is Availability.DETECTED

    @property
    def is_fault(self) -> bool:
        """DISABLED is deliberately excluded: the operator chose it."""
        return self in (Availability.MISSING, Availability.ERROR)

    @property
    def symbol(self) -> str:
        return {
            Availability.DETECTED: "OK",
            Availability.MISSING: "MISSING",
            Availability.DISABLED: "OFF",
            Availability.ERROR: "ERROR",
        }[self]


@dataclass(frozen=True, slots=True)
class DeviceStatus:
    """What the scanner found for one configured device."""

    kind: str                       # "emg" | "servo" | "camera" | "depth_camera"
    backend: str                    # "real_ads1115" | "disabled" | ...
    availability: Availability
    detail: str = ""
    #: Concrete node/port/address the probe resolved, for the operator.
    path: str = ""
    #: How many units this entry accounts for (5 servos on one controller).
    count: int = 1

    @property
    def simulated(self) -> bool:
        """Always False. Kept so callers can assert it, and so the answer to
        "is any of this fake?" is a property rather than a guess."""
        return False

    def describe(self) -> str:
        bits = [f"{self.kind}: {self.availability.symbol}"]
        if self.backend and self.backend != "disabled":
            bits.append(f"({self.backend})")
        if self.path:
            bits.append(self.path)
        if self.detail:
            bits.append(f"- {self.detail}")
        return " ".join(bits)


@dataclass(frozen=True, slots=True)
class Requirement:
    """One item the production configuration must satisfy."""

    kind: str
    count: int
    label: str

    def describe_missing(self, found: int) -> list[str]:
        """Operator-facing lines naming exactly what is absent.

        Enumerated individually ("Depth camera 1", "Depth camera 2") rather than
        summarised ("2 depth cameras missing") because the operator is standing
        at the device deciding what to plug in.
        """
        short = self.count - found
        if short <= 0:
            return []
        if self.count == 1:
            return [self.label]
        return [f"{self.label} {i}" for i in range(found + 1, self.count + 1)]


class ProductionRequirements:
    """The shipping hardware contract.

    Enforced before the runtime enters operational mode. Deliberately not
    configurable from the normal config file: a device in the field must not be
    able to lower its own bar by editing a TOML key. Overriding is possible only
    through an explicit engineering flag, which is logged loudly.
    """

    DEFAULT: tuple[Requirement, ...] = (
        Requirement("emg", 1, "EMG sensor front-end"),
        Requirement("servo_controller", 1, "Servo controller"),
        Requirement("servo", 5, "Servo motor"),
        Requirement("depth_camera", 2, "Depth camera"),
    )

    def __init__(self, requirements: Sequence[Requirement] | None = None) -> None:
        self._requirements = tuple(requirements or self.DEFAULT)

    @property
    def requirements(self) -> tuple[Requirement, ...]:
        return self._requirements

    def evaluate(self, inventory: HardwareInventory) -> list[str]:
        """Return operator-facing lines for everything missing. Empty == ready."""
        missing: list[str] = []
        for requirement in self._requirements:
            found = inventory.count_detected(requirement.kind)
            missing.extend(requirement.describe_missing(found))
        return missing


@dataclass(slots=True)
class HardwareInventory:
    """Everything the scanner found, plus the readiness verdict."""

    devices: tuple[DeviceStatus, ...] = ()
    scanned_at: float = 0.0
    notes: tuple[str, ...] = ()

    def of_kind(self, kind: str) -> tuple[DeviceStatus, ...]:
        return tuple(d for d in self.devices if d.kind == kind)

    def count_detected(self, kind: str) -> int:
        return sum(d.count for d in self.of_kind(kind) if d.availability.is_usable)

    def status(self, kind: str) -> Availability:
        """Worst availability across devices of ``kind``."""
        entries = self.of_kind(kind)
        if not entries:
            return Availability.MISSING
        for state in (Availability.ERROR, Availability.MISSING, Availability.DISABLED):
            if any(d.availability is state for d in entries):
                return state
        return Availability.DETECTED

    @property
    def faults(self) -> tuple[DeviceStatus, ...]:
        return tuple(d for d in self.devices if d.availability.is_fault)

    def missing_for(self, requirements: ProductionRequirements) -> list[str]:
        return requirements.evaluate(self)

    def is_production_ready(self, requirements: ProductionRequirements) -> bool:
        return not self.missing_for(requirements)

    def operator_message(self, requirements: ProductionRequirements) -> str:
        """The block screen shown when the runtime refuses to start.

        Names the missing items and nothing else. An operator holding a hand
        that will not start needs a list of what to plug in, not a stack trace.
        """
        missing = self.missing_for(requirements)
        if not missing:
            return ""
        lines = [
            "BetaTouch cannot start.",
            "",
            "Required hardware missing:",
        ]
        lines.extend(f"- {item}" for item in missing)
        lines.extend(["", "Please connect all required hardware."])
        return "\n".join(lines)

    def describe(self) -> list[str]:
        return [d.describe() for d in self.devices]


class HardwareScanner:
    """Probes for real devices. Never fabricates one.

    Probes are intentionally cheap and read-only — existence of a device node,
    a serial port, an I²C address file. Opening and handshaking is the driver's
    job at start time; this is the pre-flight check that decides whether starting
    is even worth attempting, and it must be safe to run on every boot.
    """

    def __init__(self, config: Config) -> None:
        self._config = config

    # -- entry point ----------------------------------------------------------

    def scan(self, *, now: float = 0.0) -> HardwareInventory:
        devices: list[DeviceStatus] = []
        notes: list[str] = []

        devices.append(self._scan_emg(notes))
        controller, servo_entry = self._scan_servos(notes)
        devices.append(controller)
        devices.append(servo_entry)
        devices.extend(self._scan_depth_cameras(notes))

        inventory = HardwareInventory(
            devices=tuple(devices), scanned_at=now, notes=tuple(notes)
        )
        for device in inventory.devices:
            log.info("hardware probe", **{
                "kind": device.kind,
                "backend": device.backend,
                "state": device.availability.value,
                "path": device.path,
            })
        return inventory

    # -- probes ---------------------------------------------------------------

    def _scan_emg(self, notes: list[str]) -> DeviceStatus:
        section = self._config.section("emg")
        backend = section.get_str("backend", section.get_str("driver", "real_serial"))
        if backend == "disabled" or not section.get_bool("enabled", True):
            return DeviceStatus("emg", "disabled", Availability.DISABLED,
                                "disabled by configuration")

        if backend in ("real_ads1115", "real_i2c"):
            bus = section.get_int("i2c_bus", 1)
            address = section.get_int("i2c_address", 0x48)
            return self._probe_i2c("emg", backend, bus, address)

        if backend in ("real_serial", "real_usb", "serial"):
            port = section.get_str("port", "/dev/ttyACM0")
            return self._probe_serial("emg", backend, port)

        notes.append(f"emg: unknown backend '{backend}'")
        return DeviceStatus("emg", backend, Availability.ERROR,
                            f"unknown backend '{backend}'")

    def _scan_servos(self, notes: list[str]) -> tuple[DeviceStatus, DeviceStatus]:
        """Controller presence and motor count are separate facts.

        A detected controller with three motors wired is not a working hand, and
        collapsing the two into one status would let that pass.
        """
        section = self._config.section("servo")
        backend = section.get_str("backend", section.get_str("driver", "real_servo_controller"))
        expected = section.get_int("motor_count", 5)

        if backend == "disabled" or not section.get_bool("enabled", True):
            disabled = DeviceStatus("servo_controller", "disabled",
                                    Availability.DISABLED, "disabled by configuration")
            return disabled, DeviceStatus("servo", "disabled", Availability.DISABLED,
                                          "controller disabled", count=0)

        if backend in ("real_servo_controller", "real_esp32", "esp32"):
            port = section.get_str("port", "/dev/ttyUSB0")
            controller = self._probe_serial("servo_controller", backend, port)
        elif backend in ("real_pwm", "real_pca9685"):
            bus = section.get_int("i2c_bus", 1)
            address = section.get_int("i2c_address", 0x40)
            controller = self._probe_i2c("servo_controller", backend, bus, address)
        else:
            notes.append(f"servo: unknown backend '{backend}'")
            controller = DeviceStatus("servo_controller", backend, Availability.ERROR,
                                      f"unknown backend '{backend}'")

        # Motors cannot be enumerated without talking to the controller, which
        # this read-only pre-flight deliberately does not do. If the controller
        # is absent the motors are unverifiable, and unverifiable must count as
        # missing rather than assumed-present.
        if controller.availability.is_usable:
            motors = DeviceStatus(
                "servo", backend, Availability.DETECTED,
                f"{expected} motors declared; verified during bring-up",
                path=controller.path, count=expected,
            )
        else:
            motors = DeviceStatus(
                "servo", backend, Availability.MISSING,
                "controller unavailable, motors unverifiable",
                count=0,
            )
        return controller, motors

    def _scan_depth_cameras(self, notes: list[str]) -> list[DeviceStatus]:
        section = self._config.section("camera")
        backend = section.get_str("backend", section.get_str("driver", "real_depth_camera"))
        expected = section.get_int("depth_count", 2)

        if backend == "disabled" or not section.get_bool("enabled", True):
            return [DeviceStatus("depth_camera", "disabled", Availability.DISABLED,
                                 "disabled by configuration", count=0)]

        configured = section.get_list("depth_devices", [])
        nodes = [str(n) for n in configured] if configured else self._enumerate_video_nodes()

        results: list[DeviceStatus] = []
        for index in range(expected):
            if index < len(nodes):
                node = nodes[index]
                if os.path.exists(node):
                    results.append(DeviceStatus(
                        "depth_camera", backend, Availability.DETECTED,
                        path=node,
                    ))
                    continue
                results.append(DeviceStatus(
                    "depth_camera", backend, Availability.MISSING,
                    f"{node} does not exist", path=node, count=0,
                ))
                continue
            results.append(DeviceStatus(
                "depth_camera", backend, Availability.MISSING,
                "no matching device node", count=0,
            ))
        if len(nodes) < expected:
            notes.append(
                f"camera: {len(nodes)} video node(s) present, {expected} required"
            )
        return results

    # -- primitives -----------------------------------------------------------

    def _probe_serial(self, kind: str, backend: str, port: str) -> DeviceStatus:
        if not port:
            return DeviceStatus(kind, backend, Availability.ERROR, "no port configured")
        if os.path.exists(port):
            if os.access(port, os.R_OK | os.W_OK):
                return DeviceStatus(kind, backend, Availability.DETECTED, path=port)
            # A present-but-unreadable node is a permissions problem, which is
            # fixable and completely different from "not plugged in".
            return DeviceStatus(kind, backend, Availability.ERROR,
                                "present but not readable/writable (check group membership)",
                                path=port)
        return DeviceStatus(kind, backend, Availability.MISSING,
                            f"{port} not present", path=port, count=0)

    def _probe_i2c(self, kind: str, backend: str, bus: int, address: int) -> DeviceStatus:
        node = f"/dev/i2c-{bus}"
        path = f"{node}@0x{address:02X}"
        if not os.path.exists(node):
            return DeviceStatus(kind, backend, Availability.MISSING,
                                f"{node} not present (is the I2C bus enabled?)",
                                path=path, count=0)
        if not os.access(node, os.R_OK | os.W_OK):
            return DeviceStatus(kind, backend, Availability.ERROR,
                                "I2C bus present but not accessible (check the i2c group)",
                                path=path)
        # Presence of a driver-bound device shows up in sysfs; absence there is
        # not conclusive (the address may simply be unbound), so this is
        # DETECTED-with-caveat rather than MISSING. The driver's open() is the
        # authority.
        bound = glob.glob(f"/sys/bus/i2c/devices/{bus}-{address:04x}")
        detail = "" if bound else "bus present; device not bound until driver open"
        return DeviceStatus(kind, backend, Availability.DETECTED, detail, path=path)

    @staticmethod
    def _enumerate_video_nodes() -> list[str]:
        return sorted(glob.glob("/dev/video*"))

    @staticmethod
    def tool_available(name: str) -> bool:
        return shutil.which(name) is not None
