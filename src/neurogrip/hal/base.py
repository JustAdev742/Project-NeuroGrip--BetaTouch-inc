"""Common device abstractions.

Every physical peripheral is reached through a narrow ``Protocol`` defined in this
package. The rule the HAL exists to enforce: **no module outside ``neurogrip.hal``
may import a vendor library, open a file descriptor, or know a pin number.**

That is what makes "swap the ESP32 for a CAN servo driver" or "swap MyoWare for a
Delsys array" a change to one file plus a config line, rather than a rewrite.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

__all__ = ["Device", "DeviceCapability", "DeviceInfo", "DeviceKind"]


class DeviceKind(str, Enum):
    """Category of a device, used for diagnostics grouping and UI icons."""

    SERVO_BUS = "servo_bus"
    EMG = "emg"
    CAMERA = "camera"
    DISPLAY = "display"
    POWER = "power"
    SYSTEM = "system"
    TRANSPORT = "transport"


class DeviceCapability(str, Enum):
    """Optional features a concrete driver may or may not provide.

    Callers query capabilities instead of type-checking the driver, so a new
    backend that happens to support current sensing needs no ``isinstance``
    special-casing anywhere in the control stack.
    """

    #: Servo bus reports per-finger motor current (enables adaptive grip force).
    CURRENT_SENSING = "current_sensing"
    #: Servo bus reports true position feedback rather than commanded position.
    POSITION_FEEDBACK = "position_feedback"
    #: Servo bus reports motor/driver temperature.
    TEMPERATURE = "temperature"
    #: Device can perform a homing sequence against hard stops.
    HOMING = "homing"
    #: Camera provides per-pixel depth alongside colour.
    DEPTH = "depth"
    #: EMG source provides a hardware-filtered envelope in addition to raw samples.
    HARDWARE_ENVELOPE = "hardware_envelope"
    #: Device is simulated; the UI marks data from it as non-physical.
    SIMULATED = "simulated"


@dataclass(frozen=True, slots=True)
class DeviceInfo:
    """Identity and capabilities of a concrete device instance."""

    name: str
    kind: DeviceKind
    driver: str
    #: Free-form connection description, e.g. ``/dev/ttyUSB0@921600`` or ``sim``.
    connection: str = ""
    firmware_version: str = ""
    serial_number: str = ""
    capabilities: frozenset[DeviceCapability] = field(default_factory=frozenset)
    extra: dict[str, Any] = field(default_factory=dict)

    def supports(self, capability: DeviceCapability) -> bool:
        return capability in self.capabilities

    @property
    def is_simulated(self) -> bool:
        return DeviceCapability.SIMULATED in self.capabilities

    def __str__(self) -> str:  # pragma: no cover - display helper
        where = f" @ {self.connection}" if self.connection else ""
        return f"{self.name} ({self.driver}{where})"


@runtime_checkable
class Device(Protocol):
    """Anything with an open/close lifecycle and an identity."""

    def open(self) -> None:
        """Acquire the device. Raises :class:`~neurogrip.core.errors.DeviceError`."""
        ...

    def close(self) -> None:
        """Release the device. Must be idempotent and must not raise."""
        ...

    @property
    def is_open(self) -> bool:
        ...

    def info(self) -> DeviceInfo:
        ...
