"""Servo bus abstraction.

The five finger actuators are addressed as one *bus* rather than five independent
servos. That choice matters: a grasp is a coordinated multi-finger motion, and
sending one atomic "here are all five targets" command per control cycle keeps
the fingers synchronised and halves the link traffic compared with per-finger
messages. It also gives the firmware a single, unambiguous watchdog to refresh.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from ...core.types import FINGER_COUNT, Finger, HandPose
from ..base import DeviceInfo

__all__ = [
    "ALL_FINGERS_MASK",
    "FingerState",
    "ServoBus",
    "ServoBusState",
    "ServoCalibration",
    "ServoLimits",
]

#: Bit mask selecting every finger (bit 0 = thumb).
ALL_FINGERS_MASK = 0b11111


@dataclass(frozen=True, slots=True)
class ServoLimits:
    """Hard limits enforced by both the host controller and the firmware.

    Values are in normalised closure units (see :mod:`neurogrip.core.types`):
    ``max_velocity = 2.0`` means "fully open to fully closed in 0.5 s".

    These are *safety* limits, not comfort settings. Modes may request slower
    motion, never faster: :meth:`scaled` only ever reduces.
    """

    max_velocity: float = 2.0
    max_acceleration: float = 8.0
    max_current_ma: int = 900
    max_temperature_c: float = 65.0
    #: Ceiling on commanded grip force in ``[0, 1]``; see ``docs/safety.md``.
    max_force: float = 0.85

    def scaled(self, factor: float) -> ServoLimits:
        """Return limits scaled *down* by ``factor`` (clamped to ``<= 1``)."""
        f = max(0.05, min(1.0, factor))
        return ServoLimits(
            max_velocity=self.max_velocity * f,
            max_acceleration=self.max_acceleration * f,
            max_current_ma=self.max_current_ma,
            max_temperature_c=self.max_temperature_c,
            max_force=self.max_force * f,
        )

    def validate(self) -> None:
        """Raise if the limits are physically implausible (called at startup)."""
        from ...core.errors import ConfigurationError

        if self.max_velocity <= 0 or self.max_velocity > 10:
            raise ConfigurationError(f"implausible max_velocity: {self.max_velocity}")
        if self.max_acceleration <= 0 or self.max_acceleration > 200:
            raise ConfigurationError(f"implausible max_acceleration: {self.max_acceleration}")
        if not 0 < self.max_current_ma <= 5000:
            raise ConfigurationError(f"implausible max_current_ma: {self.max_current_ma}")
        if not 0 < self.max_force <= 1:
            raise ConfigurationError(f"max_force must be in (0, 1]: {self.max_force}")


@dataclass(frozen=True, slots=True)
class ServoCalibration:
    """Per-finger mapping from normalised closure to servo pulse width.

    Tendon-driven fingers need individual endpoints: fishing-line routing length
    differs per digit, and a thumb that over-travels will either snap the tendon
    or stall the horn against a hard stop.
    """

    finger: Finger
    min_pulse_us: int = 1000
    max_pulse_us: int = 2000
    inverted: bool = False
    #: Extra travel, in closure units, taken up before the tendon goes taut.
    slack: float = 0.0

    def to_pulse(self, closure: float) -> int:
        """Convert normalised closure to a pulse width in microseconds.

        ``slack`` models the servo rotation consumed before the fishing-line
        tendon goes taut: closure ``0`` maps to the slack offset, not to the
        mechanical endpoint, so the commanded range stays linear in *finger*
        motion rather than in *servo* motion.
        """
        clamped = min(1.0, max(0.0, closure))
        travel = self.slack + clamped * (1.0 - self.slack)
        if self.inverted:
            travel = 1.0 - travel
        return int(round(self.min_pulse_us + travel * (self.max_pulse_us - self.min_pulse_us)))

    def from_pulse(self, pulse_us: int) -> float:
        """Inverse of :meth:`to_pulse`, for reading back firmware calibration."""
        span = self.max_pulse_us - self.min_pulse_us
        if span == 0:
            return 0.0
        travel = (pulse_us - self.min_pulse_us) / span
        if self.inverted:
            travel = 1.0 - travel
        if self.slack >= 1.0:
            return 0.0
        return min(1.0, max(0.0, (travel - self.slack) / (1.0 - self.slack)))


@dataclass(frozen=True, slots=True)
class FingerState:
    """Measured state of a single finger."""

    finger: Finger
    position: float = 0.0
    target: float = 0.0
    velocity: float = 0.0
    current_ma: int = 0
    temperature_c: float = 25.0
    stalled: bool = False
    fault: str = ""

    @property
    def tracking_error(self) -> float:
        return self.target - self.position

    @property
    def in_contact(self) -> bool:
        """Heuristic contact flag: stalled short of the commanded target.

        The definitive contact decision lives in
        :mod:`neurogrip.control.force`, which also considers current history;
        this flag is what the firmware/simulation reports directly.
        """
        return self.stalled and abs(self.tracking_error) > 0.02


@dataclass(frozen=True, slots=True)
class ServoBusState:
    """Snapshot of all five fingers plus bus-level status."""

    fingers: tuple[FingerState, ...]
    timestamp: float = 0.0
    sequence: int = 0
    enabled: bool = False
    homed: bool = False
    estop: bool = False
    moving: bool = False
    bus_voltage_v: float = 0.0
    comms_ok: bool = True
    faults: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if len(self.fingers) != FINGER_COUNT:
            raise ValueError(f"expected {FINGER_COUNT} finger states, got {len(self.fingers)}")

    @property
    def pose(self) -> HandPose:
        """Measured pose."""
        return HandPose.from_iterable(f.position for f in self.fingers)

    @property
    def target_pose(self) -> HandPose:
        """Commanded pose as echoed back by the controller."""
        return HandPose.from_iterable(f.target for f in self.fingers)

    @property
    def total_current_ma(self) -> int:
        return sum(f.current_ma for f in self.fingers)

    @property
    def max_temperature_c(self) -> float:
        return max((f.temperature_c for f in self.fingers), default=0.0)

    @property
    def any_stalled(self) -> bool:
        return any(f.stalled for f in self.fingers)

    def finger(self, finger: Finger) -> FingerState:
        return self.fingers[int(finger)]

    @classmethod
    def idle(cls, timestamp: float = 0.0) -> ServoBusState:
        """A safe all-zero state, used before the first telemetry arrives."""
        return cls(
            fingers=tuple(FingerState(finger=f) for f in Finger),
            timestamp=timestamp,
            comms_ok=False,
        )


@runtime_checkable
class ServoBus(Protocol):
    """Five-channel actuator bus.

    Implementations must be safe to call from a single control thread only.
    :meth:`emergency_stop` is the one exception: it may be called from any
    thread and from a signal handler, and must take effect without waiting for
    the control loop.
    """

    def open(self) -> None: ...

    def close(self) -> None: ...

    @property
    def is_open(self) -> bool: ...

    def info(self) -> DeviceInfo: ...

    def enable(self, mask: int = ALL_FINGERS_MASK) -> None:
        """Energise the selected fingers."""
        ...

    def disable(self, mask: int = ALL_FINGERS_MASK) -> None:
        """De-energise the selected fingers (they go limp; no holding torque)."""
        ...

    def set_limits(self, limits: ServoLimits) -> None:
        """Push velocity/acceleration/current/force ceilings to the controller."""
        ...

    def set_calibration(self, calibration: ServoCalibration) -> None:
        """Persist a per-finger endpoint calibration."""
        ...

    def write_targets(self, pose: HandPose, *, speed_scale: float = 1.0, force: float = 0.6) -> None:
        """Command a target pose. Called once per control cycle."""
        ...

    def read_state(self) -> ServoBusState:
        """Latest telemetry. Must be non-blocking and always return a state."""
        ...

    def home(self) -> None:
        """Run the homing routine that establishes the open-hand reference."""
        ...

    def emergency_stop(self) -> None:
        """Cut drive immediately. Thread-safe; must not raise."""
        ...

    def clear_emergency_stop(self) -> None:
        """Release the e-stop latch. Requires an explicit user acknowledgement."""
        ...
