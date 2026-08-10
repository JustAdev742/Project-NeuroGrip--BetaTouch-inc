"""NeuroGrip Protocol (NGP) v1 — host ⇄ ESP32 motor controller.

This module is the single source of truth for the wire format. The C++ side lives
in ``firmware/esp32_motor_controller/include/ngp_protocol.h`` and mirrors these
definitions byte for byte; ``docs/protocol.md`` documents the contract and
``tests/unit/test_protocol.py`` pins the encodings.

Design notes
------------
* **Fixed-point, not floats.** Finger positions travel as ``uint16`` in units of
  1/10000 of full closure. Integer maths keeps the MCU side cheap and removes any
  float-format ambiguity between the two platforms.
* **The MCU owns the safety timeout.** Every ``SET_TARGETS`` refreshes a firmware
  watchdog. If the host stops talking (crash, USB unplug, kernel stall) the
  firmware ramps the hand to a safe hold and disables drive on its own. Safety
  must not depend on the Linux side being alive.
* **State is pushed, not polled.** The controller streams ``STATE`` at a fixed
  rate; the host only polls on demand for diagnostics. This keeps the round-trip
  out of the control loop's critical path.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import IntEnum, IntFlag

from ..core.errors import ProtocolError
from ..core.types import FINGER_COUNT, Finger, HandPose

__all__ = [
    "POSITION_SCALE",
    "PROTOCOL_VERSION",
    "ErrorCode",
    "EventCode",
    "FingerTelemetry",
    "MessageId",
    "StateMessage",
    "StatusFlags",
    "decode_error",
    "decode_event",
    "decode_pong",
    "decode_state",
    "encode_clear_estop",
    "encode_enable",
    "encode_estop",
    "encode_home",
    "encode_ping",
    "encode_set_calibration",
    "encode_set_force",
    "encode_set_limits",
    "encode_set_targets",
    "encode_set_watchdog",
    "position_from_wire",
    "position_to_wire",
]

PROTOCOL_VERSION = 1

#: Finger closure is transmitted as ``uint16`` in units of 1/POSITION_SCALE.
POSITION_SCALE = 10_000


class MessageId(IntEnum):
    """Message identifiers. Host→MCU are < 0x80, MCU→host are >= 0x80."""

    # Host -> MCU
    PING = 0x01
    SET_TARGETS = 0x02
    SET_LIMITS = 0x03
    ENABLE = 0x04
    DISABLE = 0x05
    ESTOP = 0x06
    CLEAR_ESTOP = 0x07
    REQUEST_STATE = 0x08
    SET_CALIBRATION = 0x09
    HOME = 0x0A
    SET_FORCE = 0x0B
    SET_WATCHDOG = 0x0C
    REBOOT = 0x0D

    # MCU -> Host
    PONG = 0x81
    STATE = 0x82
    EVENT = 0x83
    ERROR = 0x84
    LOG = 0x85


class StatusFlags(IntFlag):
    """Bit flags in the ``STATE`` message's status byte."""

    NONE = 0
    ENABLED = 1 << 0
    MOVING = 1 << 1
    ESTOP = 1 << 2
    HOMED = 1 << 3
    OVERCURRENT = 1 << 4
    OVERTEMP = 1 << 5
    WATCHDOG_TRIPPED = 1 << 6
    UNDERVOLTAGE = 1 << 7


class EventCode(IntEnum):
    """Asynchronous notifications pushed by the firmware."""

    STALL_DETECTED = 1
    TARGET_REACHED = 2
    CONTACT_DETECTED = 3
    HOMING_COMPLETE = 4
    ESTOP_ENGAGED = 5
    ESTOP_RELEASED = 6
    WATCHDOG_TRIP = 7
    THERMAL_THROTTLE = 8


class ErrorCode(IntEnum):
    """Error conditions reported by the firmware."""

    NONE = 0
    UNKNOWN_MESSAGE = 1
    BAD_LENGTH = 2
    BAD_PARAMETER = 3
    NOT_ENABLED = 4
    ESTOP_ACTIVE = 5
    NOT_HOMED = 6
    OVERCURRENT = 7
    OVERTEMP = 8
    UNDERVOLTAGE = 9
    HARDWARE_FAULT = 10


def position_to_wire(value: float) -> int:
    """Convert a normalised closure in ``[0, 1]`` to the wire integer."""
    return max(0, min(POSITION_SCALE, int(round(value * POSITION_SCALE))))


def position_from_wire(value: int) -> float:
    """Convert a wire integer back to normalised closure."""
    return max(0.0, min(1.0, value / POSITION_SCALE))


# ---------------------------------------------------------------------------
# Host -> MCU encoders
# ---------------------------------------------------------------------------

_TARGETS_FMT = "<5HBB"  # 5 positions, speed permille/2.55, flags


def encode_set_targets(pose: HandPose, *, speed_scale: float = 1.0, flags: int = 0) -> bytes:
    """Encode a ``SET_TARGETS`` payload.

    ``speed_scale`` in ``[0, 2]`` scales the firmware's configured maximum
    velocity; Sports Mode uses values above 1.0, precision grasps below.
    """
    speed_byte = max(0, min(255, int(round(speed_scale * 127.5))))
    return struct.pack(
        _TARGETS_FMT,
        *(position_to_wire(v) for v in pose.values),
        speed_byte,
        flags & 0xFF,
    )


def decode_set_targets(payload: bytes) -> tuple[HandPose, float, int]:
    """Decode a ``SET_TARGETS`` payload (used by the firmware emulator and tests)."""
    if len(payload) != struct.calcsize(_TARGETS_FMT):
        raise ProtocolError("SET_TARGETS payload has wrong length", context={"len": len(payload)})
    *positions, speed_byte, flags = struct.unpack(_TARGETS_FMT, payload)
    pose = HandPose.from_iterable(position_from_wire(p) for p in positions)
    return pose, speed_byte / 127.5, flags


_LIMITS_FMT = "<HHHH"


def encode_set_limits(
    *,
    max_velocity: float,
    max_acceleration: float,
    max_current_ma: int,
    max_temperature_c: int,
) -> bytes:
    """Encode ``SET_LIMITS``.

    Velocity is closure-units per second, acceleration closure-units per second
    squared; both are transmitted in 1/1000 units.
    """
    return struct.pack(
        _LIMITS_FMT,
        max(0, min(65535, int(round(max_velocity * 1000)))),
        max(0, min(65535, int(round(max_acceleration * 1000)))),
        max(0, min(65535, int(max_current_ma))),
        max(0, min(65535, int(max_temperature_c))),
    )


def decode_set_limits(payload: bytes) -> tuple[float, float, int, int]:
    """Decode a ``SET_LIMITS`` payload."""
    if len(payload) != struct.calcsize(_LIMITS_FMT):
        raise ProtocolError("SET_LIMITS payload has wrong length", context={"len": len(payload)})
    vel, acc, current, temp = struct.unpack(_LIMITS_FMT, payload)
    return vel / 1000.0, acc / 1000.0, current, temp


def encode_enable(mask: int = 0x1F) -> bytes:
    """Enable drive on the fingers selected by ``mask`` (bit 0 = thumb)."""
    return struct.pack("<B", mask & 0x1F)


def encode_disable(mask: int = 0x1F) -> bytes:
    return struct.pack("<B", mask & 0x1F)


def encode_estop() -> bytes:
    return b""


def encode_clear_estop() -> bytes:
    """Clearing e-stop carries a magic word so a corrupted frame cannot re-arm drive."""
    return struct.pack("<H", 0x5EA1)


def encode_home() -> bytes:
    return b""


def encode_ping(token: int = 0) -> bytes:
    return struct.pack("<I", token & 0xFFFFFFFF)


def encode_set_calibration(
    finger: Finger, *, min_pulse_us: int, max_pulse_us: int, inverted: bool, slack: float = 0.0
) -> bytes:
    """Persist per-finger servo endpoints and tendon slack in the firmware's NVS.

    ``slack`` is quantised to a byte (1/255 of full closure, ~0.4%), which is far
    finer than the mechanism can resolve. It has to travel with the endpoints
    rather than being applied host-side: the firmware owns the closure→pulse
    mapping, so a host-side correction would be applied twice on hardware and
    once in simulation.
    """
    return struct.pack(
        "<BHHBB",
        int(finger),
        int(min_pulse_us),
        int(max_pulse_us),
        1 if inverted else 0,
        max(0, min(255, int(round(slack * 255)))),
    )


def encode_set_watchdog(timeout_ms: int) -> bytes:
    """Set the firmware command-timeout, after which drive is safed."""
    return struct.pack("<H", max(0, min(65535, int(timeout_ms))))


def encode_set_force(mask: int, force: float) -> bytes:
    """Set the grip-force ceiling (0..1) for the selected fingers."""
    return struct.pack("<BB", mask & 0x1F, max(0, min(255, int(round(force * 255)))))


# ---------------------------------------------------------------------------
# MCU -> Host decoders
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FingerTelemetry:
    """Per-finger feedback carried in a ``STATE`` message."""

    finger: Finger
    position: float
    target: float
    current_ma: int
    temperature_c: float

    @property
    def error(self) -> float:
        """Signed tracking error (target minus measured)."""
        return self.target - self.position


@dataclass(frozen=True, slots=True)
class StateMessage:
    """Decoded ``STATE`` message."""

    sequence: int
    flags: StatusFlags
    fingers: tuple[FingerTelemetry, ...]
    bus_voltage_v: float
    uptime_ms: int

    @property
    def positions(self) -> HandPose:
        return HandPose.from_iterable(f.position for f in self.fingers)

    @property
    def targets(self) -> HandPose:
        return HandPose.from_iterable(f.target for f in self.fingers)

    @property
    def estop_active(self) -> bool:
        return bool(self.flags & StatusFlags.ESTOP)

    @property
    def enabled(self) -> bool:
        return bool(self.flags & StatusFlags.ENABLED)

    @property
    def moving(self) -> bool:
        return bool(self.flags & StatusFlags.MOVING)

    @property
    def total_current_ma(self) -> int:
        return sum(f.current_ma for f in self.fingers)


#: sequence(B) flags(B) [pos(H) target(H) current(H) temp(b)] × 5, voltage(H), uptime(I)
_STATE_HEADER_FMT = "<BB"
_STATE_FINGER_FMT = "<HHHb"
_STATE_TAIL_FMT = "<HI"
_STATE_SIZE = (
    struct.calcsize(_STATE_HEADER_FMT)
    + struct.calcsize(_STATE_FINGER_FMT) * FINGER_COUNT
    + struct.calcsize(_STATE_TAIL_FMT)
)


def encode_state(
    *,
    sequence: int,
    flags: StatusFlags,
    fingers: tuple[FingerTelemetry, ...],
    bus_voltage_v: float,
    uptime_ms: int,
) -> bytes:
    """Encode a ``STATE`` message (firmware side; used by the emulator and tests)."""
    if len(fingers) != FINGER_COUNT:
        raise ProtocolError(f"STATE requires {FINGER_COUNT} fingers, got {len(fingers)}")
    out = bytearray(struct.pack(_STATE_HEADER_FMT, sequence & 0xFF, int(flags) & 0xFF))
    for telemetry in fingers:
        out += struct.pack(
            _STATE_FINGER_FMT,
            position_to_wire(telemetry.position),
            position_to_wire(telemetry.target),
            max(0, min(65535, int(telemetry.current_ma))),
            max(-128, min(127, int(round(telemetry.temperature_c)))),
        )
    out += struct.pack(
        _STATE_TAIL_FMT,
        max(0, min(65535, int(round(bus_voltage_v * 1000)))),
        uptime_ms & 0xFFFFFFFF,
    )
    return bytes(out)


def decode_state(payload: bytes) -> StateMessage:
    """Decode a ``STATE`` message."""
    if len(payload) != _STATE_SIZE:
        raise ProtocolError(
            "STATE payload has wrong length",
            context={"expected": _STATE_SIZE, "actual": len(payload)},
        )
    sequence, raw_flags = struct.unpack_from(_STATE_HEADER_FMT, payload, 0)
    offset = struct.calcsize(_STATE_HEADER_FMT)
    stride = struct.calcsize(_STATE_FINGER_FMT)
    fingers = []
    for index in range(FINGER_COUNT):
        pos, target, current, temp = struct.unpack_from(_STATE_FINGER_FMT, payload, offset)
        offset += stride
        fingers.append(
            FingerTelemetry(
                finger=Finger(index),
                position=position_from_wire(pos),
                target=position_from_wire(target),
                current_ma=current,
                temperature_c=float(temp),
            )
        )
    voltage_mv, uptime_ms = struct.unpack_from(_STATE_TAIL_FMT, payload, offset)
    return StateMessage(
        sequence=sequence,
        flags=StatusFlags(raw_flags),
        fingers=tuple(fingers),
        bus_voltage_v=voltage_mv / 1000.0,
        uptime_ms=uptime_ms,
    )


_PONG_FMT = "<IBBBI"


def encode_pong(token: int, version: tuple[int, int, int], uptime_ms: int) -> bytes:
    major, minor, patch = version
    return struct.pack(_PONG_FMT, token & 0xFFFFFFFF, major, minor, patch, uptime_ms & 0xFFFFFFFF)


def decode_pong(payload: bytes) -> tuple[int, tuple[int, int, int], int]:
    """Return ``(token, (major, minor, patch), uptime_ms)``."""
    if len(payload) != struct.calcsize(_PONG_FMT):
        raise ProtocolError("PONG payload has wrong length", context={"len": len(payload)})
    token, major, minor, patch, uptime = struct.unpack(_PONG_FMT, payload)
    return token, (major, minor, patch), uptime


_EVENT_FMT = "<BBH"


def encode_event(code: EventCode, finger: int, detail: int) -> bytes:
    return struct.pack(_EVENT_FMT, int(code), finger & 0xFF, detail & 0xFFFF)


def decode_event(payload: bytes) -> tuple[EventCode, int, int]:
    """Return ``(code, finger_index, detail)``; ``finger_index`` 0xFF means "all"."""
    if len(payload) != struct.calcsize(_EVENT_FMT):
        raise ProtocolError("EVENT payload has wrong length", context={"len": len(payload)})
    code, finger, detail = struct.unpack(_EVENT_FMT, payload)
    try:
        return EventCode(code), finger, detail
    except ValueError as exc:
        raise ProtocolError("unknown event code", context={"code": code}) from exc


_ERROR_FMT = "<BH"


def encode_error(code: ErrorCode, detail: int = 0) -> bytes:
    return struct.pack(_ERROR_FMT, int(code), detail & 0xFFFF)


def decode_error(payload: bytes) -> tuple[ErrorCode, int]:
    if len(payload) != struct.calcsize(_ERROR_FMT):
        raise ProtocolError("ERROR payload has wrong length", context={"len": len(payload)})
    code, detail = struct.unpack(_ERROR_FMT, payload)
    try:
        return ErrorCode(code), detail
    except ValueError as exc:
        raise ProtocolError("unknown error code", context={"code": code}) from exc


def decode_log(payload: bytes) -> tuple[int, str]:
    """Decode a firmware ``LOG`` message into ``(level, text)``."""
    if not payload:
        raise ProtocolError("LOG payload is empty")
    return payload[0], payload[1:].decode("utf-8", errors="replace")
