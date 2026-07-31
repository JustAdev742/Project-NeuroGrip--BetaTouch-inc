"""ESP32 motor-controller driver.

Speaks NGP v1 (:mod:`neurogrip.hal.protocol`) over any
:class:`~neurogrip.hal.transport.base.Transport`. Responsibilities:

* frame encode/decode and sequence tracking;
* caching the most recent :class:`~.base.ServoBusState` so ``read_state()`` is
  non-blocking for the control loop;
* declaring the link *stale* when telemetry stops arriving — this is what turns a
  yanked USB cable into "fall back to manual" rather than a hand frozen mid-grasp;
* surfacing firmware events (stall, contact, thermal throttle) to callers.

Note the division of responsibility with the firmware: the host commands *poses*;
the firmware owns pulse generation, its own watchdog, and the final current
limit. If this process dies, the hand still safes itself.
"""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Callable

from ...core.clock import Clock, RealClock
from ...core.errors import CommunicationError, DeviceNotAvailableError
from ...core.logging import get_logger
from ...core.types import Finger, HandPose
from ..base import DeviceCapability, DeviceInfo, DeviceKind
from ..protocol import (
    ErrorCode,
    EventCode,
    MessageId,
    StatusFlags,
    decode_error,
    decode_event,
    decode_log,
    decode_pong,
    decode_state,
    encode_clear_estop,
    encode_disable,
    encode_enable,
    encode_estop,
    encode_home,
    encode_ping,
    encode_set_calibration,
    encode_set_force,
    encode_set_limits,
    encode_set_targets,
    encode_set_watchdog,
)
from ..transport.base import Transport
from ..transport.framing import FrameParser, encode_frame
from .base import (
    ALL_FINGERS_MASK,
    FingerState,
    ServoBusState,
    ServoCalibration,
    ServoLimits,
)

__all__ = ["Esp32ServoBus"]

log = get_logger(__name__)


class Esp32ServoBus:
    """Host-side driver for the ESP32 tendon controller."""

    def __init__(
        self,
        transport: Transport,
        clock: Clock | None = None,
        *,
        state_timeout: float = 0.25,
        watchdog_ms: int = 300,
        limits: ServoLimits | None = None,
    ) -> None:
        self._transport = transport
        self._clock = clock or RealClock()
        self._parser = FrameParser()
        #: Telemetry older than this marks the link stale (see :meth:`read_state`).
        self._state_timeout = state_timeout
        self._watchdog_ms = watchdog_ms
        self._limits = limits or ServoLimits()
        self._sequence = 0
        self._lock = threading.RLock()
        self._state = ServoBusState.idle()
        self._last_state_at = 0.0
        self._firmware_version = ""
        self._estop_latched = False
        self._velocities: dict[Finger, float] = dict.fromkeys(Finger, 0.0)
        self._last_positions: dict[Finger, float] = dict.fromkeys(Finger, 0.0)
        self._events: deque[tuple[EventCode, int, int]] = deque(maxlen=64)
        self._errors: deque[tuple[ErrorCode, int]] = deque(maxlen=64)
        #: Optional sink for firmware events, wired to the event bus by the service.
        self.on_event: Callable[[EventCode, int, int], None] | None = None
        self.on_error: Callable[[ErrorCode, int], None] | None = None

    # -- lifecycle ------------------------------------------------------------

    def open(self) -> None:
        self._transport.open()
        self._parser.reset()
        with self._lock:
            self._last_state_at = 0.0
            self._state = ServoBusState.idle(self._clock.monotonic())
        self._send(MessageId.SET_WATCHDOG, encode_set_watchdog(self._watchdog_ms))
        self.set_limits(self._limits)
        self._send(MessageId.PING, encode_ping(0xC0FFEE))
        log.info("motor controller link opened", transport=str(self._transport.info()))

    def close(self) -> None:
        try:
            if self._transport.is_open:
                self._send(MessageId.DISABLE, encode_disable(ALL_FINGERS_MASK))
        except Exception:
            log.warning("could not disable drive during shutdown")
        finally:
            self._transport.close()

    @property
    def is_open(self) -> bool:
        return self._transport.is_open

    def info(self) -> DeviceInfo:
        return DeviceInfo(
            name="servo-bus",
            kind=DeviceKind.SERVO_BUS,
            driver="esp32-ngp1",
            connection=self._transport.info().connection,
            firmware_version=self._firmware_version,
            capabilities=frozenset(
                {
                    DeviceCapability.CURRENT_SENSING,
                    DeviceCapability.POSITION_FEEDBACK,
                    DeviceCapability.TEMPERATURE,
                    DeviceCapability.HOMING,
                }
            ),
            extra=self._parser.stats(),
        )

    # -- commands -------------------------------------------------------------

    def enable(self, mask: int = ALL_FINGERS_MASK) -> None:
        if self._estop_latched:
            raise CommunicationError("refusing to enable drive while e-stop is latched")
        self._send(MessageId.ENABLE, encode_enable(mask))

    def disable(self, mask: int = ALL_FINGERS_MASK) -> None:
        self._send(MessageId.DISABLE, encode_disable(mask))

    def set_limits(self, limits: ServoLimits) -> None:
        limits.validate()
        with self._lock:
            self._limits = limits
        self._send(
            MessageId.SET_LIMITS,
            encode_set_limits(
                max_velocity=limits.max_velocity,
                max_acceleration=limits.max_acceleration,
                max_current_ma=limits.max_current_ma,
                max_temperature_c=int(limits.max_temperature_c),
            ),
        )

    def set_calibration(self, calibration: ServoCalibration) -> None:
        self._send(
            MessageId.SET_CALIBRATION,
            encode_set_calibration(
                calibration.finger,
                min_pulse_us=calibration.min_pulse_us,
                max_pulse_us=calibration.max_pulse_us,
                inverted=calibration.inverted,
            ),
        )

    def write_targets(self, pose: HandPose, *, speed_scale: float = 1.0, force: float = 0.6) -> None:
        if self._estop_latched:
            return
        capped = min(force, self._limits.max_force)
        self._send(MessageId.SET_FORCE, encode_set_force(ALL_FINGERS_MASK, capped))
        self._send(MessageId.SET_TARGETS, encode_set_targets(pose, speed_scale=speed_scale))

    def home(self) -> None:
        self._send(MessageId.HOME, encode_home())

    def emergency_stop(self) -> None:
        """Latch e-stop locally *and* command the firmware.

        The local latch is set first and unconditionally: if the link is down,
        the host must still refuse to issue further motion commands.
        """
        self._estop_latched = True
        try:
            self._send(MessageId.ESTOP, encode_estop())
        except Exception:
            log.error("e-stop frame could not be sent; local latch engaged")

    def clear_emergency_stop(self) -> None:
        self._estop_latched = False
        self._send(MessageId.CLEAR_ESTOP, encode_clear_estop())

    def ping(self, token: int = 1) -> None:
        self._send(MessageId.PING, encode_ping(token))

    # -- telemetry ------------------------------------------------------------

    def read_state(self) -> ServoBusState:
        """Pump the link and return the most recent state.

        Never blocks and never raises for a transport hiccup: a control loop that
        can be knocked over by a serial read is not a control loop you want on a
        limb. Transport failures mark the state ``comms_ok=False`` and are
        handled by the safety layer.
        """
        try:
            self._pump()
        except CommunicationError as exc:
            log.throttled(
                "servo-pump",
                "error",
                "motor controller link error",
                now=self._clock.monotonic(),
                error=str(exc),
            )
            with self._lock:
                self._state = ServoBusState(
                    fingers=self._state.fingers,
                    timestamp=self._clock.monotonic(),
                    sequence=self._state.sequence,
                    enabled=False,
                    homed=self._state.homed,
                    estop=self._estop_latched,
                    moving=False,
                    bus_voltage_v=self._state.bus_voltage_v,
                    comms_ok=False,
                    faults=("transport_error",),
                )
        with self._lock:
            state = self._state
            age = self._clock.monotonic() - self._last_state_at
            if self._last_state_at == 0.0 or age > self._state_timeout:
                # Stale telemetry: report it rather than pretending the cached
                # pose is current. SafetyMonitor turns this into a comms fault.
                state = ServoBusState(
                    fingers=state.fingers,
                    timestamp=state.timestamp,
                    sequence=state.sequence,
                    enabled=state.enabled,
                    homed=state.homed,
                    estop=state.estop or self._estop_latched,
                    moving=False,
                    bus_voltage_v=state.bus_voltage_v,
                    comms_ok=False,
                    faults=(*state.faults, "telemetry_stale"),
                )
            return state

    def drain_events(self) -> list[tuple[EventCode, int, int]]:
        """Pop firmware events accumulated since the last call."""
        with self._lock:
            events = list(self._events)
            self._events.clear()
        return events

    def link_stats(self) -> dict[str, int]:
        return self._parser.stats()

    # -- internals ------------------------------------------------------------

    def _send(self, msg_id: MessageId, payload: bytes) -> None:
        if not self._transport.is_open:
            raise DeviceNotAvailableError(
                "motor controller transport is not open", context={"msg": msg_id.name}
            )
        with self._lock:
            self._sequence = (self._sequence + 1) & 0xFF
            sequence = self._sequence
        self._transport.write(encode_frame(int(msg_id), sequence, payload))

    def _pump(self) -> None:
        """Read available bytes and dispatch every complete frame."""
        if not self._transport.is_open:
            return
        data = self._transport.read()
        if not data:
            return
        for frame in self._parser.feed(data):
            self._dispatch(frame.msg_id, frame.payload)

    def _dispatch(self, msg_id: int, payload: bytes) -> None:
        try:
            if msg_id == MessageId.STATE:
                self._on_state(payload)
            elif msg_id == MessageId.PONG:
                _, version, _uptime = decode_pong(payload)
                self._firmware_version = ".".join(str(v) for v in version)
            elif msg_id == MessageId.EVENT:
                code, finger, detail = decode_event(payload)
                with self._lock:
                    self._events.append((code, finger, detail))
                if code in (EventCode.ESTOP_ENGAGED, EventCode.WATCHDOG_TRIP):
                    self._estop_latched = True
                if self.on_event is not None:
                    self.on_event(code, finger, detail)
            elif msg_id == MessageId.ERROR:
                code, detail = decode_error(payload)
                with self._lock:
                    self._errors.append((code, detail))
                log.warning("firmware error", code=code.name, detail=detail)
                if self.on_error is not None:
                    self.on_error(code, detail)
            elif msg_id == MessageId.LOG:
                level, text = decode_log(payload)
                log.debug("firmware log", level=level, text=text)
            else:
                log.throttled(
                    f"unknown-msg-{msg_id}",
                    "warning",
                    "unknown message from firmware",
                    now=self._clock.monotonic(),
                    msg_id=msg_id,
                )
        except Exception as exc:
            log.throttled(
                "decode-error",
                "warning",
                "failed to decode firmware frame",
                now=self._clock.monotonic(),
                msg_id=msg_id,
                error=str(exc),
            )

    def _on_state(self, payload: bytes) -> None:
        message = decode_state(payload)
        now = self._clock.monotonic()
        with self._lock:
            dt = now - self._last_state_at if self._last_state_at else 0.0
            fingers = []
            for telemetry in message.fingers:
                previous = self._last_positions[telemetry.finger]
                velocity = (telemetry.position - previous) / dt if dt > 1e-4 else 0.0
                self._velocities[telemetry.finger] = velocity
                self._last_positions[telemetry.finger] = telemetry.position
                fingers.append(
                    FingerState(
                        finger=telemetry.finger,
                        position=telemetry.position,
                        target=telemetry.target,
                        velocity=velocity,
                        current_ma=telemetry.current_ma,
                        temperature_c=telemetry.temperature_c,
                        stalled=abs(telemetry.error) > 0.03 and abs(velocity) < 0.02,
                    )
                )

            faults: list[str] = []
            if message.flags & StatusFlags.OVERCURRENT:
                faults.append("overcurrent")
            if message.flags & StatusFlags.OVERTEMP:
                faults.append("overtemperature")
            if message.flags & StatusFlags.UNDERVOLTAGE:
                faults.append("undervoltage")
            if message.flags & StatusFlags.WATCHDOG_TRIPPED:
                faults.append("firmware_watchdog")

            self._state = ServoBusState(
                fingers=tuple(fingers),
                timestamp=now,
                sequence=message.sequence,
                enabled=message.enabled,
                homed=bool(message.flags & StatusFlags.HOMED),
                estop=message.estop_active or self._estop_latched,
                moving=message.moving,
                bus_voltage_v=message.bus_voltage_v,
                comms_ok=True,
                faults=tuple(faults),
            )
            self._last_state_at = now
            if message.estop_active:
                self._estop_latched = True
