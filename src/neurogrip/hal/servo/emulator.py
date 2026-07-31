"""Software emulator of the ESP32 motor-controller firmware.

Paired with :class:`~neurogrip.hal.transport.loopback.LoopbackTransport`, this lets
the production :class:`~neurogrip.hal.servo.esp32.Esp32ServoBus` driver run without
hardware — exercising the real framing, the real protocol encoders, sequence
handling, the firmware watchdog and the e-stop latch.

It mirrors the behaviour of ``firmware/esp32_motor_controller``. When the firmware
changes, this emulator and ``tests/unit/test_protocol.py`` must change with it;
that coupling is intentional, and is what keeps the two implementations honest.
"""

from __future__ import annotations

from ...core.clock import Clock, RealClock
from ...core.types import Finger
from ..protocol import (
    ErrorCode,
    EventCode,
    FingerTelemetry,
    MessageId,
    StatusFlags,
    decode_set_limits,
    decode_set_targets,
    encode_error,
    encode_event,
    encode_pong,
    encode_state,
)
from ..transport.framing import FrameParser, encode_frame
from .base import ServoLimits
from .simulated import ContactModel, SimulatedServoBus

__all__ = ["FIRMWARE_VERSION", "Esp32Emulator"]

FIRMWARE_VERSION = (1, 0, 0)


class Esp32Emulator:
    """MCU-side state machine driving a :class:`SimulatedServoBus` plant."""

    #: Telemetry push rate, matching the firmware's ``STATE`` cadence.
    STATE_RATE_HZ = 100.0

    def __init__(
        self,
        clock: Clock | None = None,
        *,
        plant: SimulatedServoBus | None = None,
        watchdog_ms: int = 300,
    ) -> None:
        self._clock = clock or RealClock()
        self._plant = plant or SimulatedServoBus(self._clock)
        self._parser = FrameParser()
        self._outbox = bytearray()
        self._sequence = 0
        self._boot_time = self._clock.monotonic()
        self._next_state_at = 0.0
        self._watchdog_timeout = watchdog_ms / 1000.0
        self._last_command_at = self._clock.monotonic()
        self._watchdog_tripped = False
        self._estop = False
        self._homed = False
        self._enabled_mask = 0
        self._limits = ServoLimits()
        self._pending_events: list[bytes] = []
        self._plant.open()

    # -- EmulatedDevice protocol ---------------------------------------------

    def reset(self) -> None:
        """Simulate a power-on reset."""
        self._parser.reset()
        self._outbox.clear()
        self._boot_time = self._clock.monotonic()
        self._last_command_at = self._boot_time
        self._watchdog_tripped = False
        self._estop = False
        self._homed = False
        self._enabled_mask = 0
        self._plant.clear_emergency_stop()
        self._plant.close()
        self._plant.open()

    def on_host_bytes(self, data: bytes) -> None:
        for frame in self._parser.feed(data):
            self._handle(frame.msg_id, frame.payload)

    def poll(self, now: float) -> bytes:
        """Advance firmware time and return bytes to transmit."""
        self._check_watchdog(now)
        if now >= self._next_state_at:
            self._next_state_at = now + 1.0 / self.STATE_RATE_HZ
            self._emit_state()
        out = bytes(self._outbox)
        self._outbox.clear()
        return out

    # -- plant access (used by the simulated world) ---------------------------

    @property
    def plant(self) -> SimulatedServoBus:
        return self._plant

    def set_contact(self, contact: ContactModel) -> None:
        self._plant.set_contact(contact)

    # -- message handling -----------------------------------------------------

    def _handle(self, msg_id: int, payload: bytes) -> None:
        now = self._clock.monotonic()
        try:
            message = MessageId(msg_id)
        except ValueError:
            self._send(MessageId.ERROR, encode_error(ErrorCode.UNKNOWN_MESSAGE, msg_id))
            return

        if message == MessageId.PING:
            token = int.from_bytes(payload[:4], "little") if len(payload) >= 4 else 0
            self._send(
                MessageId.PONG,
                encode_pong(token, FIRMWARE_VERSION, int((now - self._boot_time) * 1000)),
            )
            return

        if message == MessageId.SET_TARGETS:
            self._last_command_at = now
            if self._estop:
                self._send(MessageId.ERROR, encode_error(ErrorCode.ESTOP_ACTIVE))
                return
            if not self._enabled_mask:
                self._send(MessageId.ERROR, encode_error(ErrorCode.NOT_ENABLED))
                return
            pose, speed, _flags = decode_set_targets(payload)
            self._plant.write_targets(pose, speed_scale=speed)
            if self._watchdog_tripped:
                self._watchdog_tripped = False
            return

        if message == MessageId.SET_LIMITS:
            velocity, accel, current, temperature = decode_set_limits(payload)
            self._limits = ServoLimits(
                max_velocity=velocity,
                max_acceleration=accel,
                max_current_ma=current,
                max_temperature_c=float(temperature),
                max_force=self._limits.max_force,
            )
            self._plant.set_limits(self._limits)
            return

        if message == MessageId.SET_FORCE:
            self._last_command_at = now
            return

        if message == MessageId.ENABLE:
            if self._estop:
                self._send(MessageId.ERROR, encode_error(ErrorCode.ESTOP_ACTIVE))
                return
            self._enabled_mask = payload[0] if payload else 0x1F
            self._plant.enable(self._enabled_mask)
            self._last_command_at = now
            return

        if message == MessageId.DISABLE:
            mask = payload[0] if payload else 0x1F
            self._enabled_mask &= ~mask
            self._plant.disable(mask)
            return

        if message == MessageId.ESTOP:
            self._engage_estop()
            return

        if message == MessageId.CLEAR_ESTOP:
            magic = int.from_bytes(payload[:2], "little") if len(payload) >= 2 else 0
            if magic != 0x5EA1:
                self._send(MessageId.ERROR, encode_error(ErrorCode.BAD_PARAMETER))
                return
            self._estop = False
            self._watchdog_tripped = False
            self._plant.clear_emergency_stop()
            self._send(MessageId.EVENT, encode_event(EventCode.ESTOP_RELEASED, 0xFF, 0))
            self._last_command_at = now
            return

        if message == MessageId.HOME:
            self._plant.home()
            self._homed = True
            self._last_command_at = now
            self._send(MessageId.EVENT, encode_event(EventCode.HOMING_COMPLETE, 0xFF, 0))
            return

        if message == MessageId.SET_WATCHDOG:
            timeout_ms = int.from_bytes(payload[:2], "little") if len(payload) >= 2 else 0
            self._watchdog_timeout = timeout_ms / 1000.0
            self._last_command_at = now
            return

        if message == MessageId.REQUEST_STATE:
            self._emit_state()
            return

        if message == MessageId.REBOOT:
            self.reset()
            return

        self._send(MessageId.ERROR, encode_error(ErrorCode.UNKNOWN_MESSAGE, msg_id))

    def _check_watchdog(self, now: float) -> None:
        """Safe the actuators when the host stops issuing commands.

        This is the single most important firmware behaviour: it means a host
        crash results in a hand that holds still and relaxes, not one that keeps
        squeezing whatever it was holding.
        """
        if self._watchdog_timeout <= 0 or self._watchdog_tripped or self._estop:
            return
        if now - self._last_command_at <= self._watchdog_timeout:
            return
        self._watchdog_tripped = True
        self._plant.disable()
        self._enabled_mask = 0
        self._send(MessageId.EVENT, encode_event(EventCode.WATCHDOG_TRIP, 0xFF, 0))

    def _engage_estop(self) -> None:
        self._estop = True
        self._enabled_mask = 0
        self._plant.emergency_stop()
        self._send(MessageId.EVENT, encode_event(EventCode.ESTOP_ENGAGED, 0xFF, 0))

    def _emit_state(self) -> None:
        state = self._plant.read_state()
        flags = StatusFlags.NONE
        if self._enabled_mask:
            flags |= StatusFlags.ENABLED
        if state.moving:
            flags |= StatusFlags.MOVING
        if self._estop:
            flags |= StatusFlags.ESTOP
        if self._homed:
            flags |= StatusFlags.HOMED
        if state.total_current_ma > self._limits.max_current_ma * 0.95:
            flags |= StatusFlags.OVERCURRENT
        if state.max_temperature_c > self._limits.max_temperature_c:
            flags |= StatusFlags.OVERTEMP
        if self._watchdog_tripped:
            flags |= StatusFlags.WATCHDOG_TRIPPED
        if state.bus_voltage_v < 6.4:
            flags |= StatusFlags.UNDERVOLTAGE

        telemetry = tuple(
            FingerTelemetry(
                finger=Finger(index),
                position=fs.position,
                target=fs.target,
                current_ma=fs.current_ma,
                temperature_c=fs.temperature_c,
            )
            for index, fs in enumerate(state.fingers)
        )
        for finger_state in state.fingers:
            if finger_state.stalled:
                self._send(
                    MessageId.EVENT,
                    encode_event(EventCode.CONTACT_DETECTED, int(finger_state.finger), 0),
                )
                break

        self._send(
            MessageId.STATE,
            encode_state(
                sequence=state.sequence & 0xFF,
                flags=flags,
                fingers=telemetry,
                bus_voltage_v=state.bus_voltage_v,
                uptime_ms=int((self._clock.monotonic() - self._boot_time) * 1000),
            ),
        )

    def _send(self, msg_id: MessageId, payload: bytes) -> None:
        self._sequence = (self._sequence + 1) & 0xFF
        self._outbox.extend(encode_frame(int(msg_id), self._sequence, payload))
