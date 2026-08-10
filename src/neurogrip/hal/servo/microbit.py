"""BBC micro:bit motor-controller driver.

The micro:bit speaks the same NGP v1 wire format as the ESP32 controller, so
this is deliberately a thin subclass of :class:`~neurogrip.hal.servo.esp32.Esp32ServoBus`
rather than a parallel implementation. Framing, CRC checking, sequence tracking,
telemetry caching, staleness detection and the reconnect/resync path are all
shared code, and shared code is code that gets exercised by both boards' tests.

What differs is what the *board* can do, and the difference is not cosmetic:

**No current sensing.** The ESP32 board carries a shunt amplifier per finger.
The micro:bit has none, and the firmware reports zero. Everything that infers
from motor current is affected:

* contact detection (:mod:`neurogrip.control.force`) cannot tell "touching the
  object" from "still moving";
* :class:`~neurogrip.safety.rules.ServoTimeoutRule` cannot distinguish a stalled
  finger from a loaded one;
* :class:`~neurogrip.control.servo_calibration.ServoCalibrationWizard` cannot
  measure tendon slack at all — take-up detection *is* a current measurement.

So this driver does not declare ``CURRENT_SENSING``, and the layers above check
the capability rather than assuming it. A zero that means "no sensor" must never
be read as a zero that means "no load".

**No position feedback.** A hobby servo does not report where it is. The position
in ``STATE`` is the commanded position advanced through the firmware's rate
limiter — an open-loop estimate. ``POSITION_FEEDBACK`` is therefore not declared
either, and the tracking-error checks that depend on it are skipped.

**No bus-voltage measurement.** The firmware reports zero rather than a plausible
constant, so the actuator-supply self-test reports "not measured" instead of
passing on a number nobody read.

**Lower rates.** MicroPython runs the firmware loop at 50 Hz. That is not so much
a compromise as an admission: standard hobby servos accept a new pulse every
20 ms, so 50 Hz is the rate the actuator has regardless of what the host sends.

The host keeps running its control group at 200 Hz — trajectory generation,
limit enforcement and the safety checks all belong there — but this driver
**coalesces** target writes down to the firmware's rate. Sending all 200 would
put roughly half the 115200-baud link's capacity into commands the firmware
cannot act on, and would ask MicroPython to CRC-check 200 frames a second, which
it cannot do while also generating servo pulses. Coalescing keeps the latest
target and drops the intermediate ones, which is exactly right: they describe a
trajectory the firmware is already interpolating along.

Emergency stop, enable, disable and calibration are **never** coalesced. Those
are events, not samples, and dropping one would be a safety bug.
"""

from __future__ import annotations

from ...core.clock import Clock
from ...core.logging import get_logger
from ..base import DeviceInfo, DeviceKind
from ..transport.base import Transport
from .base import ServoLimits
from .esp32 import Esp32ServoBus

__all__ = ["MicrobitServoBus"]

log = get_logger(__name__)

#: The firmware loop rate. Telemetry arrives at half this.
FIRMWARE_LOOP_HZ = 50.0

#: Velocity ceiling for a board with no feedback and no current limit of its
#: own. Deliberately below the ESP32's: without current sensing there is nothing
#: to notice a finger driving into a hard stop, so the mechanism has to survive
#: it on its own, and slower motion is what makes that likely.
DEFAULT_MAX_VELOCITY = 1.2
DEFAULT_MAX_ACCELERATION = 4.0


class MicrobitServoBus(Esp32ServoBus):
    """NGP driver for a micro:bit v2 running ``firmware/microbit_servo_controller``."""

    def __init__(
        self,
        transport: Transport,
        clock: Clock | None = None,
        *,
        state_timeout: float = 0.5,
        watchdog_ms: int = 400,
        limits: ServoLimits | None = None,
        driver_board: str = "auto",
    ) -> None:
        # Defaults are looser than the ESP32's because the firmware is slower:
        # telemetry arrives at 25 Hz, so a 0.25 s staleness window would flap.
        super().__init__(
            transport,
            clock,
            state_timeout=state_timeout,
            watchdog_ms=watchdog_ms,
            limits=limits or ServoLimits(
                max_velocity=DEFAULT_MAX_VELOCITY,
                max_acceleration=DEFAULT_MAX_ACCELERATION,
            ),
        )
        #: ``pca9685``, ``direct`` or ``auto``. Recorded for the diagnostics
        #: screen; the firmware decides for itself at start-up.
        self.driver_board = driver_board
        self._write_period = 1.0 / FIRMWARE_LOOP_HZ
        #: ``None`` until the first write. Not ``0.0``: a simulated clock starts
        #: at zero, so zero is a real time and cannot mean "never".
        self._last_target_write: float | None = None
        #: Diagnostics: target writes actually sent, and how many were coalesced
        #: away. The ratio is how much headroom the link has.
        self.target_writes = 0
        self.coalesced_writes = 0

    def write_targets(self, pose, *, speed_scale: float = 1.0, force: float = 0.6) -> None:
        """Command a target pose, at no more than the firmware's loop rate.

        Dropping intermediate targets is safe *because* they are samples of a
        trajectory the firmware interpolates along anyway — the endpoint is
        unchanged and arrives on the next tick. A dropped **event** would not be
        safe, which is why only this method coalesces.
        """
        now = self._clock.monotonic()
        if (
            self._last_target_write is not None
            and now - self._last_target_write < self._write_period
        ):
            self.coalesced_writes += 1
            return
        self._last_target_write = now
        self.target_writes += 1
        super().write_targets(pose, speed_scale=speed_scale, force=force)

    def emergency_stop(self) -> None:
        # Clear the coalescing gate so the first command after recovery is sent
        # immediately rather than waiting out a period that began before the stop.
        self._last_target_write = None
        super().emergency_stop()

    def info(self) -> DeviceInfo:
        return DeviceInfo(
            name="servo-bus",
            kind=DeviceKind.SERVO_BUS,
            driver="microbit-ngp1",
            connection=self._transport.info().connection,
            firmware_version=self._firmware_version,
            # Deliberately empty. Not even HOMING: `home()` is a host-side
            # convention here, not a routine against hard stops. Claiming a
            # capability this board has not got would make the layers above
            # trust numbers nobody measured.
            capabilities=frozenset(),
            extra={
                **self._parser.stats(),
                "driver_board": self.driver_board,
                "loop_hz": int(FIRMWARE_LOOP_HZ),
                "target_writes": self.target_writes,
                "coalesced_writes": self.coalesced_writes,
            },
        )

    def home(self) -> None:
        """Homing is a host-side convention on this board.

        The firmware answers ``HOME`` with ``UNSUPPORTED`` — there are no
        endstops and no feedback, so there is nothing for it to home *against*.
        The open position is simply the commanded zero, which the controller
        reaches by driving there like any other target.
        """
        log.info("micro:bit has no homing routine; open is the commanded zero")
