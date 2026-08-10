"""Simulated servo bus with a first-order actuator and contact model.

This is not a toy stub. It is the reference plant used by every integration test
and by Training Mode when no hardware is attached, so it models the behaviours the
control and safety layers actually have to cope with:

* **Rate and acceleration limiting** — targets are not reached instantly.
* **Contact and stall** — a finger meeting an object stops short of its target and
  draws increasing current. This is what
  :class:`~neurogrip.control.force.AdaptiveGripController` senses.
* **Thermal build-up** — sustained current raises motor temperature towards a
  throttling threshold, exercising the thermal safety rule.
* **Sag under load** — a loaded tendon back-drives slightly, so "position equals
  target" is never assumed anywhere in the stack.
* **Tendon slack** — part of the servo's travel is consumed taking up fishing
  line before the finger moves at all, and the line is under no tension until it
  does. This is the mechanism
  :class:`~neurogrip.control.servo_calibration.ServoCalibrationWizard` measures,
  so it has to exist here or the wizard could only be tested on hardware.

The object being grasped is injected by :mod:`neurogrip.simulation.world`; with no
object present the fingers travel freely to their targets.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass

from ...core.clock import Clock, RealClock
from ...core.types import FINGER_COUNT, Finger, HandPose, clamp
from ..base import DeviceCapability, DeviceInfo, DeviceKind
from .base import ALL_FINGERS_MASK, FingerState, ServoBusState, ServoCalibration, ServoLimits

__all__ = ["ContactModel", "SimulatedFinger", "SimulatedServoBus", "TendonModel"]


@dataclass(slots=True)
class TendonModel:
    """The *physical* tendon routing of the simulated hand.

    Distinct from :class:`~neurogrip.hal.servo.base.ServoCalibration`, which is
    what the software *believes* about the tendons. The wizard's whole job is to
    make the second match the first, so conflating them would make the
    calibration tests vacuous.

    ``slack[i]`` is the fraction of finger ``i``'s servo travel consumed before
    the line goes taut. Assembly variation of 5–25% is normal for a hand strung
    by hand.
    """

    slack: list[float]

    @classmethod
    def ideal(cls) -> TendonModel:
        """Perfectly strung tendons — no slack on any finger."""
        return cls(slack=[0.0] * FINGER_COUNT)

    @classmethod
    def uniform(cls, slack: float) -> TendonModel:
        return cls(slack=[clamp(slack)] * FINGER_COUNT)


@dataclass(slots=True)
class ContactModel:
    """Where each finger meets the grasped object.

    ``blocked_at[i]`` is the closure beyond which finger ``i`` cannot travel. A
    value of ``1.0`` means "free to close fully" (nothing in the way).
    ``stiffness`` scales how quickly current rises once blocked, standing in for
    object compliance: a steel tool is stiff, a foam ball is not.
    """

    blocked_at: list[float]
    stiffness: float = 1.0
    #: Set when the object is too heavy/slippery for the applied force.
    slipping: bool = False

    @classmethod
    def free(cls) -> ContactModel:
        return cls(blocked_at=[1.0] * FINGER_COUNT)

    @classmethod
    def uniform(cls, closure: float, stiffness: float = 1.0) -> ContactModel:
        """An object that blocks every finger at the same closure."""
        return cls(blocked_at=[clamp(closure)] * FINGER_COUNT, stiffness=stiffness)


@dataclass(slots=True)
class SimulatedFinger:
    """State of one simulated actuator."""

    finger: Finger
    position: float = 0.0
    velocity: float = 0.0
    target: float = 0.0
    current_ma: float = 0.0
    temperature_c: float = 25.0
    stalled: bool = False
    enabled: bool = False


class SimulatedServoBus:
    """In-process actuator simulation implementing :class:`~.base.ServoBus`.

    Time advances in :meth:`read_state`, driven by the injected clock, so the
    simulation runs at whatever rate the control loop polls it — including
    faster than real time under :class:`~neurogrip.core.clock.SimulatedClock`.
    """

    #: Current drawn per unit of commanded velocity (mA per closure-unit/s).
    CURRENT_PER_VELOCITY = 180.0
    #: Baseline current when energised but stationary (holding torque).
    HOLDING_CURRENT_MA = 45.0
    #: Additional current per unit of blocked-position error while stalled.
    STALL_CURRENT_GAIN = 2200.0
    #: Current while the tendon is still slack — the motor is turning but the
    #: line carries no load, so it draws far less than its holding current.
    #: The *step* from this to ``HOLDING_CURRENT_MA`` at take-up is what makes
    #: the slack boundary detectable. With ideal tendons the line is taut from
    #: the start and this value is never used, so the plant behaves exactly as
    #: it did before slack was modelled.
    SLACK_CURRENT_MA = 12.0
    #: Thermal model: °C rise per amp-second, and passive cooling time constant.
    THERMAL_GAIN = 0.9
    THERMAL_TAU = 45.0
    AMBIENT_C = 25.0

    def __init__(
        self,
        clock: Clock | None = None,
        *,
        limits: ServoLimits | None = None,
        contact: ContactModel | None = None,
        tendons: TendonModel | None = None,
        response_lag: float = 0.02,
    ) -> None:
        self._clock = clock or RealClock()
        self._limits = limits or ServoLimits()
        self._contact = contact or ContactModel.free()
        self._tendons = tendons or TendonModel.ideal()
        self._lag = max(1e-3, response_lag)
        self._fingers = [SimulatedFinger(finger=f) for f in Finger]
        self._calibration = {f: ServoCalibration(finger=f) for f in Finger}
        self._open = False
        self._enabled_mask = 0
        self._homed = False
        self._estop = False
        self._speed_scale = 1.0
        self._force = 0.6
        self._sequence = 0
        self._last_update = self._clock.monotonic()
        self._lock = threading.RLock()
        #: Diagnostics: how many target writes have been accepted.
        self.commands_received = 0

    # -- device lifecycle -----------------------------------------------------

    def open(self) -> None:
        with self._lock:
            self._open = True
            self._last_update = self._clock.monotonic()

    def close(self) -> None:
        with self._lock:
            self._open = False
            self._enabled_mask = 0
            for finger in self._fingers:
                finger.enabled = False
                finger.velocity = 0.0
                finger.current_ma = 0.0

    @property
    def is_open(self) -> bool:
        return self._open

    def info(self) -> DeviceInfo:
        return DeviceInfo(
            name="servo-bus",
            kind=DeviceKind.SERVO_BUS,
            driver="simulated",
            connection="sim",
            firmware_version="sim-1.0.0",
            capabilities=frozenset(
                {
                    DeviceCapability.SIMULATED,
                    DeviceCapability.CURRENT_SENSING,
                    DeviceCapability.POSITION_FEEDBACK,
                    DeviceCapability.TEMPERATURE,
                    DeviceCapability.HOMING,
                }
            ),
        )

    # -- control interface ----------------------------------------------------

    def enable(self, mask: int = ALL_FINGERS_MASK) -> None:
        with self._lock:
            if self._estop:
                return
            self._enabled_mask |= mask & ALL_FINGERS_MASK
            for finger in self._fingers:
                finger.enabled = bool(self._enabled_mask & (1 << int(finger.finger)))

    def disable(self, mask: int = ALL_FINGERS_MASK) -> None:
        with self._lock:
            self._enabled_mask &= ~(mask & ALL_FINGERS_MASK)
            for finger in self._fingers:
                if not self._enabled_mask & (1 << int(finger.finger)):
                    finger.enabled = False
                    finger.velocity = 0.0

    def set_limits(self, limits: ServoLimits) -> None:
        with self._lock:
            self._limits = limits

    def set_calibration(self, calibration: ServoCalibration) -> None:
        with self._lock:
            self._calibration[calibration.finger] = calibration

    def write_targets(self, pose: HandPose, *, speed_scale: float = 1.0, force: float = 0.6) -> None:
        with self._lock:
            if self._estop:
                return
            self._speed_scale = clamp(speed_scale, 0.05, 2.0)
            self._force = clamp(force)
            for index, value in enumerate(pose.values):
                self._fingers[index].target = value
            self.commands_received += 1

    def read_state(self) -> ServoBusState:
        with self._lock:
            now = self._clock.monotonic()
            dt = now - self._last_update
            if dt > 0:
                self._step(dt)
                self._last_update = now
            self._sequence = (self._sequence + 1) & 0xFFFF
            return ServoBusState(
                fingers=tuple(
                    FingerState(
                        finger=f.finger,
                        position=f.position,
                        target=f.target,
                        velocity=f.velocity,
                        current_ma=int(f.current_ma),
                        temperature_c=f.temperature_c,
                        stalled=f.stalled,
                    )
                    for f in self._fingers
                ),
                timestamp=now,
                sequence=self._sequence,
                enabled=self._enabled_mask != 0,
                homed=self._homed,
                estop=self._estop,
                moving=any(abs(f.velocity) > 1e-3 for f in self._fingers),
                bus_voltage_v=7.4 - 0.0004 * sum(f.current_ma for f in self._fingers),
                comms_ok=self._open,
            )

    def home(self) -> None:
        """Instantaneous homing: drive to fully open and zero the reference."""
        with self._lock:
            for finger in self._fingers:
                finger.position = 0.0
                finger.target = 0.0
                finger.velocity = 0.0
                finger.stalled = False
            self._homed = True

    def emergency_stop(self) -> None:
        with self._lock:
            self._estop = True
            self._enabled_mask = 0
            for finger in self._fingers:
                finger.enabled = False
                finger.velocity = 0.0
                finger.current_ma = 0.0
                finger.target = finger.position

    def clear_emergency_stop(self) -> None:
        with self._lock:
            self._estop = False

    # -- simulation -----------------------------------------------------------

    def set_contact(self, contact: ContactModel) -> None:
        """Install a new contact model (called by the simulated world)."""
        with self._lock:
            self._contact = contact

    def set_tendons(self, tendons: TendonModel) -> None:
        """Install a new tendon model — how the hand is *actually* strung."""
        with self._lock:
            self._tendons = tendons

    def _mechanical_target(self, finger: SimulatedFinger) -> float:
        """Finger closure the commanded target actually produces.

        Two mappings in series: the host's calibration turns commanded closure
        into servo travel, and the physical tendon turns servo travel into
        finger motion. They cancel exactly when the calibration is correct,
        which is the property the calibration tests assert.
        """
        calibration = self._calibration[finger.finger]
        true_slack = self._tendons.slack[int(finger.finger)]
        servo_travel = calibration.slack + finger.target * (1.0 - calibration.slack)
        if true_slack >= 1.0:
            return 0.0
        return clamp((servo_travel - true_slack) / (1.0 - true_slack))

    def _tendon_taut(self, finger: SimulatedFinger) -> bool:
        calibration = self._calibration[finger.finger]
        servo_travel = calibration.slack + finger.target * (1.0 - calibration.slack)
        return servo_travel > self._tendons.slack[int(finger.finger)]

    def _step(self, dt: float) -> None:
        """Advance the plant by ``dt`` seconds."""
        # Guard against a very large dt after a pause: a 10 s jump must not
        # teleport the fingers. Clamp to 100 ms per step.
        dt = min(dt, 0.1)
        max_velocity = self._limits.max_velocity * self._speed_scale
        max_accel = self._limits.max_acceleration * self._speed_scale

        for finger in self._fingers:
            blocked_at = self._contact.blocked_at[int(finger.finger)]

            if not finger.enabled:
                # Unpowered: fingers relax open slowly under the return spring.
                finger.velocity = 0.0
                finger.position = max(0.0, finger.position - 0.25 * dt)
                finger.current_ma = _decay(finger.current_ma, 0.0, dt, 0.05)
                finger.stalled = False
                self._thermal(finger, dt)
                continue

            mechanical_target = self._mechanical_target(finger)
            error = mechanical_target - finger.position
            # Desired velocity from a first-order response, capped both by the
            # velocity limit and by the speed from which the actuator can still
            # stop on target (v = sqrt(2·a·|e|)). Without that second cap a step
            # input produces an overshoot the real servo's internal position
            # loop would never allow.
            stopping = math.sqrt(2.0 * max_accel * abs(error))
            desired = math.copysign(
                min(max_velocity, stopping, abs(error) / self._lag), error
            )
            delta_v = clamp(desired - finger.velocity, -max_accel * dt, max_accel * dt)
            finger.velocity += delta_v

            new_position = finger.position + finger.velocity * dt
            if (new_position - mechanical_target) * error > 0:
                # The step would carry the finger past its target; land on it.
                new_position = mechanical_target
                finger.velocity = 0.0

            if new_position >= blocked_at - 1e-6 and mechanical_target > blocked_at:
                # Contact: the finger cannot advance past the object surface.
                penetration = min(0.15, (mechanical_target - blocked_at))
                finger.position = blocked_at + penetration * 0.05 * self._contact.stiffness
                finger.velocity = 0.0
                finger.stalled = True
                # Current rises with commanded force and how hard we push past contact.
                finger.current_ma = _approach(
                    finger.current_ma,
                    self.HOLDING_CURRENT_MA
                    + self.STALL_CURRENT_GAIN * penetration * self._force * self._contact.stiffness,
                    dt,
                    0.08,
                )
            else:
                finger.position = clamp(new_position)
                finger.stalled = False
                if finger.position in (0.0, 1.0):
                    finger.velocity = 0.0
                # A slack tendon carries no load, so the motor draws only what it
                # needs to turn itself until the line goes taut.
                resting = (
                    self.HOLDING_CURRENT_MA
                    if self._tendon_taut(finger)
                    else self.SLACK_CURRENT_MA
                )
                finger.current_ma = _approach(
                    finger.current_ma,
                    resting + self.CURRENT_PER_VELOCITY * abs(finger.velocity),
                    dt,
                    0.05,
                )

            # Firmware-side current clamp; the real controller does the same.
            finger.current_ma = min(finger.current_ma, float(self._limits.max_current_ma))
            self._thermal(finger, dt)

    def _thermal(self, finger: SimulatedFinger, dt: float) -> None:
        """First-order thermal model: heating from current, passive cooling."""
        heating = self.THERMAL_GAIN * (finger.current_ma / 1000.0) ** 2
        cooling = (finger.temperature_c - self.AMBIENT_C) / self.THERMAL_TAU
        finger.temperature_c += (heating - cooling) * dt


def _approach(current: float, target: float, dt: float, tau: float) -> float:
    """Exponential approach of ``current`` towards ``target`` with time constant ``tau``."""
    alpha = 1.0 - pow(2.718281828459045, -dt / max(1e-6, tau))
    return current + (target - current) * alpha


def _decay(current: float, target: float, dt: float, tau: float) -> float:
    return _approach(current, target, dt, tau)
