"""Trajectory generation.

Motion is generated on the host, not delegated to the servo controller, for one
reason: **a movement must be interruptible at any instant**. If the ESP32 were
executing a stored profile, a cancel would have to wait for a round trip. Here,
the next control cycle simply generates a different setpoint, so an abort takes
effect within one 5 ms tick.

Profile shape
-------------
Each finger follows a trapezoidal velocity profile (accelerate, cruise,
decelerate) respecting its acceleration and velocity limits. Fingers are then
**time-synchronised**: the slowest finger sets the duration and the others are
scaled to match, so the hand arrives in one coordinated motion rather than
fingers snapping into place one after another. Uncoordinated arrival looks
mechanical and, more practically, closes some fingers on an object before others
are in position, which pushes it out of the grasp.

An optional S-curve (jerk-limited) blend smooths the acceleration corners. It
costs a little settling time, so Sports Mode turns it off.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..core.types import FINGER_COUNT, HandPose, clamp

__all__ = ["MotionLimits", "TrajectoryGenerator", "TrajectoryState"]


@dataclass(frozen=True, slots=True)
class MotionLimits:
    """Kinematic limits for trajectory generation.

    Distinct from :class:`~neurogrip.hal.servo.base.ServoLimits`: those are the
    hard actuator/firmware ceilings, these are the *planning* limits a mode
    chooses within them. :meth:`clipped_to` enforces that relationship.
    """

    #: Closure units per second.
    max_velocity: float = 1.6
    #: Closure units per second squared.
    max_acceleration: float = 6.0
    #: Closure units per second cubed; ``0`` disables jerk limiting.
    max_jerk: float = 40.0
    #: Distance from target within which motion is considered complete.
    position_tolerance: float = 0.015
    #: Smallest commanded step; below this the setpoint is held to avoid dither.
    min_step: float = 0.0005

    def scaled(self, factor: float) -> MotionLimits:
        """Scale velocity and acceleration by ``factor`` (jerk scales with it too)."""
        f = max(0.05, factor)
        return MotionLimits(
            max_velocity=self.max_velocity * f,
            max_acceleration=self.max_acceleration * f,
            max_jerk=self.max_jerk * f,
            position_tolerance=self.position_tolerance,
            min_step=self.min_step,
        )

    def clipped_to(self, ceiling: MotionLimits) -> MotionLimits:
        """Clip these limits so they never exceed a hard ceiling."""
        return MotionLimits(
            max_velocity=min(self.max_velocity, ceiling.max_velocity),
            max_acceleration=min(self.max_acceleration, ceiling.max_acceleration),
            max_jerk=min(self.max_jerk, ceiling.max_jerk) if ceiling.max_jerk > 0 else self.max_jerk,
            position_tolerance=self.position_tolerance,
            min_step=self.min_step,
        )


@dataclass(frozen=True, slots=True)
class TrajectoryState:
    """Output of one trajectory step."""

    pose: HandPose
    velocities: tuple[float, ...]
    #: Fraction of the motion completed, in ``[0, 1]``.
    progress: float
    complete: bool
    #: Seconds remaining at the current rate.
    time_remaining: float = 0.0

    @property
    def max_velocity(self) -> float:
        return max((abs(v) for v in self.velocities), default=0.0)


class TrajectoryGenerator:
    """Generates synchronised, limit-respecting motion between poses.

    Stateful: :meth:`start` sets a goal, :meth:`step` advances by ``dt``.
    :meth:`retarget` changes the goal mid-motion *without* resetting velocity,
    which is what makes continuous proportional control smooth — the user's
    changing effort updates the target every cycle.
    """

    def __init__(self, limits: MotionLimits | None = None, *, s_curve: bool = True) -> None:
        self._limits = limits or MotionLimits()
        self._s_curve = s_curve
        self._current = HandPose.open_hand()
        self._target = HandPose.open_hand()
        self._velocities = [0.0] * FINGER_COUNT
        self._start_pose = HandPose.open_hand()
        self._elapsed = 0.0
        self._duration = 0.0
        self._active = False

    # -- configuration --------------------------------------------------------

    @property
    def limits(self) -> MotionLimits:
        return self._limits

    def set_limits(self, limits: MotionLimits) -> None:
        self._limits = limits

    def set_s_curve(self, enabled: bool) -> None:
        """Sports Mode disables jerk limiting to shave settling time."""
        self._s_curve = enabled

    # -- state ----------------------------------------------------------------

    @property
    def current(self) -> HandPose:
        return self._current

    @property
    def target(self) -> HandPose:
        return self._target

    @property
    def active(self) -> bool:
        return self._active

    @property
    def velocities(self) -> tuple[float, ...]:
        return tuple(self._velocities)

    def sync(self, pose: HandPose) -> None:
        """Force the internal position to match measured reality.

        Called after homing, an e-stop or a back-drive event, so the generator
        does not continue from a position the hand is no longer in.
        """
        self._current = pose
        self._velocities = [0.0] * FINGER_COUNT

    def start(
        self,
        target: HandPose,
        *,
        from_pose: HandPose | None = None,
        speed_scale: float = 1.0,
    ) -> float:
        """Begin a motion to ``target``; returns the estimated duration."""
        self._start_pose = from_pose if from_pose is not None else self._current
        if from_pose is not None:
            self._current = from_pose
        self._target = target
        self._elapsed = 0.0
        self._duration = self.estimate_duration(self._start_pose, target, speed_scale)
        self._active = self._duration > 0.0 or not self._current.is_close(
            target, self._limits.position_tolerance
        )
        return self._duration

    def retarget(self, target: HandPose) -> None:
        """Change the goal without discarding current velocity."""
        self._target = target
        self._active = True

    def stop(self) -> None:
        """Abandon the motion immediately, holding the current position."""
        self._active = False
        self._velocities = [0.0] * FINGER_COUNT
        self._target = self._current

    # -- stepping -------------------------------------------------------------

    def step(self, dt: float, *, speed_scale: float = 1.0) -> TrajectoryState:
        """Advance the trajectory by ``dt`` seconds."""
        if dt <= 0:
            return self._state()

        limits = self._limits.scaled(speed_scale)
        errors = [self._target[i] - self._current[i] for i in range(FINGER_COUNT)]
        max_error = max((abs(e) for e in errors), default=0.0)

        if max_error <= limits.position_tolerance and self._all_slow():
            self._current = self._target
            self._velocities = [0.0] * FINGER_COUNT
            self._active = False
            return self._state(complete=True)

        # Synchronisation: the finger with the largest remaining travel sets the
        # pace; every other finger is scaled so they finish together.
        lead_error = max(max_error, 1e-9)
        positions = list(self._current.values)

        for index in range(FINGER_COUNT):
            error = errors[index]
            share = abs(error) / lead_error
            finger_velocity_limit = limits.max_velocity * max(share, 1e-6)
            finger_accel_limit = limits.max_acceleration * max(share, 1e-6)

            # Velocity that still allows stopping exactly on target:
            #   v = sqrt(2·a·|e|)
            stopping_velocity = math.sqrt(2.0 * finger_accel_limit * abs(error))
            desired = math.copysign(min(finger_velocity_limit, stopping_velocity), error)

            delta = desired - self._velocities[index]
            max_delta = finger_accel_limit * dt
            if self._s_curve and limits.max_jerk > 0:
                # Jerk limiting: cap how fast the acceleration itself may change.
                max_delta = min(max_delta, abs(self._velocities[index]) * 0.5 + limits.max_jerk * dt * dt)
            delta = max(-max_delta, min(max_delta, delta))
            self._velocities[index] += delta

            step = self._velocities[index] * dt
            if abs(step) < limits.min_step and abs(error) > limits.position_tolerance:
                # Overcome stiction/quantisation with a minimum commanded step.
                step = math.copysign(limits.min_step, error)
            # Never overshoot the target.
            if abs(step) > abs(error):
                step = error
                self._velocities[index] = error / dt if dt > 0 else 0.0
            positions[index] = clamp(positions[index] + step)

        self._current = HandPose(tuple(positions))  # type: ignore[arg-type]
        self._elapsed += dt
        return self._state()

    def _all_slow(self) -> bool:
        return all(abs(v) < 0.02 for v in self._velocities)

    def _state(self, *, complete: bool = False) -> TrajectoryState:
        total = self._start_pose.max_difference(self._target)
        remaining = self._current.max_difference(self._target)
        progress = 1.0 if total <= 1e-9 else clamp(1.0 - remaining / total)
        speed = max((abs(v) for v in self._velocities), default=0.0)
        return TrajectoryState(
            pose=self._current,
            velocities=tuple(self._velocities),
            progress=progress,
            complete=complete or not self._active,
            time_remaining=remaining / speed if speed > 1e-6 else 0.0,
        )

    # -- planning helpers -----------------------------------------------------

    def estimate_duration(
        self, start: HandPose, target: HandPose, speed_scale: float = 1.0
    ) -> float:
        """Duration of a trapezoidal move between two poses.

        Used by the UI's progress indicator and by the motion queue's timeout,
        so a stuck motion is detected rather than waited on forever.
        """
        limits = self._limits.scaled(speed_scale)
        distance = start.max_difference(target)
        if distance <= limits.position_tolerance:
            return 0.0

        v_max = limits.max_velocity
        a_max = limits.max_acceleration
        # Distance needed to reach cruise velocity and come back down.
        ramp_distance = v_max * v_max / a_max
        if distance < ramp_distance:
            # Triangular profile: never reaches cruise velocity.
            return 2.0 * math.sqrt(distance / a_max)
        return distance / v_max + v_max / a_max

    def partial_target(self, fraction: float, goal: HandPose) -> HandPose:
        """Pose ``fraction`` of the way from the current pose towards ``goal``."""
        return self._current.blend(goal, clamp(fraction))
