"""The hand controller.

**This is the only component in the system that writes to the servo bus.**

That single rule is what makes the safety properties enforceable rather than
aspirational. Emergency stop, force ceilings, velocity limits, self-collision
avoidance and cancellation each need exactly one implementation, in one place,
that nothing can bypass — because there is no other path to the actuators.

Per control cycle (200 Hz by default) it:

1. reads servo telemetry and updates the measured hand state;
2. runs contact detection and force regulation;
3. checks watchdogs, timeouts and limits;
4. advances the active trajectory;
5. writes exactly one target-pose command to the bus.

Commands enter through :meth:`submit`, which delegates arbitration to the
:class:`~neurogrip.control.queue.MotionQueue`. Callers never touch the actuators
directly, and there is deliberately no method that lets them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from ..core.clock import Clock
from ..core.errors import EmergencyStopActive
from ..core.events import EventBus
from ..core.lifecycle import HealthReport, HealthStatus, ServiceBase
from ..core.logging import get_logger
from ..core.topics import Topics
from ..core.types import Finger, HandPose, clamp
from ..hal.servo.base import ServoBus, ServoBusState, ServoCalibration, ServoLimits
from .force import AdaptiveGripController, GripSettings, GripState
from .grips import GripLibrary
from .kinematics import HandKinematics
from .motion import MotionLimits, TrajectoryGenerator
from .queue import CommandResult, MotionCommand, MotionQueue, Priority

__all__ = ["HandController", "HandState"]

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class HandState:
    """Complete observable state of the hand, published every control cycle."""

    #: Measured pose from servo feedback.
    pose: HandPose
    #: Pose currently being commanded.
    commanded: HandPose
    #: Final goal of the active motion, if any.
    goal: HandPose | None = None
    velocities: tuple[float, ...] = field(default_factory=tuple)
    grip: GripState | None = None
    moving: bool = False
    enabled: bool = False
    homed: bool = False
    estop: bool = False
    comms_ok: bool = True
    #: Force ceiling currently applied.
    force: float = 0.0
    #: Per-finger motor current, mA.
    currents: tuple[int, ...] = field(default_factory=tuple)
    #: Highest motor temperature, °C.
    temperature_c: float = 0.0
    bus_voltage_v: float = 0.0
    timestamp: float = 0.0
    #: Description of the motion in progress, for the UI.
    activity: str = "idle"
    faults: tuple[str, ...] = field(default_factory=tuple)

    @property
    def holding(self) -> bool:
        return self.grip.holding if self.grip else False

    @property
    def total_current_ma(self) -> int:
        return sum(self.currents)

    @property
    def is_open(self) -> bool:
        return self.pose.aperture > 0.85


class HandController(ServiceBase):
    """Owns the actuators, executes motion, enforces limits."""

    service_name = "control"

    def __init__(
        self,
        servo_bus: ServoBus,
        clock: Clock,
        bus: EventBus,
        *,
        grips: GripLibrary | None = None,
        kinematics: HandKinematics | None = None,
        motion_limits: MotionLimits | None = None,
        servo_limits: ServoLimits | None = None,
        grip_settings: GripSettings | None = None,
        calibration: Sequence[ServoCalibration] = (),
        publish_state: bool = True,
    ) -> None:
        super().__init__()
        self._servo = servo_bus
        #: Pushed to the bus on start. Held here rather than applied by the
        #: caller because the controller owns every write to the actuators.
        self._calibration = tuple(calibration)
        self._clock = clock
        self._bus = bus
        self._grips = grips or GripLibrary()
        self._kinematics = kinematics or HandKinematics()
        self._hard_limits = motion_limits or MotionLimits()
        self._servo_limits = servo_limits or ServoLimits()
        self._trajectory = TrajectoryGenerator(self._hard_limits)
        self._queue = MotionQueue()
        self._grip = AdaptiveGripController(
            clock, grip_settings, max_force=self._servo_limits.max_force
        )
        self._publish_state = publish_state

        self._state = HandState(
            pose=HandPose.open_hand(), commanded=HandPose.open_hand(), timestamp=0.0
        )
        #: The pose written to the bus. Held here rather than read back out of
        #: the (frozen) state snapshot, so "hold exactly where you are" after a
        #: cancel is a single unambiguous assignment.
        self._commanded = HandPose.open_hand()
        self._estop = False
        #: True while the active stop is a deliberate self-check rather than a
        #: real one. Affects only how it is described, never what it does.
        self._estop_diagnostic = False
        #: When an e-stop rehearsal last reached this controller. ``None`` until
        #: the first one; see :meth:`on_estop_record`.
        self._last_estop_rehearsal: float | None = None
        self._speed_scale = 1.0
        self._force_ceiling = self._servo_limits.max_force
        self._last_write_at = 0.0
        #: Diagnostics counters.
        self.cycles = 0
        self.writes = 0
        self.limit_corrections = 0

    # -- lifecycle ------------------------------------------------------------

    def on_start(self) -> None:
        self._servo.open()
        self._servo.set_limits(self._servo_limits)
        self._push_calibration()
        state = self._servo.read_state()
        self._trajectory.sync(state.pose)
        self._commanded = state.pose
        self._state = self._build_state(state, "starting")
        log.info("hand controller started", device=str(self._servo.info()))

    def on_stop(self) -> None:
        """Safe shutdown: stop motion, relax to open, de-energise, close."""
        try:
            self._trajectory.stop()
            self._queue.clear(self._clock.monotonic())
            if not self._estop:
                # Relax the grip so nothing is left clamped when power is removed.
                self._servo.write_targets(HandPose.open_hand(), speed_scale=0.5, force=0.2)
            self._servo.disable()
        except Exception as exc:
            log.warning("error while safing the hand during shutdown", error=str(exc))
        finally:
            self._servo.close()
            log.info("hand controller stopped")

    def _push_calibration(self) -> None:
        """Send per-finger endpoints to the controller.

        A failure here is logged but not fatal: the firmware keeps its stored
        endpoints, which are safe defaults, and the hand remains usable. Refusing
        to start over a calibration push would turn a tuning problem into a
        device that will not run at all.
        """
        for calibration in self._calibration:
            try:
                self._servo.set_calibration(calibration)
            except Exception as exc:
                log.warning(
                    "could not push servo calibration",
                    finger=calibration.finger.name.lower(),
                    error=str(exc),
                )
        if self._calibration:
            log.info(
                "servo calibration applied",
                slack={c.finger.name.lower(): round(c.slack, 3) for c in self._calibration},
            )

    def set_calibration(self, calibration: Sequence[ServoCalibration]) -> None:
        """Replace the calibration and push it immediately.

        Called after a calibration run so the new tendon slack takes effect
        without a restart.
        """
        self._calibration = tuple(calibration)
        self._push_calibration()

    # -- configuration --------------------------------------------------------

    def configure(
        self,
        *,
        motion_limits: MotionLimits | None = None,
        force_ceiling: float | None = None,
        speed_scale: float | None = None,
        s_curve: bool | None = None,
    ) -> None:
        """Apply a mode's execution envelope.

        Planning limits are always clipped to the hard servo limits, so a mode
        cannot widen the envelope beyond what the hardware permits.
        """
        if motion_limits is not None:
            self._trajectory.set_limits(motion_limits.clipped_to(self._hard_limits))
        if force_ceiling is not None:
            self._force_ceiling = clamp(min(force_ceiling, self._servo_limits.max_force))
            self._grip.set_max_force(self._force_ceiling)
        if speed_scale is not None:
            self._speed_scale = max(0.05, min(2.0, speed_scale))
        if s_curve is not None:
            self._trajectory.set_s_curve(s_curve)

    @property
    def grips(self) -> GripLibrary:
        return self._grips

    @property
    def kinematics(self) -> HandKinematics:
        return self._kinematics

    @property
    def state(self) -> HandState:
        """Most recent state snapshot; safe to read from any thread."""
        return self._state

    @property
    def queue(self) -> MotionQueue:
        return self._queue

    # -- commands -------------------------------------------------------------

    def submit(self, command: MotionCommand) -> CommandResult:
        """Offer a motion command.

        Rejects everything while the e-stop latch is engaged, including
        emergency-priority commands: releasing the latch is an explicit,
        user-acknowledged action, never a side effect of a new command.
        """
        now = self._clock.monotonic()
        if self._estop:
            return CommandResult(accepted=False, reason="emergency stop is engaged")

        safe_target, notes = self._kinematics.enforce_limits(command.target)
        if notes:
            self.limit_corrections += 1
            log.debug("motion target adjusted", corrections=list(notes))
            command = command.with_target(safe_target)

        result = self._queue.submit(command, now)
        if result.accepted:
            self._skip_redundant_preshape()
            leg_target = self._queue.current_leg_target or safe_target
            if result.same_stream:
                # Continuous control refreshing its own command: update the goal
                # without discarding the current velocity or the pre-shape leg.
                self._trajectory.retarget(leg_target)
            else:
                self._trajectory.start(
                    leg_target,
                    from_pose=self._state.pose,
                    speed_scale=command.speed * self._speed_scale,
                )
            self._grip.set_commanded_force(min(command.force, self._force_ceiling))
        if result.accepted and not result.same_stream:
            self._bus.publish(
                Topics.MOTION_STARTED,
                {
                    "target": safe_target.as_dict(),
                    "source": command.source,
                    "priority": command.priority.label,
                    "description": command.description,
                },
                source=self.name,
            )
            if result.preempted is not None:
                self._bus.publish(
                    Topics.MOTION_CANCELLED,
                    {"reason": result.reason, "source": result.preempted.source},
                    source=self.name,
                )
        return result

    def move_to(
        self,
        target: HandPose,
        *,
        priority: Priority = Priority.USER_DIRECT,
        force: float = 0.5,
        speed: float = 1.0,
        source: str = "",
        description: str = "",
        preshape: HandPose | None = None,
    ) -> CommandResult:
        """Convenience wrapper around :meth:`submit`."""
        return self.submit(
            MotionCommand(
                target=target,
                priority=priority,
                force=force,
                speed=speed,
                preshape=preshape,
                source=source,
                description=description,
                issued_at=self._clock.monotonic(),
            )
        )

    def cancel(self, reason: str = "cancelled") -> None:
        """Abort motion and hold the current position.

        Holding — rather than opening — is the correct response to a cancel: if
        the user is holding a cup and aborts, dropping it would be worse than
        stopping where they are.
        """
        now = self._clock.monotonic()
        command = self._queue.cancel(now, reason)
        self._trajectory.stop()
        # Hold at the *measured* pose, not the last setpoint: a cancel means
        # "stop here", and continuing towards a stale setpoint would be motion
        # the user explicitly asked to end.
        self._trajectory.sync(self._state.pose)
        self._commanded = self._state.pose
        if command is not None:
            log.info("motion cancelled", reason=reason, source=command.source)
        self._bus.publish(Topics.MOTION_CANCELLED, {"reason": reason}, source=self.name)

    def on_estop_record(self, record) -> None:
        """Listener registered with :class:`~neurogrip.safety.estop.EmergencyStop`.

        This is the *only* path by which a software emergency stop reaches the
        actuators promptly. The decision loop also checks, but that is a 100 Hz
        backstop: a stop must not wait for a scheduler group that may itself be
        what has stalled.

        Three record kinds, three responses:

        * **rehearsal** — acknowledge and do nothing. The acknowledgement is what
          lets :mod:`neurogrip.safety.integrity` prove this wire still exists.
        * **engaged** — cut drive now.
        * **released** — do nothing. Clearing the latch re-arms the drive, which
          is a deliberate act with an owner, not a consequence of a notification.
        """
        if record.rehearsal:
            self._last_estop_rehearsal = self._clock.monotonic()
            return
        if record.engaged:
            self.emergency_stop(record.reason)

    @property
    def last_estop_rehearsal(self) -> float | None:
        """When a rehearsal last reached this controller; ``None`` if never.

        ``None`` rather than ``0.0``: a simulated clock starts at zero, so zero
        is a real timestamp here.
        """
        return self._last_estop_rehearsal

    def emergency_stop(self, reason: str = "emergency stop", *, diagnostic: bool = False) -> None:
        """Cut drive immediately and latch. Safe to call from any thread.

        ``diagnostic`` marks a deliberate stop run by
        :class:`~neurogrip.safety.integrity.EstopSelfCheck` to prove the path
        still works. It changes **nothing** about what happens to the actuators —
        that is the entire point, since a proof test that took a different route
        would prove nothing — and only changes how the event is announced.

        The distinction matters because the alternative is worse than it sounds:
        a routine check logging ``CRITICAL`` and flushing an incident file every
        few hours would bury real stops among hundreds of fake ones and teach
        whoever reads the logs to skip them.
        """
        self._estop = True
        self._estop_diagnostic = diagnostic
        try:
            self._servo.emergency_stop()
        finally:
            self._trajectory.stop()
            self._queue.clear(self._clock.monotonic())
            self._grip.reset()
            if diagnostic:
                log.info("e-stop proof test: cutting drive", reason=reason)
            else:
                log.critical("EMERGENCY STOP", reason=reason)
            self._bus.publish(
                Topics.ESTOP_ENGAGED,
                {"reason": reason, "diagnostic": diagnostic},
                source=self.name,
            )

    def clear_emergency_stop(self) -> None:
        """Release the latch. Requires an explicit user acknowledgement upstream."""
        if not self._estop:
            return
        self._estop = False
        self._estop_diagnostic = False
        self._servo.clear_emergency_stop()
        state = self._servo.read_state()
        self._trajectory.sync(state.pose)
        self._commanded = state.pose
        log.info("emergency stop cleared")
        self._bus.publish(Topics.ESTOP_RELEASED, {}, source=self.name)

    def enable(self) -> None:
        if self._estop:
            raise EmergencyStopActive("cannot enable drive while e-stop is engaged")
        self._servo.enable()

    def disable(self) -> None:
        """De-energise the actuators; the hand goes limp."""
        self._trajectory.stop()
        self._queue.clear(self._clock.monotonic())
        self._servo.disable()

    def home(self) -> None:
        """Run the homing routine and re-synchronise the trajectory generator."""
        if self._estop:
            raise EmergencyStopActive("cannot home while e-stop is engaged")
        self._servo.home()
        state = self._servo.read_state()
        self._trajectory.sync(state.pose)
        self._commanded = state.pose
        self._grip.reset()
        log.info("hand homed")

    def apply_grip(
        self,
        grasp,
        *,
        priority: Priority = Priority.USER_DIRECT,
        fraction: float = 1.0,
        source: str = "",
    ) -> CommandResult:
        """Move to a named grip preset, optionally only partway into it."""
        preset = self._grips.get(grasp)
        target = preset.pose if fraction >= 1.0 else preset.partial(fraction)
        return self.move_to(
            target,
            priority=priority,
            force=preset.force,
            speed=preset.speed,
            preshape=preset.preshape if fraction >= 1.0 else None,
            source=source or "grip-preset",
            description=preset.label,
        )

    def _skip_redundant_preshape(self) -> None:
        """Drop the pre-shape leg when the hand is already open enough for it.

        A pre-shape exists to *widen* the aperture before closing, so it is
        redundant in two situations, and running it anyway would open the hand
        back up:

        * the hand is already at or inside the pre-shape pose (the usual case
          when starting a grasp from an open hand);
        * the hand has already reached the final target, which happens on every
          cycle once a continuously re-issued assisted command has completed.
          Without this check the hand oscillates between the grip and the
          pre-shape instead of holding the grip.
        """
        command = self._queue.active
        if command is None or not self._queue.preshape_pending or command.preshape is None:
            return
        tolerance = self._trajectory.limits.position_tolerance
        measured = self._state.pose
        already_open_enough = all(
            measured[index] <= command.preshape[index] + tolerance
            for index in range(len(measured))
        )
        already_arrived = measured.is_close(command.target, tolerance * 4)
        if already_open_enough or already_arrived:
            self._queue.mark_preshape_done()

    # -- control cycle --------------------------------------------------------

    def tick(self) -> HandState:
        """Execute one control cycle. Called by the scheduler at the control rate."""
        self.cycles += 1
        now = self._clock.monotonic()

        state = self._servo.read_state()
        # Note: the trajectory is *not* re-synced to the measured pose every
        # cycle. Under contact the fingers lag their setpoint by design, and
        # continuously resetting the generator to the measured position would
        # stall the motion the moment an object was touched. Re-sync happens only
        # at discrete events: start, home, cancel and e-stop recovery.

        grip = self._grip.update(
            state, commanded=self._commanded, moving=self._trajectory.active
        )

        if self._estop or state.estop:
            self._estop = True
            # Say what is actually happening. Telling a user their hand is in
            # emergency stop, when in fact it is running a two-hundred
            # millisecond self-check, is how a safety message stops meaning
            # anything.
            activity = "self-check" if self._estop_diagnostic else "emergency stop"
            self._state = self._build_state(state, activity, grip)
            return self._state

        if not state.comms_ok:
            # The link is down. Do not keep writing setpoints into the void; the
            # firmware watchdog will safe the actuators, and the safety monitor
            # will fall back to manual or stop.
            self._state = self._build_state(state, "communication lost", grip)
            return self._state

        timed_out = self._queue.check_timeout(now)
        if timed_out is not None:
            log.warning(
                "motion timed out", source=timed_out.source, description=timed_out.description
            )
            self._trajectory.stop()
            self._bus.publish(
                Topics.MOTION_CANCELLED,
                {"reason": "timeout", "source": timed_out.source},
                source=self.name,
            )

        activity = "idle"
        command = self._queue.active

        if command is not None:
            leg_target = self._queue.current_leg_target or command.target
            if leg_target != self._trajectory.target:
                self._trajectory.retarget(leg_target)

            trajectory = self._trajectory.step(
                now - self._last_write_at if self._last_write_at else 1.0 / 200.0,
                speed_scale=command.speed * self._speed_scale,
            )
            activity = command.description or command.source or "moving"

            if trajectory.complete:
                if self._queue.preshape_pending:
                    # Pre-shape leg finished; continue to the real target.
                    self._queue.mark_preshape_done()
                    self._trajectory.start(
                        command.target,
                        from_pose=trajectory.pose,
                        speed_scale=command.speed * self._speed_scale,
                    )
                else:
                    finished = self._queue.complete(now)
                    if finished is not None:
                        self._bus.publish(
                            Topics.MOTION_COMPLETED,
                            {"source": finished.source, "target": finished.target.as_dict()},
                            source=self.name,
                        )
                    activity = "holding" if grip.holding else "idle"

            # Contact limiting: stop fingers where they met the object.
            self._commanded = self._grip.contact_limited_target(trajectory.pose)
        # With no active command the last commanded pose is held, which is what
        # keeps a grasped object gripped between control decisions.
        commanded = self._commanded

        force = min(self._grip.effective_force, self._force_ceiling)
        self._servo.write_targets(
            commanded,
            speed_scale=(command.speed * self._speed_scale) if command else self._speed_scale,
            force=force,
        )
        self.writes += 1
        self._last_write_at = now

        self._state = self._build_state(state, activity, grip, commanded=commanded, force=force)
        if self._publish_state:
            self._bus.publish(Topics.HAND_STATE, self._state, source=self.name)
        if grip.slipping:
            self._bus.publish(
                Topics.GRIP_SLIP, {"events": grip.slip_events, "force": force}, source=self.name
            )
        elif grip.holding and not (self._state.grip and self._state.grip.holding):
            self._bus.publish(
                Topics.GRIP_CONTACT,
                {"fingers": [f.name for f in grip.contacts], "force": force},
                source=self.name,
            )
        return self._state

    # -- helpers --------------------------------------------------------------

    def _build_state(
        self,
        servo_state: ServoBusState,
        activity: str,
        grip: GripState | None = None,
        *,
        commanded: HandPose | None = None,
        force: float | None = None,
    ) -> HandState:
        command = self._queue.active
        return HandState(
            pose=servo_state.pose,
            commanded=commanded if commanded is not None else self._state.commanded,
            goal=command.target if command else None,
            velocities=tuple(f.velocity for f in servo_state.fingers),
            grip=grip,
            moving=servo_state.moving or self._trajectory.active,
            enabled=servo_state.enabled,
            homed=servo_state.homed,
            estop=self._estop or servo_state.estop,
            comms_ok=servo_state.comms_ok,
            force=force if force is not None else self._state.force,
            currents=tuple(f.current_ma for f in servo_state.fingers),
            temperature_c=servo_state.max_temperature_c,
            bus_voltage_v=servo_state.bus_voltage_v,
            timestamp=servo_state.timestamp or self._clock.monotonic(),
            activity=activity,
            faults=servo_state.faults,
        )

    def health(self) -> HealthReport:
        state = self._state
        if not self.running:
            return HealthReport.offline(self.name)
        if state.estop:
            return HealthReport.failed(self.name, "emergency stop engaged")
        if not state.comms_ok:
            return HealthReport.failed(self.name, "no communication with the motor controller")
        if state.faults:
            return HealthReport.degraded(self.name, ", ".join(state.faults))
        if state.temperature_c > self._servo_limits.max_temperature_c * 0.9:
            return HealthReport.degraded(
                self.name, f"motors running hot ({state.temperature_c:.0f} °C)"
            )
        return HealthReport(
            name=self.name,
            status=HealthStatus.OK,
            metrics={
                "cycles": self.cycles,
                "writes": self.writes,
                "current_ma": state.total_current_ma,
                "holding": state.holding,
                **self._queue.stats(),
            },
        )

    def finger_state(self, finger: Finger) -> float:
        """Measured closure of one finger."""
        return self._state.pose[finger]
