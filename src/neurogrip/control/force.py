"""Adaptive grip force from motor-current feedback.

The hand has no fingertip pressure sensors, so contact and grip force are
inferred from servo current — the same trick most commercial myoelectric hands
use. The signal is noisy but the physics is clean: a finger moving freely draws
current proportional to speed; a finger pressed against an object draws current
proportional to force.

Two behaviours are built on that:

* **Contact detection.** A sustained current rise while the finger is not moving
  means it has met something. That stops the finger *there* rather than driving
  it to the commanded closure, which is what prevents a bottle from being
  crushed and a servo from stalling.
* **Slip response.** If the held object starts moving relative to the fingers
  (detected as position drift under load) the grip tightens by a bounded
  increment. Bounded, because an unbounded "grip harder until it stops slipping"
  loop is precisely how you crush the thing you are holding.

Force is regulated per finger. Fingers reach an object at different moments, and
a single global force would over-squeeze the ones that arrived first.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core.clock import Clock
from ..core.ringbuffer import RingBuffer
from ..core.types import FINGER_COUNT, Finger, HandPose, clamp
from ..hal.servo.base import ServoBusState

__all__ = ["AdaptiveGripController", "ContactState", "GripSettings", "GripState"]


@dataclass(frozen=True, slots=True)
class GripSettings:
    """Tuning for contact detection and force regulation."""

    #: Current above the static holding baseline that indicates contact, in mA.
    #: Compared only while the finger is stalled, so the velocity-dependent part
    #: of the current has already dropped out and this is a force threshold.
    contact_current_ma: float = 150.0
    #: Static current drawn by an energised, stationary, unloaded finger, in mA.
    #: Used as the initial baseline before one has been measured.
    holding_current_ma: float = 60.0
    #: Speed below which a finger counts as "not moving", closure units/s.
    stall_velocity: float = 0.03
    #: Seconds the contact condition must hold before it is believed.
    contact_dwell_s: float = 0.06
    #: Compliant objects (foam, fruit, a paper cup) deform instead of resisting,
    #: so they raise the current much less than a rigid object does. A smaller
    #: rise held for longer is the signature of soft contact, and missing it
    #: would mean squeezing exactly the objects that least tolerate it.
    soft_contact_current_ma: float = 45.0
    soft_contact_dwell_s: float = 0.30
    #: Position drift that indicates the object is slipping, in closure units.
    slip_threshold: float = 0.025
    #: Force increment applied per slip event, in ``[0, 1]``.
    slip_increment: float = 0.08
    #: Hard ceiling on the total force increase from slip response.
    max_slip_boost: float = 0.25
    #: Seconds between slip responses, so one slip is not counted repeatedly.
    slip_cooldown_s: float = 0.4
    #: Force applied once contact is confirmed, as a fraction of the commanded
    #: ceiling. Below 1.0 so the initial contact is gentle.
    contact_force_fraction: float = 0.75


@dataclass(slots=True)
class ContactState:
    """Per-finger contact tracking."""

    finger: Finger
    in_contact: bool = False
    #: Closure at which contact was first detected.
    contact_position: float = 0.0
    #: Static holding current with the finger free, for comparison.
    baseline_ma: float = 60.0
    #: Current at the moment of contact.
    contact_current_ma: float = 0.0
    #: When the contact condition first became true (for dwell); ``None`` when
    #: the condition is not currently met. Not ``0.0``: a simulated clock starts
    #: at zero, and a zero sentinel would swallow the very first contact.
    candidate_since: float | None = None
    #: Applied force for this finger, ``[0, 1]``.
    force: float = 0.0
    history: RingBuffer = field(default_factory=lambda: RingBuffer(40))

    @property
    def pressure_ma(self) -> float:
        """Current above baseline — a proxy for contact force."""
        return max(0.0, self.contact_current_ma - self.baseline_ma)


@dataclass(frozen=True, slots=True)
class GripState:
    """Whole-hand grip status, published for the UI and the safety layer."""

    holding: bool
    #: Fingers currently in contact.
    contacts: tuple[Finger, ...]
    #: Force ceiling currently applied, ``[0, 1]``.
    force: float
    #: Contact positions, for the "grip closed at" readout.
    contact_pose: HandPose | None = None
    slipping: bool = False
    slip_events: int = 0
    #: Estimated grip force in newtons, for the maximum-force safety rule.
    estimated_force_n: float = 0.0

    @property
    def contact_count(self) -> int:
        return len(self.contacts)


class AdaptiveGripController:
    """Detects contact and regulates grip force from current feedback."""

    def __init__(
        self,
        clock: Clock,
        settings: GripSettings | None = None,
        *,
        max_force: float = 0.85,
    ) -> None:
        self._clock = clock
        self._settings = settings or GripSettings()
        self._max_force = max_force
        self._contacts = {
            finger: ContactState(
                finger=finger, baseline_ma=(settings or GripSettings()).holding_current_ma
            )
            for finger in Finger
        }
        self._commanded_force = 0.5
        self._slip_boost = 0.0
        self._last_slip_at = -1e18
        self._slip_events = 0
        self._holding = False
        self._grip_pose: HandPose | None = None

    # -- configuration --------------------------------------------------------

    @property
    def settings(self) -> GripSettings:
        return self._settings

    def set_max_force(self, value: float) -> None:
        """Update the hard ceiling (mode change or safety derating)."""
        self._max_force = clamp(value)

    def set_commanded_force(self, value: float) -> None:
        """Set the force the current grasp plan asked for."""
        self._commanded_force = clamp(value)

    def reset(self) -> None:
        """Clear all contact state (hand opened, e-stop, mode change)."""
        for state in self._contacts.values():
            state.in_contact = False
            state.force = 0.0
            state.candidate_since = None
            state.baseline_ma = self._settings.holding_current_ma
            state.history.clear()
        self._slip_boost = 0.0
        self._holding = False
        self._grip_pose = None

    # -- update ---------------------------------------------------------------

    def update(self, state: ServoBusState, *, commanded: HandPose, moving: bool) -> GripState:
        """Process one telemetry frame and return the grip status."""
        now = state.timestamp or self._clock.monotonic()
        settings = self._settings
        contacts: list[Finger] = []
        slipping = False

        for finger_state in state.fingers:
            tracker = self._contacts[finger_state.finger]
            tracker.history.append(float(finger_state.current_ma))

            target = commanded[finger_state.finger]
            short_of_target = target - finger_state.position > 0.02
            stalled = abs(finger_state.velocity) <= settings.stall_velocity

            if stalled and not tracker.in_contact and not short_of_target:
                # Settled at the commanded position with nothing in the way: this
                # is the unloaded holding current, which is the only meaningful
                # baseline. Sampling it while the finger is moving would fold in
                # the velocity-proportional term and mask real contact forces.
                tracker.baseline_ma = min(tracker.baseline_ma, tracker.history.median())

            excess = finger_state.current_ma - tracker.baseline_ma
            over_current = excess >= settings.contact_current_ma
            soft_contact = excess >= settings.soft_contact_current_ma

            if not tracker.in_contact:
                if soft_contact and stalled and short_of_target:
                    if tracker.candidate_since is None:
                        tracker.candidate_since = now
                    required = (
                        settings.contact_dwell_s
                        if over_current
                        else settings.soft_contact_dwell_s
                    )
                    if now - tracker.candidate_since >= required:
                        tracker.in_contact = True
                        tracker.contact_position = finger_state.position
                        tracker.contact_current_ma = float(finger_state.current_ma)
                        tracker.force = clamp(
                            self._commanded_force * settings.contact_force_fraction
                        )
                else:
                    tracker.candidate_since = None
            else:
                # Release detection: current has fallen back towards baseline.
                if excess < settings.soft_contact_current_ma * 0.5:
                    tracker.in_contact = False
                    tracker.candidate_since = None
                    tracker.force = 0.0
                else:
                    # Keep the live current so the force estimate tracks reality.
                    tracker.contact_current_ma = float(finger_state.current_ma)
                    drift = finger_state.position - tracker.contact_position
                    if (
                        drift > settings.slip_threshold
                        and now - self._last_slip_at > settings.slip_cooldown_s
                    ):
                        # The finger has advanced past where it made contact
                        # while still loaded: the object is slipping through.
                        slipping = True
                        self._slip_events += 1
                        self._last_slip_at = now
                        self._slip_boost = min(
                            settings.max_slip_boost, self._slip_boost + settings.slip_increment
                        )
                        tracker.contact_position = finger_state.position
                    tracker.force = clamp(
                        self._commanded_force * settings.contact_force_fraction + self._slip_boost
                    )

            if tracker.in_contact:
                contacts.append(finger_state.finger)

        holding = len(contacts) >= 2
        if holding and self._grip_pose is None:
            self._grip_pose = HandPose.from_iterable(
                self._contacts[f].contact_position if self._contacts[f].in_contact else state.pose[f]
                for f in Finger
            )
        elif not holding:
            self._grip_pose = None
            if not contacts:
                self._slip_boost = 0.0
        self._holding = holding

        return GripState(
            holding=holding,
            contacts=tuple(contacts),
            force=self.effective_force,
            contact_pose=self._grip_pose,
            slipping=slipping,
            slip_events=self._slip_events,
            estimated_force_n=self._estimate_force_n(),
        )

    # -- outputs --------------------------------------------------------------

    @property
    def effective_force(self) -> float:
        """Force to command this cycle, including any slip boost, clamped."""
        return clamp(min(self._max_force, self._commanded_force + self._slip_boost))

    @property
    def holding(self) -> bool:
        return self._holding

    def contact_limited_target(self, requested: HandPose) -> HandPose:
        """Clip a requested pose so fingers stop where they met the object.

        Once a finger is in contact, continuing to command it further closed only
        raises current and squeezes harder. Holding it at the contact position and
        letting *force* do the work is both gentler and more stable.
        """
        values = list(requested.values)
        for index in range(FINGER_COUNT):
            tracker = self._contacts[Finger(index)]
            if tracker.in_contact:
                values[index] = min(values[index], tracker.contact_position + 0.02)
        return HandPose(tuple(values))  # type: ignore[arg-type]

    def contact_state(self, finger: Finger) -> ContactState:
        return self._contacts[finger]

    def _estimate_force_n(self) -> float:
        """Rough total grip force, for the safety rule.

        Uses a linear current-to-force constant per finger. TODO(hardware):
        calibrate this against a force gauge per unit; the constant below is from
        the reference hand's datasheet torque figure and the horn radius, not
        from measurement.
        """
        newtons_per_ma = 0.018
        return sum(state.pressure_ma * newtons_per_ma for state in self._contacts.values())
