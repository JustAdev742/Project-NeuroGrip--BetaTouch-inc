"""Tendon kinematics and self-collision constraints.

Converts between the three coordinate systems the hand works in:

1. **Closure** — normalised ``[0, 1]`` per finger. Everything above the HAL
   speaks this.
2. **Tendon travel** — millimetres of fishing line pulled through the routing
   channels. This is what actually determines finger position.
3. **Servo angle / pulse width** — what the actuator is commanded with.

The relationship between closure and tendon travel is *not* linear. A
tendon-driven finger with two revolute joints and a single antagonistic return
spring flexes progressively: the proximal joint takes up most of the early
travel, the distal joints the later travel. Modelling that with a mild
polynomial gives noticeably more natural motion than a straight line, and makes
the aperture-to-closure inversion (used to pre-shape the hand around an object)
meaningfully more accurate.

Also enforced here: **self-collision limits**. Fingers on a real hand interfere,
and the thumb can be driven into the palm hard enough to stall a servo or shear
a tendon. :meth:`HandKinematics.enforce_limits` clips any requested pose into the
mechanically reachable set before it reaches the actuators.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core.types import FINGER_COUNT, Finger, HandPose, clamp

__all__ = ["DEFAULT_GEOMETRY", "FingerGeometry", "HandKinematics"]


@dataclass(frozen=True, slots=True)
class FingerGeometry:
    """Mechanical description of one tendon-driven finger."""

    finger: Finger
    #: Total tendon travel from fully open to fully closed, in millimetres.
    travel_mm: float = 24.0
    #: Radius of the servo horn the tendon winds onto, in millimetres.
    horn_radius_mm: float = 8.0
    #: Fingertip distance from the palm centre when open, in metres.
    reach_m: float = 0.09
    #: Nonlinearity exponent of closure → travel; 1.0 is linear, >1 progressive.
    curve_exponent: float = 1.35
    #: Lowest closure the finger may be commanded to (mechanical hard stop).
    min_closure: float = 0.0
    #: Highest closure before the fingertip contacts the palm.
    max_closure: float = 1.0
    #: Force the tendon can transmit at full drive current, in newtons.
    max_force_n: float = 12.0


#: Geometry for the reference hand. The thumb is shorter-travel and limited to
#: 0.92 closure because at full travel it collides with the index finger's
#: proximal phalanx — see ``docs/hardware.md``.
DEFAULT_GEOMETRY: tuple[FingerGeometry, ...] = (
    FingerGeometry(Finger.THUMB, travel_mm=18.0, reach_m=0.062, max_closure=0.92, max_force_n=14.0),
    FingerGeometry(Finger.INDEX, travel_mm=26.0, reach_m=0.095, max_force_n=12.0),
    FingerGeometry(Finger.MIDDLE, travel_mm=27.0, reach_m=0.098, max_force_n=12.0),
    FingerGeometry(Finger.RING, travel_mm=25.0, reach_m=0.092, max_force_n=10.0),
    FingerGeometry(Finger.PINKY, travel_mm=22.0, reach_m=0.078, max_force_n=8.0),
)


@dataclass(frozen=True, slots=True)
class CollisionRule:
    """A constraint coupling two fingers.

    ``if fingers[a] >= a_threshold then fingers[b] <= b_limit``. Simple, explicit
    and inspectable — which is what you want for a rule that exists to stop the
    mechanism destroying itself.
    """

    a: Finger
    b: Finger
    a_threshold: float
    b_limit: float
    reason: str = ""


class HandKinematics:
    """Closure ↔ travel ↔ aperture conversions plus mechanical limit enforcement."""

    #: Interference constraints for the reference hand.
    COLLISION_RULES: tuple[CollisionRule, ...] = (
        CollisionRule(
            Finger.THUMB,
            Finger.INDEX,
            a_threshold=0.80,
            b_limit=0.95,
            reason="thumb tip fouls the index proximal phalanx",
        ),
        CollisionRule(
            Finger.INDEX,
            Finger.THUMB,
            a_threshold=0.90,
            b_limit=0.85,
            reason="index sweeps through the thumb opposition arc",
        ),
    )

    def __init__(
        self,
        geometry: tuple[FingerGeometry, ...] = DEFAULT_GEOMETRY,
        *,
        max_aperture_m: float = 0.11,
        collision_rules: tuple[CollisionRule, ...] | None = None,
    ) -> None:
        if len(geometry) != FINGER_COUNT:
            raise ValueError(f"expected {FINGER_COUNT} finger geometries")
        self._geometry = {g.finger: g for g in geometry}
        #: Fingertip-to-thumb-tip span with the hand fully open, in metres.
        self._max_aperture = max_aperture_m
        self._rules = collision_rules if collision_rules is not None else self.COLLISION_RULES

    def geometry(self, finger: Finger) -> FingerGeometry:
        return self._geometry[finger]

    @property
    def max_aperture_m(self) -> float:
        return self._max_aperture

    # -- closure <-> travel ---------------------------------------------------

    def closure_to_travel_mm(self, finger: Finger, closure: float) -> float:
        """Tendon travel required for a given closure."""
        geometry = self._geometry[finger]
        return geometry.travel_mm * (clamp(closure) ** geometry.curve_exponent)

    def travel_to_closure(self, finger: Finger, travel_mm: float) -> float:
        """Inverse of :meth:`closure_to_travel_mm`."""
        geometry = self._geometry[finger]
        if geometry.travel_mm <= 0:
            return 0.0
        ratio = clamp(travel_mm / geometry.travel_mm)
        return ratio ** (1.0 / geometry.curve_exponent)

    def closure_to_servo_degrees(self, finger: Finger, closure: float) -> float:
        """Servo rotation for a closure, from tendon travel and horn radius."""
        import math

        travel = self.closure_to_travel_mm(finger, closure)
        geometry = self._geometry[finger]
        return math.degrees(travel / max(1e-6, geometry.horn_radius_mm))

    # -- aperture -------------------------------------------------------------

    def aperture_m(self, pose: HandPose) -> float:
        """Opening between the thumb and the finger group, in metres.

        Uses the index/middle pair because those are the digits that actually
        oppose the thumb in a pinch or a cylindrical wrap.
        """
        opposition = max(pose[Finger.INDEX], pose[Finger.MIDDLE])
        closed = max(opposition, pose[Finger.THUMB])
        return self._max_aperture * (1.0 - clamp(closed))

    def closure_for_aperture(self, aperture_m: float) -> float:
        """Uniform closure that produces a given aperture.

        Used to pre-shape the hand: given an object's measured width, open just
        wide enough to clear it rather than opening fully, which is both faster
        and less startling to bystanders.
        """
        if self._max_aperture <= 0:
            return 0.0
        return clamp(1.0 - aperture_m / self._max_aperture)

    def grip_force_n(self, finger: Finger, drive: float) -> float:
        """Approximate fingertip force for a normalised drive level."""
        return self._geometry[finger].max_force_n * clamp(drive)

    def total_grip_force_n(self, pose: HandPose, drive: float) -> float:
        """Sum of fingertip forces for engaged fingers.

        Only fingers past 20 % closure are counted: a barely-flexed digit is not
        contributing to the grip. This feeds the maximum-grip-force safety rule.
        """
        return sum(
            self.grip_force_n(finger, drive)
            for finger in Finger
            if pose[finger] > 0.2
        )

    # -- limits ---------------------------------------------------------------

    def enforce_limits(self, pose: HandPose) -> tuple[HandPose, tuple[str, ...]]:
        """Clip a pose into the mechanically safe set.

        Returns the corrected pose and a description of every applied
        correction, so the UI and the black-box log can show that the requested
        pose was modified and why.
        """
        values = list(pose.values)
        notes: list[str] = []

        for finger in Finger:
            geometry = self._geometry[finger]
            requested = values[int(finger)]
            limited = min(max(requested, geometry.min_closure), geometry.max_closure)
            if abs(limited - requested) > 1e-6:
                notes.append(
                    f"{finger.label} clipped {requested:.2f}→{limited:.2f} (mechanical limit)"
                )
            values[int(finger)] = limited

        for rule in self._rules:
            if values[int(rule.a)] >= rule.a_threshold and values[int(rule.b)] > rule.b_limit:
                notes.append(
                    f"{rule.b.label} limited to {rule.b_limit:.2f}: {rule.reason}"
                )
                values[int(rule.b)] = rule.b_limit

        return HandPose(tuple(values)), tuple(notes)  # type: ignore[arg-type]

    def is_reachable(self, pose: HandPose) -> bool:
        """Whether ``pose`` needs no correction."""
        corrected, notes = self.enforce_limits(pose)
        return not notes and corrected.is_close(pose, 1e-6)
