"""Planner that consumes 6-DoF grasp poses (AnyGrasp and similar).

AnyGrasp predicts full SE(3) gripper poses from a point cloud. That is more
information than HGGD-MCU's planar grasps, and adapting it well means confronting
a constraint that does not exist for the robot arms these models were designed
for:

**This hand has no powered wrist.** Five servos, one per finger. The approach
direction is whatever the user's arm is doing, and the software cannot change it.

A 6-DoF planner assumes the manipulator can be placed in any pose it proposes. On
a prosthesis that assumption is false, and taking it at face value produces the
worst possible behaviour: pre-shaping for a grasp geometry the hand will never be
in, so the fingers close on nothing, or on the wrong part of the object. So this
adapter:

* **rejects** candidates whose approach vector is far from where the hand is
  actually pointing — they belong to a grasp the user is not making;
* **attenuates confidence** smoothly as the misalignment grows, rather than using
  a hard cut-off, so a slightly-off approach degrades into a more conservative
  grip instead of vanishing;
* **falls through** — returning ``None`` rather than a poor plan lets the
  composite chain reach the affordance-based heuristic, which does not depend on
  approach geometry at all.

Alignment is judged against the *camera* axis, because the camera is hand-mounted:
what it points at is what the hand points at. A wrist-mounted IMU would give a
better estimate, and :attr:`AnyGraspPlanner.hand_forward` is where it would be
injected — the interface already takes the direction as a parameter rather than
assuming it.

The AnyGrasp runtime itself is not bundled: it needs a licence, CUDA, and a depth
sensor this hand does not have. :mod:`neurogrip.vision.backends.anygrasp` holds
the backend that would produce these candidates, and reports clearly when it
cannot run. This planner is useful without it — any backend that fills in
``approach_vector`` gets the same treatment.
"""

from __future__ import annotations

import math

from ...control.grips import GripLibrary
from ...control.kinematics import HandKinematics
from ...core.logging import get_logger
from ...core.types import GraspType, HandPose, clamp
from ...vision.types import GraspApproach, GraspCandidate
from ..objects import AffordanceDatabase
from .base import GraspContext, GraspPlan, PlanSource

__all__ = ["AnyGraspPlanner"]

log = get_logger(__name__)


class AnyGraspPlanner:
    """Adapts 6-DoF grasp poses to a wristless five-finger hand."""

    #: Candidate score below which this planner declines, leaving the chain to
    #: fall through to affordance reasoning.
    MIN_SCORE = 0.35

    #: Angle between the predicted approach and where the hand points, beyond
    #: which the grasp is treated as belonging to a different reach entirely.
    #: 60° is roughly the point at which a user would have to rotate their
    #: forearm rather than simply adjust it.
    MAX_MISALIGNMENT_DEG = 60.0

    #: Misalignment below which no confidence penalty applies at all. Users do
    #: not hold their arm perfectly square to an object, and penalising normal
    #: variation would make assistance feel arbitrary.
    FREE_MISALIGNMENT_DEG = 18.0

    def __init__(
        self,
        grips: GripLibrary,
        affordances: AffordanceDatabase,
        kinematics: HandKinematics | None = None,
        *,
        hand_forward: tuple[float, float, float] = (0.0, 0.0, 1.0),
    ) -> None:
        self._grips = grips
        self._affordances = affordances
        self._kinematics = kinematics or HandKinematics()
        #: Direction the hand points, in camera coordinates. The camera is
        #: hand-mounted and boresighted, so +Z. Replace with a live estimate if
        #: an IMU is ever fitted.
        self.hand_forward = _normalise(hand_forward)

    @property
    def name(self) -> str:
        return "anygrasp"

    def plan(self, context: GraspContext) -> GraspPlan | None:
        vision = context.vision
        if vision is None or not vision.grasps:
            return None

        selected = self._select(vision.grasps)
        if selected is None:
            return None
        candidate, misalignment_deg = selected

        reasons = [
            f"6-DoF grasp at ({candidate.center_x:.2f}, {candidate.center_y:.2f}), "
            f"score {candidate.quality:.2f}"
        ]
        if candidate.approach_vector is not None:
            reasons.append(f"approach {misalignment_deg:.0f}° from the hand axis")

        grasp = self._grasp_for(candidate, reasons)
        preset = self._grips.get(grasp)

        preshape = preset.preshape
        if candidate.width_m is not None and candidate.width_m > 0:
            clearance = min(self._kinematics.max_aperture_m, candidate.width_m * 1.3 + 0.012)
            preshape = HandPose.uniform(self._kinematics.closure_for_aperture(clearance))
            reasons.append(f"predicted opening {candidate.width_m * 100:.1f} cm")

        label = candidate.label or context.object_label
        affordance = self._affordances.get(label)
        force = affordance.force_for(context.force_ceiling)
        if affordance.fragile:
            force = min(force, 0.35)
            reasons.append(f"{affordance.label} is fragile — force capped")

        effort = clamp(0.55 + 0.45 * context.intent.strength)
        force = clamp(force * effort)

        alignment = self._alignment_factor(misalignment_deg)
        if alignment < 1.0:
            reasons.append(f"approach mismatch — confidence scaled by {alignment:.2f}")

        offset = math.hypot(candidate.center_x - 0.5, candidate.center_y - 0.5)
        centrality = clamp(1.0 - offset / 0.38)
        confidence = clamp(
            (
                candidate.quality * 0.50
                + centrality * 0.20
                + clamp(context.intent.confidence) * 0.30
            )
            * alignment
        )

        target_pose, notes = self._kinematics.enforce_limits(preset.pose)
        reasons.extend(notes)

        return GraspPlan(
            grasp=grasp,
            target=target_pose,
            preshape=preshape,
            force=force,
            speed=min(context.speed_ceiling, preset.speed * affordance.speed_scale),
            confidence=confidence,
            source=PlanSource.LEARNED,
            label=affordance.label if affordance.label != "unknown" else label,
            candidate=candidate,
            reasons=tuple(reasons),
        )

    # -- internals ------------------------------------------------------------

    def _select(
        self, candidates: tuple[GraspCandidate, ...]
    ) -> tuple[GraspCandidate, float] | None:
        """Best reachable candidate, with its misalignment in degrees."""
        best: tuple[GraspCandidate, float] | None = None
        best_score = 0.0
        for candidate in candidates:
            if candidate.quality < self.MIN_SCORE:
                continue
            misalignment = self._misalignment_deg(candidate)
            if misalignment > self.MAX_MISALIGNMENT_DEG:
                # Not a rejection of the grasp — a recognition that it is a grasp
                # for a different arm position than the one the user is in.
                continue
            offset = math.hypot(candidate.center_x - 0.5, candidate.center_y - 0.5)
            score = candidate.quality * self._alignment_factor(misalignment) * (1.0 - offset)
            if score > best_score:
                best_score = score
                best = (candidate, misalignment)
        return best

    def _misalignment_deg(self, candidate: GraspCandidate) -> float:
        """Angle between the predicted approach and where the hand points.

        A candidate with no approach vector is planar, not forward-facing, so it
        is reported as perfectly aligned: this planner must not penalise a
        backend for producing less information than AnyGrasp does.
        """
        vector = candidate.approach_vector
        if vector is None:
            return 0.0
        approach = _normalise(vector)
        dot = sum(a * b for a, b in zip(approach, self.hand_forward))
        return math.degrees(math.acos(max(-1.0, min(1.0, dot))))

    def _alignment_factor(self, misalignment_deg: float) -> float:
        """Confidence multiplier from misalignment: 1.0 when aligned, 0 at the limit."""
        if misalignment_deg <= self.FREE_MISALIGNMENT_DEG:
            return 1.0
        span = self.MAX_MISALIGNMENT_DEG - self.FREE_MISALIGNMENT_DEG
        if span <= 0:
            return 0.0
        return clamp(1.0 - (misalignment_deg - self.FREE_MISALIGNMENT_DEG) / span)

    def _grasp_for(self, candidate: GraspCandidate, reasons: list[str]) -> GraspType:
        """Map a 6-DoF pose onto a five-finger preset.

        Uses the vertical component of the approach vector where available: a
        grasp approached from above wants a different preset from one approached
        from the side, and unlike the planar case that is directly measurable
        rather than inferred from a 2D angle.
        """
        width_m = candidate.width_m
        vector = candidate.approach_vector
        top_down = candidate.approach is GraspApproach.TOP_DOWN
        if vector is not None:
            # +Y is down in camera coordinates, so a positive Y component means
            # the hand comes down onto the object.
            top_down = _normalise(vector)[1] > 0.5

        if width_m is None:
            if candidate.width < 0.10:
                reasons.append("narrow opening → precision pinch")
                return GraspType.PRECISION_PINCH
            if candidate.width < 0.22:
                reasons.append("moderate opening → tripod")
                return GraspType.TRIPOD
            reasons.append("wide opening → cylindrical wrap")
            return GraspType.CYLINDRICAL

        if width_m <= 0.02:
            reasons.append(f"{width_m * 100:.1f} cm opening → precision pinch")
            return GraspType.PRECISION_PINCH
        if width_m <= 0.045:
            grasp = GraspType.TRIPOD if top_down else GraspType.LATERAL_KEY
            reasons.append(f"{width_m * 100:.1f} cm opening, from {'above' if top_down else 'the side'}")
            return grasp
        if width_m <= self._kinematics.max_aperture_m * 0.85:
            grasp = GraspType.SPHERICAL if top_down else GraspType.CYLINDRICAL
            reasons.append(f"{width_m * 100:.1f} cm opening, from {'above' if top_down else 'the side'}")
            return grasp

        reasons.append(f"{width_m * 100:.1f} cm exceeds the aperture → hook grip")
        return GraspType.HOOK


def _normalise(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    """Unit vector, or +Z for a degenerate input."""
    magnitude = math.sqrt(sum(component * component for component in vector))
    if magnitude < 1e-9:
        return (0.0, 0.0, 1.0)
    return (vector[0] / magnitude, vector[1] / magnitude, vector[2] / magnitude)
