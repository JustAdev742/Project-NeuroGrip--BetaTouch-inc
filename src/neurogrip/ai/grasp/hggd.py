"""Planner that consumes HGGD-MCU grasp candidates.

The network predicts *where* and *at what angle and opening* to grasp. This
adapter turns that into hand configuration: which of the five-finger presets best
realises the predicted gripper geometry, at what aperture, and with what force.

The mapping from a parallel-jaw grasp (what HGGD predicts) to an anthropomorphic
hand (what we have) is the interesting part, and it is a deliberate
approximation:

* **opening → preset.** A narrow predicted opening implies a pinch; a wide one a
  wrap. The thresholds come from the hand's geometry, not from the network.
* **angle → approach.** A grasp axis near horizontal means closing across a
  vertical object (a wrap); near vertical means reaching over it (a top-down
  pinch or tripod).
* **quality → confidence,** attenuated when the grasp point is far from the image
  centre, because the hand-mounted camera points where the user is reaching and
  an off-centre grasp is probably on a different object.

When the network is running its classical fallback (no weights), candidates carry
``label == "unknown"``; the composite chain then prefers the heuristic planner,
which at least has an affordance table to reason with.
"""

from __future__ import annotations

import math

from ...control.grips import GripLibrary
from ...control.kinematics import HandKinematics
from ...core.logging import get_logger
from ...core.types import GraspType, clamp
from ...vision.types import GraspApproach, GraspCandidate
from ..objects import AffordanceDatabase
from .base import GraspContext, GraspPlan, PlanSource

__all__ = ["HggdGraspPlanner"]

log = get_logger(__name__)


class HggdGraspPlanner:
    """Adapts network-predicted grasps to the five-finger hand."""

    #: Candidate quality below which this planner declines to plan, so the
    #: composite chain can fall through to affordance reasoning.
    MIN_QUALITY = 0.40
    #: Normalised distance from the image centre beyond which a candidate is
    #: assumed to belong to an object the user is not reaching for.
    MAX_OFFSET = 0.38

    def __init__(
        self,
        grips: GripLibrary,
        affordances: AffordanceDatabase,
        kinematics: HandKinematics | None = None,
    ) -> None:
        self._grips = grips
        self._affordances = affordances
        self._kinematics = kinematics or HandKinematics()

    @property
    def name(self) -> str:
        return "hggd"

    def plan(self, context: GraspContext) -> GraspPlan | None:
        vision = context.vision
        if vision is None or not vision.grasps:
            return None

        candidate = self._select(vision.grasps)
        if candidate is None:
            return None

        reasons = [
            f"HGGD-MCU grasp at ({candidate.center_x:.2f}, {candidate.center_y:.2f}), "
            f"{candidate.angle_degrees:.0f}°, quality {candidate.quality:.2f}"
        ]

        width_m = candidate.width_m
        grasp = self._grasp_for(candidate, width_m, reasons)
        preset = self._grips.get(grasp)

        # Pre-shape to the predicted opening plus clearance.
        preshape = preset.preshape
        if width_m is not None and width_m > 0:
            clearance = min(self._kinematics.max_aperture_m, width_m * 1.3 + 0.012)
            from ...core.types import HandPose

            preshape = HandPose.uniform(self._kinematics.closure_for_aperture(clearance))
            reasons.append(f"predicted opening {width_m * 100:.1f} cm")

        # The network does not know how fragile an object is; the affordance
        # table does. Force always takes the lower of the two ceilings.
        label = candidate.label or context.object_label
        affordance = self._affordances.get(label)
        force = affordance.force_for(context.force_ceiling)
        if affordance.fragile:
            force = min(force, 0.35)
            reasons.append(f"{affordance.label} is fragile — force capped")

        effort = clamp(0.55 + 0.45 * context.intent.strength)
        force = clamp(force * effort)

        offset = math.hypot(candidate.center_x - 0.5, candidate.center_y - 0.5)
        centrality = clamp(1.0 - offset / self.MAX_OFFSET)
        confidence = clamp(
            candidate.quality * 0.55 + centrality * 0.25 + clamp(context.intent.confidence) * 0.2
        )
        if centrality < 0.5:
            reasons.append(f"grasp point is off-centre ({offset:.2f}) — confidence reduced")

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

    def _select(self, candidates: tuple[GraspCandidate, ...]) -> GraspCandidate | None:
        """Pick the candidate the user is most plausibly aiming at."""
        best: GraspCandidate | None = None
        best_score = 0.0
        for candidate in candidates:
            if candidate.quality < self.MIN_QUALITY:
                continue
            offset = math.hypot(candidate.center_x - 0.5, candidate.center_y - 0.5)
            if offset > self.MAX_OFFSET:
                continue
            score = candidate.quality * (1.0 - offset / (self.MAX_OFFSET * 2))
            if score > best_score:
                best_score = score
                best = candidate
        return best

    def _grasp_for(
        self, candidate: GraspCandidate, width_m: float | None, reasons: list[str]
    ) -> GraspType:
        """Map predicted gripper geometry onto a five-finger preset."""
        top_down = candidate.approach is GraspApproach.TOP_DOWN

        if width_m is None:
            # No metric scale: fall back to the normalised image width, which is
            # still monotonic in object size at a fixed working distance.
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
            reasons.append(f"{width_m * 100:.1f} cm opening, {candidate.approach.value} → {grasp.label}")
            return grasp
        if width_m <= self._kinematics.max_aperture_m * 0.85:
            grasp = GraspType.SPHERICAL if top_down else GraspType.CYLINDRICAL
            reasons.append(f"{width_m * 100:.1f} cm opening, {candidate.approach.value} → {grasp.label}")
            return grasp

        reasons.append(f"{width_m * 100:.1f} cm exceeds the aperture → hook grip")
        return GraspType.HOOK
