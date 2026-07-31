"""Affordance-driven grasp planner.

Given an object label and its measured size, choose the grasp a person would use.
Deterministic, inspectable, and it works with nothing more than a bounding box —
which is why it is the fallback whenever the learned grasp head is unavailable,
degraded or unconfident.

Decision procedure:

1. Look up the object's :class:`~neurogrip.ai.objects.Affordance`.
2. Walk its preferred grasps in order and take the first that is *feasible*
   given the object's estimated width and the hand's aperture.
3. Pre-shape the hand to just clear the object rather than opening fully.
4. Scale force by the object's ceiling, its fragility and the user's own effort.
5. Record why, at every step.

Step 5 is not optional: a user who cannot find out why their hand chose a pinch
instead of a wrap will stop trusting it, and a prosthesis that is not trusted
gets left in a drawer.
"""

from __future__ import annotations

from ...control.grips import GripLibrary
from ...control.kinematics import HandKinematics
from ...core.logging import get_logger
from ...core.types import GraspType, HandPose, clamp
from ...vision.depth import OBJECT_SIZE_PRIORS
from ..objects import Affordance, AffordanceDatabase
from .base import GraspContext, GraspPlan, PlanSource

__all__ = ["HeuristicGraspPlanner"]

log = get_logger(__name__)


class HeuristicGraspPlanner:
    """Rule-based planner over the affordance database."""

    #: Vision confidence below which the planner will not commit to a class-
    #: specific grasp and falls back to a generic power grip.
    MIN_LABEL_CONFIDENCE = 0.45

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
        return "heuristic"

    def plan(self, context: GraspContext) -> GraspPlan | None:
        reasons: list[str] = []
        target = context.target
        label = context.object_label
        vision_confidence = context.vision_confidence

        if target is None or vision_confidence < self.MIN_LABEL_CONFIDENCE:
            return self._generic_plan(context, reasons, vision_confidence)

        affordance = self._affordances.get(label)
        if affordance.label == "unknown":
            reasons.append(f"'{label}' is not in the affordance database")
        else:
            reasons.append(f"recognised {label} ({vision_confidence * 100:.0f}%)")

        width_m = self._estimate_width(context, affordance)
        grasp = self._select_grasp(affordance, width_m, context, reasons)
        preset = self._grips.get(grasp)

        # Aperture pre-shaping: open just enough to clear the object, plus a
        # margin. Opening fully wastes time and looks unnatural.
        preshape = preset.preshape
        if width_m is not None:
            clearance = min(self._kinematics.max_aperture_m, width_m * 1.35 + 0.012)
            closure = self._kinematics.closure_for_aperture(clearance)
            preshape = HandPose.uniform(closure)
            reasons.append(
                f"pre-shaping to {clearance * 100:.1f} cm aperture for a "
                f"{width_m * 100:.1f} cm object"
            )

        force = affordance.force_for(context.force_ceiling)
        if affordance.fragile:
            force = min(force, 0.35)
            reasons.append("fragile object — force capped at 35%")
        if affordance.heavy:
            force = min(context.force_ceiling, force * 1.15)
            reasons.append("heavy object — extra grip margin against slip")

        # The user's own effort scales the force within the allowed band, so a
        # gentle contraction produces a gentle grip. This is the most direct way
        # the user stays in control of *how hard*, not just *whether*.
        effort = clamp(0.55 + 0.45 * context.intent.strength)
        force = clamp(force * effort)

        speed = min(context.speed_ceiling, preset.speed * affordance.speed_scale)
        confidence = clamp(
            0.35 + 0.45 * vision_confidence + 0.2 * clamp(context.intent.confidence)
        )

        target_pose, notes = self._kinematics.enforce_limits(preset.pose)
        reasons.extend(notes)

        return GraspPlan(
            grasp=grasp,
            target=target_pose,
            preshape=preshape,
            force=force,
            speed=speed,
            confidence=confidence,
            source=PlanSource.HEURISTIC,
            label=affordance.label,
            reasons=tuple(reasons),
        )

    # -- internals ------------------------------------------------------------

    def _generic_plan(
        self, context: GraspContext, reasons: list[str], vision_confidence: float
    ) -> GraspPlan:
        """Plan with no usable object information.

        This is the path that matters most for the safety story: the user asked
        for a grasp and vision has nothing useful to say. The correct answer is
        *still grasp* — slowly, gently, with a generic power grip — because
        refusing to move would take control away from the user.
        """
        if vision_confidence <= 0.0:
            reasons.append("no object recognised — generic power grip")
        else:
            reasons.append(
                f"object confidence {vision_confidence * 100:.0f}% below "
                f"{self.MIN_LABEL_CONFIDENCE * 100:.0f}% — generic power grip"
            )
        preset = self._grips.get(GraspType.POWER)
        effort = clamp(0.5 + 0.5 * context.intent.strength)
        target_pose, notes = self._kinematics.enforce_limits(preset.pose)
        reasons.extend(notes)
        return GraspPlan(
            grasp=GraspType.POWER,
            target=target_pose,
            preshape=preset.preshape,
            # Deliberately conservative: unknown object, reduced force and speed.
            force=clamp(min(context.force_ceiling, 0.45) * effort),
            speed=min(context.speed_ceiling, 0.8),
            confidence=clamp(0.3 + 0.3 * context.intent.confidence),
            source=PlanSource.DEFAULT,
            label="unknown",
            reasons=tuple(reasons),
        )

    def _estimate_width(self, context: GraspContext, affordance: Affordance) -> float | None:
        """Object width in metres, from measured geometry or the class prior."""
        target = context.target
        distance = context.distance_m
        if target is not None and distance is not None and context.vision is not None:
            # Prefer the measured apparent size: it reflects *this* object, not
            # the average of its class.
            prior = OBJECT_SIZE_PRIORS.get(affordance.label)
            measured = target.bbox.width * distance * 1.6  # ~62° horizontal FOV
            if prior is not None:
                # Blend measurement with the prior, weighted by how far the
                # measurement is from plausible. A wildly off measurement (bad
                # depth) gets pulled back towards the prior rather than trusted.
                ratio = measured / max(1e-6, prior.width_m)
                weight = 0.7 if 0.5 <= ratio <= 2.0 else 0.25
                return measured * weight + prior.width_m * (1 - weight)
            return measured
        if affordance.typical_width_m > 0:
            return affordance.typical_width_m
        return None

    def _select_grasp(
        self,
        affordance: Affordance,
        width_m: float | None,
        context: GraspContext,
        reasons: list[str],
    ) -> GraspType:
        """First preferred grasp that the hand can physically achieve."""
        max_aperture = self._kinematics.max_aperture_m

        for grasp in affordance.grasps:
            if grasp not in self._grips:
                continue
            if width_m is not None and not self._is_feasible(grasp, width_m, max_aperture):
                reasons.append(
                    f"{grasp.label} rejected: {width_m * 100:.1f} cm does not suit it"
                )
                continue
            reasons.append(f"selected {grasp.label} from the {affordance.label} affordance")
            return grasp

        reasons.append("no preferred grasp is feasible — falling back to power grip")
        return GraspType.POWER

    @staticmethod
    def _is_feasible(grasp: GraspType, width_m: float, max_aperture_m: float) -> bool:
        """Whether a grasp suits an object of this width.

        Bounds come from hand geometry: a precision pinch cannot span a 9 cm
        bottle, and a power wrap around a 5 mm pen has nothing to close on.
        """
        if width_m > max_aperture_m * 0.95:
            # Too wide for any enclosing grasp; only a hook or lateral press works.
            return grasp in (GraspType.HOOK, GraspType.LATERAL_KEY)
        if grasp is GraspType.PRECISION_PINCH:
            return width_m <= 0.035
        if grasp is GraspType.TRIPOD:
            return width_m <= 0.055
        if grasp is GraspType.LATERAL_KEY:
            return width_m <= 0.030
        if grasp is GraspType.SPHERICAL:
            return 0.03 <= width_m <= max_aperture_m
        if grasp is GraspType.CYLINDRICAL:
            return 0.02 <= width_m <= max_aperture_m
        if grasp is GraspType.HOOK:
            return width_m <= 0.06
        return True
