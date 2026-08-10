"""Composite planner: an ordered chain with a guaranteed floor.

The chain expresses the degradation policy explicitly:

    HGGD-MCU (learned)  →  affordance heuristic  →  safe default

Each planner may decline (return ``None``); the first plan that clears
``min_confidence`` wins. If every planner declines, :class:`CompositeGraspPlanner`
still returns a plan — a slow, low-force power grip.

That last property is the important one. The user has already asked for a grasp;
the AI's job is to improve it, not to veto it. "The model was unsure so nothing
happened" is a failure mode this class exists to make impossible.
"""

from __future__ import annotations

from ...control.grips import GripLibrary
from ...control.kinematics import HandKinematics
from ...core.logging import get_logger
from ...core.types import GraspType, clamp
from .base import GraspContext, GraspPlan, GraspPlanner, PlanSource

__all__ = ["CompositeGraspPlanner"]

log = get_logger(__name__)


class CompositeGraspPlanner:
    """Tries planners in order and always produces a usable plan."""

    def __init__(
        self,
        planners: tuple[GraspPlanner, ...],
        grips: GripLibrary,
        kinematics: HandKinematics | None = None,
        *,
        min_confidence: float = 0.35,
    ) -> None:
        self._planners = planners
        self._grips = grips
        self._kinematics = kinematics or HandKinematics()
        #: A plan below this confidence is not accepted from a planner, but the
        #: chain still falls through to the safe default rather than to nothing.
        self._min_confidence = min_confidence
        #: Diagnostics: how often each planner won.
        self.wins: dict[str, int] = {p.name: 0 for p in planners}
        self.wins["default"] = 0

    @property
    def name(self) -> str:
        return "composite(" + "→".join(p.name for p in self._planners) + ")"

    @property
    def planners(self) -> tuple[GraspPlanner, ...]:
        return self._planners

    def plan(self, context: GraspContext) -> GraspPlan:
        attempted: list[str] = []

        for planner in self._planners:
            try:
                candidate = planner.plan(context)
            except Exception as exc:
                log.throttled(
                    f"planner-{planner.name}",
                    "error",
                    "grasp planner raised; skipping",
                    now=context.timestamp,
                    planner=planner.name,
                    error=str(exc),
                )
                attempted.append(f"{planner.name}: error")
                continue

            if candidate is None:
                attempted.append(f"{planner.name}: no plan")
                continue
            if candidate.confidence < self._min_confidence:
                attempted.append(f"{planner.name}: confidence {candidate.confidence:.2f} too low")
                continue

            self.wins[planner.name] = self.wins.get(planner.name, 0) + 1
            if attempted:
                candidate = candidate.with_reason("tried first: " + "; ".join(attempted))
            return candidate.scaled(
                force_ceiling=context.force_ceiling, speed_ceiling=context.speed_ceiling
            )

        self.wins["default"] += 1
        return self._safe_default(context, attempted)

    def _safe_default(self, context: GraspContext, attempted: list[str]) -> GraspPlan:
        """The floor of the chain: a slow, gentle power grip.

        Confidence is reported honestly (low), so the UI shows the user that the
        AI is not contributing much — but the hand still closes, because that is
        what they asked for.
        """
        preset = self._grips.get(GraspType.POWER)
        target, notes = self._kinematics.enforce_limits(preset.pose)
        effort = clamp(0.5 + 0.5 * context.intent.strength)
        reasons = ["no planner produced a confident plan — safe default power grip"]
        reasons.extend(attempted)
        reasons.extend(notes)
        return GraspPlan(
            grasp=GraspType.POWER,
            target=target,
            preshape=preset.preshape,
            force=clamp(min(context.force_ceiling, 0.40) * effort),
            speed=min(context.speed_ceiling, 0.7),
            confidence=clamp(0.25 * context.intent.confidence),
            source=PlanSource.DEFAULT,
            label=context.object_label or "unknown",
            reasons=tuple(reasons),
        )
