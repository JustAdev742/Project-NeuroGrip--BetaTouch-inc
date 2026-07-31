"""Grasp planning interface.

A grasp planner answers exactly one question: *given that the user has decided to
grasp, what is the best way to do it?* It is never asked whether to grasp — that
decision belongs to the user, via
:class:`~neurogrip.emg.intent.IntentEstimate`, and is enforced upstream in
:mod:`neurogrip.fusion`.

Planners are interchangeable. :class:`GraspContext` in, :class:`GraspPlan` out,
no other coupling. Three ship:

* :class:`~neurogrip.ai.grasp.hggd.HggdGraspPlanner` — uses the grasp candidates
  the HGGD-MCU network predicts directly;
* :class:`~neurogrip.ai.grasp.heuristic.HeuristicGraspPlanner` — affordance-table
  reasoning from an object label and its measured size;
* :class:`~neurogrip.ai.grasp.composite.CompositeGraspPlanner` — tries planners in
  order and takes the first usable plan, which is how the system degrades from
  learned grasping to classical reasoning to a safe default without a gap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable

from ...core.types import GraspType, HandPose, ModeId, clamp
from ...emg.intent import IntentEstimate
from ...vision.types import Detection, GraspCandidate, VisionResult

__all__ = ["GraspContext", "GraspPlan", "GraspPlanner", "PlanSource"]


class PlanSource(str, Enum):
    """Which planner produced a plan — shown in the UI and logged."""

    LEARNED = "learned"
    HEURISTIC = "heuristic"
    DEFAULT = "default"
    USER = "user"


@dataclass(frozen=True, slots=True)
class GraspContext:
    """Everything a planner may look at.

    Passing one immutable context (rather than a long argument list) means adding
    a new signal — wrist orientation, an IMU, a second camera — does not change
    every planner's signature.
    """

    #: The user's confirmed intent. Never ``None``: planning does not start
    #: without it.
    intent: IntentEstimate
    #: Latest vision result, or ``None`` when vision is unavailable/stale.
    vision: VisionResult | None
    #: Current measured hand pose.
    current_pose: HandPose
    #: Active operating mode, which sets the force/speed envelope.
    mode: ModeId
    timestamp: float = 0.0
    #: Maximum grip force the mode and safety layer allow, in ``[0, 1]``.
    force_ceiling: float = 0.85
    #: Speed multiplier the mode allows.
    speed_ceiling: float = 1.0
    #: True when the hand is already holding something.
    holding: bool = False

    @property
    def target(self) -> Detection | None:
        """The detection the user is most likely aiming at."""
        return self.vision.primary if self.vision is not None else None

    @property
    def object_label(self) -> str:
        target = self.target
        return target.label if target is not None else ""

    @property
    def vision_confidence(self) -> float:
        return self.vision.object_confidence if self.vision is not None else 0.0

    @property
    def distance_m(self) -> float | None:
        if self.vision is None or self.vision.depth is None:
            return None
        return self.vision.depth.distance_m


@dataclass(frozen=True, slots=True)
class GraspPlan:
    """A concrete, executable grasp.

    ``reasons`` is not decoration. A shared-control device has to be able to
    explain itself: the UI shows these lines under "why this grip?", and the
    black-box recorder stores them alongside the decision, so any surprising
    behaviour can be reconstructed after the fact.
    """

    grasp: GraspType
    #: Final target pose.
    target: HandPose
    #: Optional intermediate pose to pass through (aperture pre-shaping).
    preshape: HandPose | None = None
    #: Grip force ceiling in ``[0, 1]``.
    force: float = 0.5
    #: Speed multiplier for the motion.
    speed: float = 1.0
    #: Planner's confidence in this plan.
    confidence: float = 0.0
    source: PlanSource = PlanSource.DEFAULT
    #: Object class the plan was built for.
    label: str = ""
    #: Vision grasp candidate that informed the plan, when there was one.
    candidate: GraspCandidate | None = None
    reasons: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_usable(self) -> bool:
        """Whether this plan is good enough to execute as-is."""
        return self.confidence > 0.0

    def scaled(self, *, force_ceiling: float, speed_ceiling: float) -> GraspPlan:
        """Clip the plan to an envelope. Only ever reduces force and speed."""
        return GraspPlan(
            grasp=self.grasp,
            target=self.target,
            preshape=self.preshape,
            force=clamp(min(self.force, force_ceiling)),
            speed=max(0.05, min(self.speed, speed_ceiling)),
            confidence=self.confidence,
            source=self.source,
            label=self.label,
            candidate=self.candidate,
            reasons=self.reasons,
        )

    def with_reason(self, reason: str) -> GraspPlan:
        return GraspPlan(
            grasp=self.grasp,
            target=self.target,
            preshape=self.preshape,
            force=self.force,
            speed=self.speed,
            confidence=self.confidence,
            source=self.source,
            label=self.label,
            candidate=self.candidate,
            reasons=(*self.reasons, reason),
        )

    def explain(self) -> str:
        """One-line explanation for the dashboard."""
        head = f"{self.grasp.label} ({self.confidence * 100:.0f}%)"
        if self.label:
            head += f" for {self.label}"
        return head + (f" — {self.reasons[0]}" if self.reasons else "")


@runtime_checkable
class GraspPlanner(Protocol):
    """Chooses how to perform a grasp the user has already asked for."""

    @property
    def name(self) -> str: ...

    def plan(self, context: GraspContext) -> GraspPlan | None:
        """Produce a plan, or ``None`` if this planner cannot contribute.

        Returning ``None`` is a normal outcome, not an error: it is how a
        composite chain moves on to the next planner.
        """
        ...
