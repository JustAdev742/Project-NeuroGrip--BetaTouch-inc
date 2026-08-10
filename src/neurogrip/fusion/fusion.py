"""Decision fusion — where user intent and machine perception meet.

This is the most safety-relevant module in the stack, and it is written to be
read. The whole of :meth:`DecisionFusion.evaluate` is a sequence of explicit,
ordered gates, each of which can only ever *reduce* what the system is allowed to
do.

Gate order (earlier gates dominate):

===  =====================  ==================================================
  1  Safety                 Motion blocked → ``BLOCKED``. Nothing overrides this.
  2  Cancel                 Co-contraction → ``CANCEL``. Beats every other intent.
  3  Intent presence        No fresh intent → ``IDLE``. **The AI cannot pass
                            this gate on its own.**
  4  Intent confidence      Weak intent → ``IDLE`` with an explanation.
  5  Mode policy            AI disabled (Manual/Training) → ``DIRECT``.
  6  Evidence               Combined confidence below threshold → ``DIRECT``.
  7  Plan                   Planner produces a plan → ``ASSISTED``.
===  =====================  ==================================================

Note what gates 5–7 do when they fail: they fall through to ``DIRECT``, i.e.
direct proportional control by the user. They never fall through to "do nothing".
The failure of assistance is never allowed to become the failure of the hand.

Every decision carries ``reasons``, which the dashboard renders verbatim and the
black-box recorder stores. If the hand does something the user did not expect,
the reason it did so is already written down.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ..ai.grasp import GraspContext, GraspPlan, GraspPlanner
from ..core.clock import Clock
from ..core.logging import get_logger
from ..core.types import HandPose, IntentKind, ModeId, clamp
from ..emg.intent import IntentEstimate
from ..emg.quality import SignalQuality
from ..vision.types import VisionResult
from .evidence import Evidence, EvidenceSet
from .policy import FusionPolicy

__all__ = ["Decision", "DecisionAction", "DecisionFusion", "FusionInputs"]

log = get_logger(__name__)


class DecisionAction(str, Enum):
    """What the control layer should do with this decision."""

    #: Hold position; no motion authorised.
    IDLE = "idle"
    #: Follow the user's EMG directly and proportionally — no AI shaping.
    DIRECT = "direct"
    #: Execute the AI's grasp plan, still scaled by the user's effort.
    ASSISTED = "assisted"
    #: Open the hand and release whatever is held.
    RELEASE = "release"
    #: Abort any motion in progress and hold.
    CANCEL = "cancel"
    #: Safety refuses motion.
    BLOCKED = "blocked"

    @property
    def commands_motion(self) -> bool:
        return self in (DecisionAction.DIRECT, DecisionAction.ASSISTED, DecisionAction.RELEASE)


@dataclass(frozen=True, slots=True)
class FusionInputs:
    """Everything fusion looks at for one evaluation."""

    intent: IntentEstimate | None
    vision: VisionResult | None
    current_pose: HandPose
    mode: ModeId
    timestamp: float
    #: False when the safety layer forbids motion for any reason.
    motion_allowed: bool = True
    #: False when safety forbids AI assistance but still permits direct control.
    ai_allowed: bool = True
    #: Reason string from the safety layer, surfaced verbatim to the user.
    safety_reason: str = ""
    #: True when the hand is currently holding an object.
    holding: bool = False
    #: Force ceiling imposed by safety (battery, thermal, fault state).
    safety_force_ceiling: float = 1.0


@dataclass(frozen=True, slots=True)
class Decision:
    """The fused decision for one control cycle."""

    action: DecisionAction
    timestamp: float
    #: Combined evidence score behind this decision.
    confidence: float = 0.0
    #: Target pose for DIRECT actions; ``None`` when the plan supplies it.
    target: HandPose | None = None
    #: The AI plan, present only for ``ASSISTED``.
    plan: GraspPlan | None = None
    #: Force ceiling for this decision, in ``[0, 1]``.
    force: float = 0.5
    #: Speed multiplier for this decision.
    speed: float = 1.0
    reasons: tuple[str, ...] = field(default_factory=tuple)
    evidence: EvidenceSet | None = None

    @property
    def commands_motion(self) -> bool:
        return self.action.commands_motion

    @property
    def ai_contributed(self) -> bool:
        return self.action is DecisionAction.ASSISTED and self.plan is not None

    def explain(self) -> str:
        """One-line summary for the dashboard."""
        head = self.action.value.upper()
        if self.plan is not None:
            head += f" · {self.plan.grasp.label}"
        return f"{head} ({self.confidence * 100:.0f}%)" + (
            f" — {self.reasons[0]}" if self.reasons else ""
        )

    @classmethod
    def idle(cls, timestamp: float, *reasons: str) -> Decision:
        return cls(action=DecisionAction.IDLE, timestamp=timestamp, reasons=tuple(reasons))


def _label_of(vision: VisionResult | None, fallback: str) -> str:
    """Label of the current primary detection, or ``fallback`` when there is none."""
    if vision is None or vision.primary is None:
        return fallback
    return vision.primary.label


class DecisionFusion:
    """Combines user intent, perception, hand state, mode and safety."""

    def __init__(
        self,
        planner: GraspPlanner,
        clock: Clock,
        *,
        policy: FusionPolicy | None = None,
    ) -> None:
        self._planner = planner
        self._clock = clock
        self._policy = policy
        #: The plan currently being executed, held for ``plan_hold_s`` so the
        #: hand does not re-plan every cycle mid-reach.
        self._held_plan: GraspPlan | None = None
        self._held_since = 0.0
        #: Diagnostics counters.
        self.decisions = 0
        self.assisted = 0
        self.direct = 0
        self.blocked = 0
        self.cancels = 0

    # -- configuration --------------------------------------------------------

    def set_policy(self, policy: FusionPolicy) -> None:
        """Install the active mode's policy. Clears any held plan."""
        self._policy = policy
        self._held_plan = None

    @property
    def policy(self) -> FusionPolicy | None:
        return self._policy

    def reset(self) -> None:
        """Forget the held plan (mode change, e-stop, cancel)."""
        self._held_plan = None

    # -- evaluation -----------------------------------------------------------

    def evaluate(self, inputs: FusionInputs) -> Decision:
        """Run the gate sequence and produce a decision."""
        self.decisions += 1
        policy = self._policy
        if policy is None or policy.mode is not inputs.mode:
            from .policy import policy_for_mode

            policy = policy_for_mode(inputs.mode)
            self._policy = policy

        now = inputs.timestamp
        reasons: list[str] = []
        evidence = self._collect_evidence(inputs, policy, now)

        # ---- Gate 1: safety ------------------------------------------------
        if not inputs.motion_allowed:
            self.blocked += 1
            self._held_plan = None
            return Decision(
                action=DecisionAction.BLOCKED,
                timestamp=now,
                reasons=(inputs.safety_reason or "motion blocked by the safety monitor",),
                evidence=evidence,
            )

        intent = inputs.intent

        # ---- Gate 2: cancel -------------------------------------------------
        # Checked before intent freshness: an abort must work even if it is the
        # only thing the EMG system has managed to produce.
        if intent is not None and intent.is_cancel:
            self.cancels += 1
            self._held_plan = None
            return Decision(
                action=DecisionAction.CANCEL,
                timestamp=now,
                confidence=intent.confidence,
                reasons=("user cancelled (co-contraction)",),
                evidence=evidence,
            )

        # ---- Gate 3: intent presence ---------------------------------------
        # This is the structural expression of "the AI never moves the hand".
        # There is no branch below this point that can be reached without a
        # fresh, motion-requesting intent from the user.
        if intent is None:
            return Decision(
                action=DecisionAction.IDLE,
                timestamp=now,
                reasons=("no EMG intent available",),
                evidence=evidence,
            )
        if not intent.is_fresh(now, policy.max_intent_age_s):
            self._held_plan = None
            return Decision(
                action=DecisionAction.IDLE,
                timestamp=now,
                reasons=(f"EMG intent is stale ({intent.age(now) * 1000:.0f} ms old)",),
                evidence=evidence,
            )
        if not intent.requests_motion:
            if intent.kind is IntentKind.REST and inputs.holding:
                # Holding an object while at rest: maintain the grip, do not
                # spontaneously open. Releasing must be a deliberate act.
                return Decision(
                    action=DecisionAction.IDLE,
                    timestamp=now,
                    confidence=intent.confidence,
                    reasons=("holding grip; no new intent",),
                    evidence=evidence,
                )
            self._held_plan = None
            return Decision(
                action=DecisionAction.IDLE,
                timestamp=now,
                confidence=intent.confidence,
                reasons=(f"intent is '{intent.kind.value}' — no motion requested",),
                evidence=evidence,
            )

        # ---- Gate 4: intent confidence -------------------------------------
        if intent.confidence < policy.min_intent_confidence:
            return Decision(
                action=DecisionAction.IDLE,
                timestamp=now,
                confidence=intent.confidence,
                reasons=(
                    f"intent confidence {intent.confidence:.2f} below "
                    f"{policy.min_intent_confidence:.2f}",
                ),
                evidence=evidence,
            )
        if intent.quality < SignalQuality.FAIR:
            reasons.append(f"EMG quality is {intent.quality.label}; assistance limited")

        # ---- Opening is always direct --------------------------------------
        # Nothing about opening the hand benefits from a grasp plan, and making
        # release depend on the AI would be a poor failure mode.
        if intent.kind is IntentKind.OPEN:
            self.direct += 1
            self._held_plan = None
            return Decision(
                action=DecisionAction.RELEASE,
                timestamp=now,
                confidence=intent.confidence,
                target=HandPose.open_hand(),
                force=0.25,
                speed=policy.speed_ceiling,
                reasons=("user is opening the hand", *reasons),
                evidence=evidence,
            )

        force_ceiling = clamp(min(policy.force_ceiling, inputs.safety_force_ceiling))

        # ---- Gate 5: mode policy -------------------------------------------
        if not policy.ai_enabled or not inputs.ai_allowed:
            self.direct += 1
            why = (
                "AI assistance is disabled in this mode"
                if not policy.ai_enabled
                else inputs.safety_reason or "AI assistance suspended by safety"
            )
            return self._direct_decision(inputs, policy, intent, force_ceiling, [why, *reasons], evidence)

        # ---- Gate 6: combined evidence -------------------------------------
        vision = inputs.vision if self._vision_usable(inputs, policy, now) else None
        vision_confidence = vision.object_confidence if vision is not None else 0.0
        stability = self._stability(vision)
        combined = policy.combined_confidence(
            intent.confidence, vision_confidence, stability=stability
        )

        if combined < policy.min_combined_confidence:
            self.direct += 1
            return self._direct_decision(
                inputs,
                policy,
                intent,
                force_ceiling,
                [
                    f"combined confidence {combined:.2f} below "
                    f"{policy.min_combined_confidence:.2f} — direct control",
                    *reasons,
                ],
                evidence,
                confidence=combined,
            )

        # ---- Gate 7: plan ---------------------------------------------------
        plan = self._plan(inputs, policy, intent, vision, force_ceiling, now)
        if plan is None:
            self.direct += 1
            return self._direct_decision(
                inputs, policy, intent, force_ceiling, ["no usable grasp plan", *reasons], evidence,
                confidence=combined,
            )

        self.assisted += 1
        return Decision(
            action=DecisionAction.ASSISTED,
            timestamp=now,
            confidence=combined,
            target=plan.target,
            plan=plan,
            force=clamp(min(plan.force, force_ceiling)),
            speed=min(plan.speed, policy.speed_ceiling),
            reasons=(
                f"assisting: {plan.explain()}",
                *reasons,
                *plan.reasons[:3],
            ),
            evidence=evidence,
        )

    # -- helpers --------------------------------------------------------------

    def _direct_decision(
        self,
        inputs: FusionInputs,
        policy: FusionPolicy,
        intent: IntentEstimate,
        force_ceiling: float,
        reasons: list[str],
        evidence: EvidenceSet,
        *,
        confidence: float | None = None,
    ) -> Decision:
        """Build a direct proportional-control decision.

        Finger closure tracks the user's effort one-to-one. This is the
        behaviour Manual Mode always uses and the behaviour every assisted path
        degrades to.
        """
        closure = clamp(intent.strength) if policy.proportional_control else 1.0
        target = HandPose.uniform(closure)
        return Decision(
            action=DecisionAction.DIRECT,
            timestamp=inputs.timestamp,
            confidence=intent.confidence if confidence is None else confidence,
            target=target,
            force=clamp(force_ceiling * (0.4 + 0.6 * clamp(intent.strength))),
            speed=policy.speed_ceiling,
            reasons=tuple(reasons),
            evidence=evidence,
        )

    def _plan(
        self,
        inputs: FusionInputs,
        policy: FusionPolicy,
        intent: IntentEstimate,
        vision: VisionResult | None,
        force_ceiling: float,
        now: float,
    ) -> GraspPlan | None:
        """Get a plan, reusing the held one while it is still current."""
        if (
            self._held_plan is not None
            and now - self._held_since < policy.plan_hold_s
            and self._held_plan.label == _label_of(vision, self._held_plan.label)
        ):
            return self._held_plan

        context = GraspContext(
            intent=intent,
            vision=vision,
            current_pose=inputs.current_pose,
            mode=inputs.mode,
            timestamp=now,
            force_ceiling=force_ceiling,
            speed_ceiling=policy.speed_ceiling,
            holding=inputs.holding,
        )
        try:
            plan = self._planner.plan(context)
        except Exception as exc:
            log.throttled("fusion-plan", "error", "grasp planner failed", now=now, error=str(exc))
            return None

        if plan is None or not plan.is_usable:
            return None

        self._held_plan = plan
        self._held_since = now
        return plan

    def _vision_usable(self, inputs: FusionInputs, policy: FusionPolicy, now: float) -> bool:
        vision = inputs.vision
        if vision is None or not vision.ok:
            return False
        if not vision.is_fresh(now, policy.max_vision_age_s):
            return False
        return vision.primary is not None

    @staticmethod
    def _stability(vision: VisionResult | None) -> float:
        """How settled the perception is, in ``[0, 1]``."""
        if vision is None or vision.primary is None:
            return 0.0
        primary = vision.primary
        age_score = clamp(primary.age / 8.0)
        agreement = float(primary.attributes.get("label_agreement", 1.0))
        return clamp(age_score * 0.5 + agreement * 0.5)

    def _collect_evidence(
        self, inputs: FusionInputs, policy: FusionPolicy, now: float
    ) -> EvidenceSet:
        """Snapshot every input as weighted, timestamped evidence.

        Kept even when a decision short-circuits at gate 1, because the
        diagnostics screen and the incident recorder need to show *what the
        system knew* at the moment it decided, not just what it decided.
        """
        items: list[Evidence] = []
        intent = inputs.intent
        if intent is not None:
            items.append(
                Evidence(
                    source="emg",
                    label=intent.kind.value,
                    confidence=intent.confidence,
                    weight=policy.emg_weight,
                    timestamp=intent.timestamp,
                    detail=f"strength {intent.strength:.2f}, quality {intent.quality.label}",
                )
            )
        vision = inputs.vision
        if vision is not None and vision.primary is not None:
            items.append(
                Evidence(
                    source="vision",
                    label=vision.primary.label,
                    confidence=vision.object_confidence,
                    weight=policy.vision_weight,
                    timestamp=vision.timestamp,
                    detail=f"track {vision.primary.track_id}, age {vision.primary.age}",
                )
            )
        if vision is not None and vision.depth is not None:
            items.append(
                Evidence(
                    source="depth",
                    label=f"{vision.depth.distance_m * 100:.0f} cm",
                    confidence=vision.depth.confidence,
                    weight=0.0,  # informational only
                    timestamp=vision.timestamp,
                    detail=vision.depth.method,
                )
            )
        return EvidenceSet(items=tuple(items), evaluated_at=now)

    # -- statistics -----------------------------------------------------------

    def stats(self) -> dict[str, int | float]:
        """Counters for the diagnostics screen."""
        return {
            "decisions": self.decisions,
            "assisted": self.assisted,
            "direct": self.direct,
            "blocked": self.blocked,
            "cancels": self.cancels,
            "assist_rate": self.assisted / self.decisions if self.decisions else 0.0,
        }
