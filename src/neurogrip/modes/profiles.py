"""The four operating modes.

Each is a :class:`~neurogrip.modes.base.ModeProfile` — a set of parameters — plus
at most a few lines of behaviour. Reading these four definitions side by side is
the fastest way to understand what actually differs between modes.

===============  =========  ==========  ==========  ==========  ================
mode             AI         force max   speed max   intent      vision rate
===============  =========  ==========  ==========  ==========  ================
Manual           **off**    0.70        1.0×        120 ms      0 Hz (off)
AI Assist        on         0.75        1.0×        120 ms      20 Hz
Sports           on         0.80        1.6×        70 ms       30 Hz
Training         **off**    0.50        1.0×        100 ms      0 Hz (off)
===============  =========  ==========  ==========  ==========  ================
"""

from __future__ import annotations

from ..control.controller import HandController
from ..control.motion import MotionLimits
from ..core.clock import Clock
from ..core.events import EventBus
from ..core.logging import get_logger
from ..core.topics import Topics
from ..core.types import GraspType, ModeId
from ..emg.intent import IntentSettings
from ..fusion.fusion import Decision, DecisionFusion
from ..fusion.policy import POLICIES
from .base import ModeBase, ModeContext, ModeProfile

__all__ = ["AiAssistMode", "ManualMode", "SportsMode", "TrainingMode", "build_modes"]

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Manual
# ---------------------------------------------------------------------------


MANUAL_PROFILE = ModeProfile(
    mode=ModeId.MANUAL,
    policy=POLICIES[ModeId.MANUAL],
    motion_limits=MotionLimits(max_velocity=1.5, max_acceleration=5.5),
    intent_settings=IntentSettings(dwell_s=0.12, release_s=0.18),
    title="Manual",
    subtitle="Direct EMG control — AI disabled",
    show_ai_disabled_banner=True,
    accent="neutral",
    s_curve=True,
    vision_rate_hz=0.0,
    control_rate_hz=200.0,
    notes=(
        "Every finger follows your muscle signal directly.",
        "Nothing is interpreted, predicted or optimised.",
        "Use this for learning, diagnostics, calibration, or whenever you want "
        "the hand to do exactly and only what you tell it.",
    ),
)


class ManualMode(ModeBase):
    """No assistance whatsoever.

    The camera is not merely ignored — it is not run (``vision_rate_hz = 0``), so
    there is no possibility of a stale perception result influencing anything.
    Combined with ``ai_enabled = False`` in the policy, the AI path is closed at
    two independent points.
    """

    def on_enter(self, previous: ModeId | None) -> None:
        super().on_enter(previous)
        log.info("manual mode active — AI assistance disabled")
        self._bus.publish(
            Topics.UI_NOTIFICATION,
            {"level": "info", "text": "Manual mode — AI disabled", "sticky": True},
            source="mode",
        )


# ---------------------------------------------------------------------------
# AI Assist
# ---------------------------------------------------------------------------


AI_ASSIST_PROFILE = ModeProfile(
    mode=ModeId.AI_ASSIST,
    policy=POLICIES[ModeId.AI_ASSIST],
    motion_limits=MotionLimits(max_velocity=1.6, max_acceleration=6.0),
    intent_settings=IntentSettings(dwell_s=0.12, release_s=0.15),
    title="AI Assist",
    subtitle="Shared control — you decide when, the AI decides how",
    accent="primary",
    vision_rate_hz=20.0,
    control_rate_hz=200.0,
    notes=(
        "Point the camera at an object, then contract to grasp.",
        "The hand chooses a grip that suits what it sees.",
        "Co-contract at any time to cancel.",
    ),
)


class AiAssistMode(ModeBase):
    """The primary mode.

    Sequence, enforced by :mod:`neurogrip.fusion` rather than by this class:

    1. the camera continuously perceives whatever is in front of the hand;
    2. **nothing happens** until EMG expresses a grasp intent;
    3. only then is the grasp planner consulted;
    4. the planned grip executes, scaled by the user's own effort;
    5. the user can cancel at any moment.

    Steps 1 and 3 being separate is the entire point: perception runs
    continuously so that a plan is *ready*, but a plan is never *executed*
    without step 2.
    """

    def on_enter(self, previous: ModeId | None) -> None:
        super().on_enter(previous)
        log.info("AI assist mode active")

    def update(self, context: ModeContext) -> Decision | None:
        decision = super().update(context)
        if decision is not None and decision.ai_contributed and decision.plan is not None:
            self._bus.publish(
                Topics.GRASP_PLANNED,
                {
                    "grasp": decision.plan.grasp.value,
                    "label": decision.plan.label,
                    "confidence": round(decision.confidence, 3),
                    "source": decision.plan.source.value,
                    "reasons": list(decision.plan.reasons[:3]),
                },
                source="mode",
            )
        return decision


# ---------------------------------------------------------------------------
# Sports
# ---------------------------------------------------------------------------


SPORTS_PROFILE = ModeProfile(
    mode=ModeId.SPORTS,
    policy=POLICIES[ModeId.SPORTS],
    # Higher velocity and acceleration, and a wider position tolerance so the
    # controller stops fussing over the last 3 % of travel it does not need.
    motion_limits=MotionLimits(
        max_velocity=2.6, max_acceleration=12.0, max_jerk=0.0, position_tolerance=0.03
    ),
    # Shorter dwell: reacting to a ball does not leave 120 ms to spare. The
    # trade-off is a higher false-activation rate, which is why this is not the
    # default mode.
    intent_settings=IntentSettings(
        dwell_s=0.07, cancel_dwell_s=0.03, release_s=0.10, max_age_s=0.20
    ),
    title="Sports",
    subtitle="Optimised for speed",
    accent="warning",
    s_curve=False,
    vision_rate_hz=30.0,
    control_rate_hz=250.0,
    notes=(
        "Faster movement and shorter reaction times.",
        "Muscle intent is still required for every action.",
        "Expect more accidental activations than in AI Assist.",
    ),
)


class SportsMode(ModeBase):
    """Speed-optimised shared control.

    Everything that adds latency is reduced: jerk limiting is off, the intent
    dwell is halved, the plan is refreshed more often, and the control loop runs
    faster. What is *not* reduced is the requirement for user intent — the same
    fusion gates apply, with the same structure.
    """

    def on_enter(self, previous: ModeId | None) -> None:
        super().on_enter(previous)
        log.info("sports mode active", speed_ceiling=self._profile.policy.speed_ceiling)
        self._bus.publish(
            Topics.UI_NOTIFICATION,
            {"level": "warning", "text": "Sports mode — faster, more sensitive", "sticky": False},
            source="mode",
        )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


TRAINING_PROFILE = ModeProfile(
    mode=ModeId.TRAINING,
    policy=POLICIES[ModeId.TRAINING],
    motion_limits=MotionLimits(max_velocity=1.3, max_acceleration=5.0),
    intent_settings=IntentSettings(dwell_s=0.10, release_s=0.12),
    title="Training",
    subtitle="Exercises and games for learning EMG control",
    show_ai_disabled_banner=True,
    accent="success",
    vision_rate_hz=0.0,
    control_rate_hz=200.0,
    notes=(
        "Practise with guided exercises and track your progress.",
        "Assistance is off so you see your real control.",
        "Force is reduced for comfort during repetition.",
    ),
)


class TrainingMode(ModeBase):
    """Hosts the training exercises.

    Assistance is deliberately off. An exercise that measures grip accuracy while
    an AI quietly corrects the user measures nothing, and a user who improves
    only with assistance has not improved.

    The exercise session is driven by
    :class:`~neurogrip.training.session.TrainingSession`, which this mode owns
    and pumps; hand control still flows through the same fusion path so what the
    user practises is what they will use.
    """

    def __init__(
        self,
        profile: ModeProfile,
        controller: HandController,
        fusion: DecisionFusion,
        clock: Clock,
        bus: EventBus,
        session=None,
    ) -> None:
        super().__init__(profile, controller, fusion, clock, bus)
        self._session = session

    @property
    def session(self):
        """The active :class:`~neurogrip.training.session.TrainingSession`."""
        return self._session

    def set_session(self, session) -> None:
        self._session = session

    def on_enter(self, previous: ModeId | None) -> None:
        super().on_enter(previous)
        # Start from a known, comfortable pose so exercise targets are relative
        # to the same starting point every time.
        self._controller.apply_grip(GraspType.RELAXED, source="training")
        log.info("training mode active")

    def on_exit(self, following: ModeId | None) -> None:
        if self._session is not None and self._session.active:
            self._session.stop("mode changed")
        super().on_exit(following)

    def update(self, context: ModeContext) -> Decision | None:
        decision = super().update(context)
        if self._session is not None and self._session.active and context.emg is not None:
            self._session.update(context.emg, context.hand, context.timestamp)
        return decision


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def build_modes(
    controller: HandController,
    fusion: DecisionFusion,
    clock: Clock,
    bus: EventBus,
    *,
    training_session=None,
) -> dict[ModeId, ModeBase]:
    """Instantiate every mode. Called once by the composition root."""
    return {
        ModeId.MANUAL: ManualMode(MANUAL_PROFILE, controller, fusion, clock, bus),
        ModeId.AI_ASSIST: AiAssistMode(AI_ASSIST_PROFILE, controller, fusion, clock, bus),
        ModeId.SPORTS: SportsMode(SPORTS_PROFILE, controller, fusion, clock, bus),
        ModeId.TRAINING: TrainingMode(
            TRAINING_PROFILE, controller, fusion, clock, bus, training_session
        ),
    }
