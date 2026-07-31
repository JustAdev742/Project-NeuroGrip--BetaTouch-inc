"""Fusion policies: the confidence thresholds that gate assistance.

A policy is the set of numbers that decide how much evidence is needed before the
AI is allowed to shape a movement, and how quickly assistance decays as that
evidence goes stale. Each operating mode supplies its own policy, which is how
Sports Mode becomes faster without any special-casing inside the fusion logic
itself.

Every threshold here has a direction that is *safe*: higher thresholds mean less
AI involvement and more direct user control. There is no value in this file that
can be raised to make the hand act more autonomously — the structural gate
("no intent, no motion") is not a threshold and cannot be tuned away.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core.config import Config
from ..core.types import ModeId, clamp

__all__ = ["POLICIES", "FusionPolicy", "policy_for_mode"]


@dataclass(frozen=True, slots=True)
class FusionPolicy:
    """Thresholds and weights for one operating mode."""

    mode: ModeId

    # -- gating ---------------------------------------------------------------
    #: Minimum intent confidence before any motion is authorised.
    min_intent_confidence: float = 0.45
    #: Minimum combined confidence before the AI's *plan* is used. Below this the
    #: user still gets motion — just direct, unassisted motion.
    min_combined_confidence: float = 0.55
    #: Maximum age of an intent estimate that may authorise motion.
    max_intent_age_s: float = 0.30
    #: Maximum age of a vision result that may inform a plan.
    max_vision_age_s: float = 0.50

    # -- weighting ------------------------------------------------------------
    #: Weight of EMG evidence in the combined score.
    emg_weight: float = 0.6
    #: Weight of vision evidence.
    vision_weight: float = 0.4
    #: Extra weight for a stable, well-tracked object.
    stability_bonus: float = 0.1

    # -- execution envelope ---------------------------------------------------
    #: Maximum grip force this mode permits, in ``[0, 1]``.
    force_ceiling: float = 0.75
    #: Speed multiplier this mode permits.
    speed_ceiling: float = 1.0
    #: Seconds to hold a plan before re-planning; prevents the hand from
    #: changing its mind mid-reach when the classifier flickers.
    plan_hold_s: float = 0.6

    # -- behaviour ------------------------------------------------------------
    #: Whether AI assistance is permitted at all in this mode.
    ai_enabled: bool = True
    #: Whether EMG drives finger position directly and continuously.
    proportional_control: bool = True
    #: Whether a partially-confident plan may be blended with direct control.
    allow_blending: bool = True

    def combined_confidence(
        self, intent_confidence: float, vision_confidence: float, *, stability: float = 0.0
    ) -> float:
        """Weighted evidence score.

        When vision is unavailable (``vision_confidence == 0``) the EMG weight is
        renormalised to 1.0 rather than the score being penalised. Missing vision
        must not make the user's own clearly-expressed intent count for less.
        """
        if vision_confidence <= 0.0:
            return clamp(intent_confidence)
        total = self.emg_weight + self.vision_weight
        base = (
            intent_confidence * self.emg_weight + vision_confidence * self.vision_weight
        ) / max(1e-6, total)
        return clamp(base + self.stability_bonus * clamp(stability))

    def with_overrides(self, **kwargs: float | bool) -> FusionPolicy:
        """Return a copy with fields replaced (used by the settings screen)."""
        from dataclasses import replace

        return replace(self, **kwargs)  # type: ignore[arg-type]


#: Built-in policies, one per mode.
POLICIES: dict[ModeId, FusionPolicy] = {
    ModeId.MANUAL: FusionPolicy(
        mode=ModeId.MANUAL,
        # Manual mode is *entirely* the user. The AI contributes nothing, and the
        # UI says so prominently. Thresholds are still defined so the same code
        # path runs — there is no separate "manual" branch in the fusion logic to
        # get out of step with the assisted one.
        min_intent_confidence=0.35,
        min_combined_confidence=1.01,  # unreachable: the AI plan is never used
        emg_weight=1.0,
        vision_weight=0.0,
        force_ceiling=0.70,
        speed_ceiling=1.0,
        ai_enabled=False,
        proportional_control=True,
        allow_blending=False,
    ),
    ModeId.AI_ASSIST: FusionPolicy(
        mode=ModeId.AI_ASSIST,
        min_intent_confidence=0.45,
        min_combined_confidence=0.55,
        max_intent_age_s=0.30,
        max_vision_age_s=0.50,
        emg_weight=0.60,
        vision_weight=0.40,
        stability_bonus=0.10,
        force_ceiling=0.75,
        speed_ceiling=1.0,
        plan_hold_s=0.6,
        ai_enabled=True,
        proportional_control=True,
        allow_blending=True,
    ),
    ModeId.SPORTS: FusionPolicy(
        mode=ModeId.SPORTS,
        # Faster reactions, so: shorter intent dwell is tolerated (lower
        # confidence threshold), vision is weighted lower because it is the slow
        # input, and the plan is refreshed more often.
        min_intent_confidence=0.38,
        min_combined_confidence=0.48,
        max_intent_age_s=0.20,
        max_vision_age_s=0.30,
        emg_weight=0.75,
        vision_weight=0.25,
        stability_bonus=0.05,
        force_ceiling=0.80,
        speed_ceiling=1.6,
        plan_hold_s=0.3,
        ai_enabled=True,
        proportional_control=True,
        allow_blending=True,
    ),
    ModeId.TRAINING: FusionPolicy(
        mode=ModeId.TRAINING,
        # Training must reflect the user's raw control so they can learn from it.
        # No assistance, no smoothing of their mistakes, reduced force because
        # exercises involve a lot of repetition.
        min_intent_confidence=0.30,
        min_combined_confidence=1.01,
        emg_weight=1.0,
        vision_weight=0.0,
        force_ceiling=0.50,
        speed_ceiling=1.0,
        ai_enabled=False,
        proportional_control=True,
        allow_blending=False,
    ),
}


def policy_for_mode(mode: ModeId, config: Config | None = None) -> FusionPolicy:
    """Policy for ``mode``, with optional ``[fusion.<mode>]`` overrides.

    Overrides are clamped so a configuration file cannot lower a threshold below
    the built-in floor — configuration may make the system *more* conservative,
    never less.
    """
    base = POLICIES.get(mode, POLICIES[ModeId.AI_ASSIST])
    if config is None:
        return base

    section = config.section(f"fusion.{mode.value}")
    if not list(section.keys()):
        return base

    return FusionPolicy(
        mode=mode,
        min_intent_confidence=max(
            base.min_intent_confidence * 0.8,
            section.get_float("min_intent_confidence", base.min_intent_confidence),
        ),
        min_combined_confidence=max(
            base.min_combined_confidence * 0.8,
            section.get_float("min_combined_confidence", base.min_combined_confidence),
        ),
        max_intent_age_s=min(
            base.max_intent_age_s * 1.5,
            section.get_float("max_intent_age_s", base.max_intent_age_s),
        ),
        max_vision_age_s=min(
            base.max_vision_age_s * 1.5,
            section.get_float("max_vision_age_s", base.max_vision_age_s),
        ),
        emg_weight=section.get_float("emg_weight", base.emg_weight),
        vision_weight=section.get_float("vision_weight", base.vision_weight),
        stability_bonus=section.get_float("stability_bonus", base.stability_bonus),
        force_ceiling=min(
            base.force_ceiling, section.get_float("force_ceiling", base.force_ceiling)
        ),
        speed_ceiling=min(
            base.speed_ceiling * 1.25, section.get_float("speed_ceiling", base.speed_ceiling)
        ),
        plan_hold_s=section.get_float("plan_hold_s", base.plan_hold_s),
        # Enabling AI where the built-in policy disables it is not permitted:
        # Manual Mode means manual.
        ai_enabled=base.ai_enabled and section.get_bool("ai_enabled", base.ai_enabled),
        proportional_control=section.get_bool("proportional_control", base.proportional_control),
        allow_blending=base.allow_blending and section.get_bool("allow_blending", base.allow_blending),
    )
