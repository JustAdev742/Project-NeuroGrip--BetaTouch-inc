"""Decision fusion: combining user intent with machine perception.

This package answers one question per control cycle: *given what the user is
asking for, what the camera sees, where the hand is, which mode is active and
what safety permits — what should happen now?*

Its central property, and the reason the whole architecture is shaped the way it
is:

    :class:`~neurogrip.fusion.fusion.DecisionFusion` cannot emit a motion
    decision without a fresh, confident, motion-requesting
    :class:`~neurogrip.emg.intent.IntentEstimate` from the user.

Vision, the grasp planner and the affordance database can only influence *how* a
movement is performed. Their absence, failure or uncertainty degrades the result
to direct proportional control — never to inaction, and never to autonomy.

See ``docs/fusion.md`` for the gate-by-gate walkthrough and
``tests/unit/test_fusion.py`` for the executable version of the same claims.
"""

from __future__ import annotations

from .evidence import Evidence, EvidenceSet
from .fusion import Decision, DecisionAction, DecisionFusion, FusionInputs
from .policy import POLICIES, FusionPolicy, policy_for_mode

__all__ = [
    "POLICIES",
    "Decision",
    "DecisionAction",
    "DecisionFusion",
    "Evidence",
    "EvidenceSet",
    "FusionInputs",
    "FusionPolicy",
    "policy_for_mode",
]
