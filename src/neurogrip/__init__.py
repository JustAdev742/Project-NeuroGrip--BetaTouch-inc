"""NeuroGrip — shared-control software stack for an AI-assisted prosthetic hand.

The guiding rule of this codebase, enforced structurally rather than by convention:

    The AI never replaces the user. The user is always in control.

Concretely, that rule shows up as three architectural invariants:

1. :class:`~neurogrip.control.controller.HandController` is the *only* component that
   writes to the servo bus. Every motion request funnels through it, so limits,
   emergency-stop and cancellation have exactly one place to be enforced.
2. :class:`~neurogrip.fusion.fusion.DecisionFusion` cannot emit a motion decision
   unless a fresh EMG intent is present. Vision and the grasp planner only ever
   influence *how* a movement is performed, never *whether* it happens.
3. Any failure in the assistive path (vision, planner, model load, confidence too
   low) degrades to direct user control — never to a blocked or autonomous hand.

See ``docs/architecture.md`` for the full design and ``docs/safety.md`` for the
safety case.
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["__version__"]
