"""Operating modes.

A mode is a bundle of *parameters*, not a separate control path. All four run the
identical pipeline and differ only in thresholds, limits and presentation — which
is why adding a mode cannot add a new way for the hand to move.

* **Manual** — direct EMG control, AI disabled, camera not even running.
* **AI Assist** — the primary shared-control mode.
* **Sports** — the same, tuned for reaction speed.
* **Training** — assistance off, hosting the learning exercises.

:class:`~neurogrip.modes.manager.ModeManager` owns the active mode and enforces
the safety veto and the automatic fall-back to Manual.
"""

from __future__ import annotations

from .base import ModeBase, ModeContext, ModeProfile, OperatingMode
from .manager import ModeChange, ModeManager
from .profiles import (
    AI_ASSIST_PROFILE,
    MANUAL_PROFILE,
    SPORTS_PROFILE,
    TRAINING_PROFILE,
    AiAssistMode,
    ManualMode,
    SportsMode,
    TrainingMode,
    build_modes,
)

__all__ = [
    "AI_ASSIST_PROFILE",
    "MANUAL_PROFILE",
    "SPORTS_PROFILE",
    "TRAINING_PROFILE",
    "AiAssistMode",
    "ManualMode",
    "ModeBase",
    "ModeChange",
    "ModeContext",
    "ModeManager",
    "ModeProfile",
    "OperatingMode",
    "SportsMode",
    "TrainingMode",
    "build_modes",
]
