"""Motion control.

Layering, from the top::

    HandController      the single writer to the servo bus; enforces every limit
      ├─ MotionQueue    priority arbitration and preemption
      ├─ TrajectoryGen  synchronised, limit-respecting motion between poses
      ├─ AdaptiveGrip   contact detection and force regulation from current
      ├─ GripLibrary    named grasp presets (data, loadable from config)
      └─ HandKinematics closure ↔ tendon travel ↔ aperture, self-collision limits

Nothing outside :class:`~neurogrip.control.controller.HandController` may command
the actuators. See ``docs/safety.md`` for why that constraint is load-bearing.
"""

from __future__ import annotations

from .controller import HandController, HandState
from .force import AdaptiveGripController, GripSettings, GripState
from .grips import GripLibrary, GripPreset
from .kinematics import FingerGeometry, HandKinematics
from .motion import MotionLimits, TrajectoryGenerator, TrajectoryState
from .queue import CommandResult, MotionCommand, MotionQueue, Priority

__all__ = [
    "AdaptiveGripController",
    "CommandResult",
    "FingerGeometry",
    "GripLibrary",
    "GripPreset",
    "GripSettings",
    "GripState",
    "HandController",
    "HandKinematics",
    "HandState",
    "MotionCommand",
    "MotionLimits",
    "MotionQueue",
    "Priority",
    "TrajectoryGenerator",
    "TrajectoryState",
]
