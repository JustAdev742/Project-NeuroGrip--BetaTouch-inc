"""AI layer: object knowledge and grasp planning.

Deliberately narrow. The AI in this system does exactly one thing — decide *how*
to perform a grasp the user has already committed to — and this package contains
that and nothing else. It has no access to actuators, no timer of its own, and no
way to initiate motion.

Contents:

* :mod:`neurogrip.ai.objects` — the affordance database, mapping object classes
  to grasp preferences, force ceilings and approach speeds.
* :mod:`neurogrip.ai.grasp` — the planner interface and the shipped chain
  (HGGD-MCU → affordance heuristic → safe default).
* :mod:`neurogrip.ai.models` — model file location and integrity checking.
"""

from __future__ import annotations

from .grasp import (
    CompositeGraspPlanner,
    GraspContext,
    GraspPlan,
    GraspPlanner,
    HeuristicGraspPlanner,
    HggdGraspPlanner,
    PlanSource,
    build_default_planner,
)
from .models import ModelEntry, ModelRegistry, ModelStatus
from .objects import DEFAULT_AFFORDANCE, Affordance, AffordanceDatabase

__all__ = [
    "DEFAULT_AFFORDANCE",
    "Affordance",
    "AffordanceDatabase",
    "CompositeGraspPlanner",
    "GraspContext",
    "GraspPlan",
    "GraspPlanner",
    "HeuristicGraspPlanner",
    "HggdGraspPlanner",
    "ModelEntry",
    "ModelRegistry",
    "ModelStatus",
    "PlanSource",
    "build_default_planner",
]
