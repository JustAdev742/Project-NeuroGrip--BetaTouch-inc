"""Grasp planning.

The planner decides **how** to grasp. It never decides **whether** — that comes
from the user's EMG intent and is enforced in :mod:`neurogrip.fusion`.

Construct the default chain with :func:`build_default_planner`; swap or extend it
by registering your own :class:`~neurogrip.ai.grasp.base.GraspPlanner` and listing
it in ``[ai] planners``.
"""

from __future__ import annotations

from collections.abc import Callable

from ...control.grips import GripLibrary
from ...control.kinematics import HandKinematics
from ...core.config import Config
from ...core.logging import get_logger
from ..objects import AffordanceDatabase
from .base import GraspContext, GraspPlan, GraspPlanner, PlanSource
from .composite import CompositeGraspPlanner
from .heuristic import HeuristicGraspPlanner
from .hggd import HggdGraspPlanner

__all__ = [
    "CompositeGraspPlanner",
    "GraspContext",
    "GraspPlan",
    "GraspPlanner",
    "HeuristicGraspPlanner",
    "HggdGraspPlanner",
    "PlanSource",
    "available_planners",
    "build_default_planner",
    "register_planner",
]

log = get_logger(__name__)

PlannerFactory = Callable[[GripLibrary, AffordanceDatabase, HandKinematics], GraspPlanner]

_REGISTRY: dict[str, PlannerFactory] = {
    "hggd": lambda grips, affordances, kinematics: HggdGraspPlanner(grips, affordances, kinematics),
    "heuristic": lambda grips, affordances, kinematics: HeuristicGraspPlanner(
        grips, affordances, kinematics
    ),
}


def register_planner(name: str, factory: PlannerFactory) -> None:
    """Register a planner implementation for use in ``[ai] planners``."""
    _REGISTRY[name] = factory


def available_planners() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def build_default_planner(
    config: Config,
    grips: GripLibrary,
    affordances: AffordanceDatabase,
    kinematics: HandKinematics | None = None,
) -> CompositeGraspPlanner:
    """Build the configured planner chain.

    ``[ai] planners`` lists names in priority order; unknown names are logged and
    skipped. An empty or fully-unknown list still yields a working composite,
    because the composite's own safe default is the guaranteed floor.
    """
    kinematics = kinematics or HandKinematics()
    names = [str(n) for n in config.get_list("ai.planners", ["hggd", "heuristic"])]

    planners: list[GraspPlanner] = []
    for name in names:
        factory = _REGISTRY.get(name)
        if factory is None:
            log.warning("unknown grasp planner; skipping", planner=name, available=available_planners())
            continue
        planners.append(factory(grips, affordances, kinematics))

    if not planners:
        log.warning("no grasp planners configured; only the safe default will be used")

    return CompositeGraspPlanner(
        tuple(planners),
        grips,
        kinematics,
        min_confidence=config.get_float("ai.min_plan_confidence", 0.35),
    )
