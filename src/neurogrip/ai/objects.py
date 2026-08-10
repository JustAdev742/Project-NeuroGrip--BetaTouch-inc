"""Object affordance database.

An *affordance* is what an object offers to a hand: how it is normally held, how
hard it can be squeezed, how quickly it can be approached. This table is what
turns "the camera sees a bottle" into "a cylindrical grasp at 62 % force" — and
it is the piece a clinician or a user can safely tune, because it contains
knowledge about objects rather than code.

Every entry deliberately carries a ``max_force`` **lower** than the hand's
capability. A prosthesis that can crush an egg will eventually crush an egg. The
force ceiling comes from the object class first and the mode second; a mode may
lower it but never raise it above the entry's value.

Unknown objects fall through to :data:`DEFAULT_AFFORDANCE`, which selects a slow,
low-force power grasp — the choice that is least likely to damage anything or
surprise the user.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field

from ..core.config import Config
from ..core.logging import get_logger
from ..core.types import GraspType, clamp

__all__ = ["BUILTIN_AFFORDANCES", "DEFAULT_AFFORDANCE", "Affordance", "AffordanceDatabase"]

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class Affordance:
    """How an object class should be handled."""

    label: str
    #: Preferred grasps, best first. The planner picks the first that is
    #: geometrically feasible given the measured object size.
    grasps: tuple[GraspType, ...]
    #: Grip force ceiling in ``[0, 1]`` for this class.
    max_force: float = 0.6
    #: Approach speed multiplier; fragile things get approached slowly.
    speed_scale: float = 1.0
    #: Typical graspable width in metres, for aperture pre-shaping.
    typical_width_m: float = 0.07
    #: True for objects that deform or break under load.
    fragile: bool = False
    #: True for objects that must be held securely against gravity (a full mug).
    heavy: bool = False
    #: Free-text guidance shown in the UI's "why this grasp?" panel.
    notes: str = ""
    #: Alternative labels that map to this entry (classifier vocabulary drift).
    aliases: tuple[str, ...] = field(default_factory=tuple)

    @property
    def primary_grasp(self) -> GraspType:
        return self.grasps[0] if self.grasps else GraspType.POWER

    def force_for(self, mode_ceiling: float) -> float:
        """Effective force: the *lower* of the object limit and the mode limit."""
        return clamp(min(self.max_force, mode_ceiling))


#: Conservative fallback for anything unrecognised.
DEFAULT_AFFORDANCE = Affordance(
    label="unknown",
    grasps=(GraspType.POWER,),
    max_force=0.45,
    speed_scale=0.75,
    typical_width_m=0.07,
    notes="Object not recognised — using a slow, low-force power grasp.",
)


BUILTIN_AFFORDANCES: tuple[Affordance, ...] = (
    Affordance(
        label="bottle",
        grasps=(GraspType.CYLINDRICAL, GraspType.POWER),
        max_force=0.65,
        speed_scale=1.0,
        typical_width_m=0.07,
        heavy=True,
        notes="Wrap the barrel; a full bottle needs secure force against slip.",
        aliases=("water_bottle", "flask"),
    ),
    Affordance(
        label="cup",
        grasps=(GraspType.CYLINDRICAL, GraspType.SPHERICAL),
        max_force=0.50,
        speed_scale=0.9,
        typical_width_m=0.08,
        heavy=True,
        notes="Grip the body rather than the rim; hot contents make spills costly.",
        aliases=("mug", "glass", "tumbler"),
    ),
    Affordance(
        label="can",
        grasps=(GraspType.CYLINDRICAL, GraspType.POWER),
        max_force=0.42,
        speed_scale=1.0,
        typical_width_m=0.066,
        fragile=True,
        notes="Thin aluminium walls buckle — force is deliberately capped low.",
        aliases=("soda_can", "tin"),
    ),
    Affordance(
        label="box",
        grasps=(GraspType.POWER, GraspType.LATERAL_KEY),
        max_force=0.6,
        typical_width_m=0.12,
        notes="Flat faces suit a whole-hand press or a lateral pinch.",
        aliases=("carton", "package"),
    ),
    Affordance(
        label="ball",
        grasps=(GraspType.SPHERICAL, GraspType.POWER),
        max_force=0.55,
        speed_scale=1.1,
        typical_width_m=0.07,
        notes="Cage rather than crush; contact spread matters more than force.",
        aliases=("sphere",),
    ),
    Affordance(
        label="pen",
        grasps=(GraspType.TRIPOD, GraspType.PRECISION_PINCH),
        max_force=0.28,
        speed_scale=0.7,
        typical_width_m=0.012,
        notes="Tripod is the natural writing grip and keeps the tip controllable.",
        aliases=("pencil", "stylus", "marker"),
    ),
    Affordance(
        label="key",
        grasps=(GraspType.LATERAL_KEY, GraspType.PRECISION_PINCH),
        max_force=0.35,
        speed_scale=0.7,
        typical_width_m=0.02,
        notes="Lateral pinch transmits the turning torque a key needs.",
        aliases=("keys", "keyring"),
    ),
    Affordance(
        label="card",
        grasps=(GraspType.LATERAL_KEY, GraspType.PRECISION_PINCH),
        max_force=0.22,
        speed_scale=0.6,
        typical_width_m=0.005,
        fragile=True,
        notes="Thin and stiff: a light lateral pinch, approached slowly.",
        aliases=("credit_card", "id_card", "ticket"),
    ),
    Affordance(
        label="phone",
        grasps=(GraspType.LATERAL_KEY, GraspType.POWER),
        max_force=0.40,
        speed_scale=0.8,
        typical_width_m=0.072,
        fragile=True,
        notes="Held from the sides; screen faces must not be squeezed.",
        aliases=("smartphone", "mobile"),
    ),
    Affordance(
        label="book",
        grasps=(GraspType.POWER, GraspType.HOOK),
        max_force=0.55,
        typical_width_m=0.03,
        notes="Grip the spine edge; hook works for carrying a stack.",
        aliases=("notebook", "magazine"),
    ),
    Affordance(
        label="tool",
        grasps=(GraspType.POWER, GraspType.CYLINDRICAL),
        max_force=0.78,
        speed_scale=1.0,
        typical_width_m=0.035,
        heavy=True,
        notes="Handles are designed for a firm power grip; torque needs force.",
        aliases=("hammer", "screwdriver", "wrench"),
    ),
    Affordance(
        label="fruit",
        grasps=(GraspType.SPHERICAL, GraspType.PRECISION_PINCH),
        max_force=0.25,
        speed_scale=0.65,
        typical_width_m=0.075,
        fragile=True,
        notes="Bruises easily — the lowest force ceiling in the table.",
        aliases=("apple", "orange", "tomato", "egg"),
    ),
    Affordance(
        label="plate",
        grasps=(GraspType.LATERAL_KEY, GraspType.HOOK),
        max_force=0.5,
        speed_scale=0.7,
        typical_width_m=0.015,
        fragile=True,
        heavy=True,
        notes="Grip the rim; the flat face offers nothing to oppose.",
        aliases=("dish", "saucer"),
    ),
    Affordance(
        label="handle",
        grasps=(GraspType.CYLINDRICAL, GraspType.HOOK, GraspType.POWER),
        max_force=0.72,
        typical_width_m=0.03,
        heavy=True,
        notes="Doors and drawers need force to overcome a latch.",
        aliases=("door_handle", "lever", "bag_handle"),
    ),
    Affordance(
        label="bag",
        grasps=(GraspType.HOOK, GraspType.POWER),
        max_force=0.7,
        typical_width_m=0.02,
        heavy=True,
        notes="Hook carries load on the fingers rather than a fatiguing pinch.",
        aliases=("carrier_bag", "shopping_bag"),
    ),
)


class AffordanceDatabase:
    """Lookup from an object label to its handling policy."""

    def __init__(self, affordances: Mapping[str, Affordance] | None = None) -> None:
        self._by_label: dict[str, Affordance] = {}
        source = tuple(affordances.values()) if affordances else BUILTIN_AFFORDANCES
        for affordance in source:
            self.add(affordance)

    def add(self, affordance: Affordance) -> None:
        """Register an affordance under its label and every alias."""
        self._by_label[affordance.label.lower()] = affordance
        for alias in affordance.aliases:
            self._by_label[alias.lower()] = affordance

    def get(self, label: str) -> Affordance:
        """Affordance for ``label``; the conservative default when unknown."""
        return self._by_label.get((label or "").lower().strip(), DEFAULT_AFFORDANCE)

    def knows(self, label: str) -> bool:
        return (label or "").lower().strip() in self._by_label

    def __iter__(self) -> Iterator[Affordance]:
        # Dedupe: aliases point at shared instances.
        seen: set[int] = set()
        for affordance in self._by_label.values():
            if id(affordance) not in seen:
                seen.add(id(affordance))
                yield affordance

    def __len__(self) -> int:
        return sum(1 for _ in self)

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(sorted(a.label for a in self))

    def validate_against(self, classes: tuple[str, ...]) -> tuple[str, ...]:
        """Return model classes with no affordance entry.

        Called at startup and reported in the log: a class the model can predict
        but the database has never heard of will silently fall back to the
        default grasp, and that is worth knowing about before it happens in use.
        """
        return tuple(c for c in classes if not self.knows(c) and c != "unknown")

    @classmethod
    def from_config(cls, config: Config) -> AffordanceDatabase:
        """Load ``[affordances.<label>]`` tables, merged over the built-ins."""
        database = cls()
        for label, section in config.sections("affordances").items():
            grasp_names = section.get_list("grasps", [])
            grasps: list[GraspType] = []
            for name in grasp_names:
                try:
                    grasps.append(GraspType(str(name)))
                except ValueError:
                    log.warning("unknown grasp in affordance", label=label, grasp=name)
            existing = database.get(label)
            database.add(
                Affordance(
                    label=label,
                    grasps=tuple(grasps) or existing.grasps,
                    max_force=section.get_float("max_force", existing.max_force),
                    speed_scale=section.get_float("speed_scale", existing.speed_scale),
                    typical_width_m=section.get_float("typical_width_m", existing.typical_width_m),
                    fragile=section.get_bool("fragile", existing.fragile),
                    heavy=section.get_bool("heavy", existing.heavy),
                    notes=section.get_str("notes", existing.notes),
                    aliases=tuple(str(a) for a in section.get_list("aliases", [])),
                )
            )
        return database
