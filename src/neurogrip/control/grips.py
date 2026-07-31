"""Grip preset library.

Maps each :class:`~neurogrip.core.types.GraspType` onto a concrete hand pose plus
the speed and force appropriate to it. Both halves of the system use this:

* **Manual mode** — the user selects a preset directly from the touchscreen.
* **AI Assist** — the grasp planner chooses a :class:`GraspType`, and this
  library turns it into joint targets.

Presets are data, not code: they load from ``config/grasps.toml`` and can be
retuned for a different hand without touching the planner. The built-in table
below is the fallback and the documentation of what each grasp means.

Each preset carries a ``force`` and ``speed`` because they are properties of the
*grasp*, not of the mode: a precision pinch on a contact lens should be slow and
light whatever mode you are in, and Sports Mode scales from there rather than
overriding it.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field

from ..core.config import Config
from ..core.logging import get_logger
from ..core.types import Finger, GraspType, HandPose, clamp

__all__ = ["BUILTIN_PRESETS", "GripLibrary", "GripPreset"]

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class GripPreset:
    """A named hand configuration with its execution parameters."""

    grasp: GraspType
    pose: HandPose
    #: Normalised grip force ceiling in ``[0, 1]``.
    force: float = 0.55
    #: Speed multiplier applied to the mode's base velocity.
    speed: float = 1.0
    #: Pose to move through on the way in; ``None`` goes direct.
    preshape: HandPose | None = None
    description: str = ""
    #: Fingers that must reach their target for the grasp to count as achieved.
    critical_fingers: tuple[Finger, ...] = field(
        default_factory=lambda: (Finger.THUMB, Finger.INDEX)
    )

    @property
    def label(self) -> str:
        return self.grasp.label

    def scaled(self, *, force: float | None = None, speed: float | None = None) -> GripPreset:
        """Return a copy with force and/or speed overridden."""
        return GripPreset(
            grasp=self.grasp,
            pose=self.pose,
            force=clamp(self.force if force is None else force),
            speed=max(0.05, self.speed if speed is None else speed),
            preshape=self.preshape,
            description=self.description,
            critical_fingers=self.critical_fingers,
        )

    def partial(self, fraction: float) -> HandPose:
        """Pose interpolated from open towards the target by ``fraction``.

        Used for proportional control: the user's muscle effort maps directly
        onto how far into the grip the hand travels, which is what makes the
        shared-control loop feel like *their* hand rather than an automaton.
        """
        return HandPose.open_hand().blend(self.pose, clamp(fraction))


def _pose(thumb: float, index: float, middle: float, ring: float, pinky: float) -> HandPose:
    return HandPose((thumb, index, middle, ring, pinky))


#: Built-in presets. Values are tuned for the reference hand described in
#: ``docs/hardware.md``; retune in ``config/grasps.toml`` for other geometry.
BUILTIN_PRESETS: dict[GraspType, GripPreset] = {
    GraspType.OPEN: GripPreset(
        grasp=GraspType.OPEN,
        pose=_pose(0.0, 0.0, 0.0, 0.0, 0.0),
        force=0.25,
        speed=1.15,
        description="Fully extended — the safe default and the release pose.",
    ),
    GraspType.RELAXED: GripPreset(
        grasp=GraspType.RELAXED,
        pose=_pose(0.18, 0.22, 0.22, 0.20, 0.18),
        force=0.20,
        speed=0.8,
        description="Natural resting curl. What the hand idles in so it does not "
        "look conspicuously rigid.",
    ),
    GraspType.POWER: GripPreset(
        grasp=GraspType.POWER,
        pose=_pose(0.85, 0.90, 0.92, 0.90, 0.86),
        force=0.75,
        speed=1.0,
        preshape=_pose(0.10, 0.15, 0.15, 0.15, 0.15),
        description="Whole-hand wrap. The default when the object is unknown but "
        "the user clearly wants to grab something.",
        critical_fingers=(Finger.THUMB, Finger.INDEX, Finger.MIDDLE),
    ),
    GraspType.CYLINDRICAL: GripPreset(
        grasp=GraspType.CYLINDRICAL,
        pose=_pose(0.72, 0.82, 0.84, 0.82, 0.78),
        force=0.62,
        speed=0.95,
        preshape=_pose(0.15, 0.10, 0.10, 0.12, 0.15),
        description="Wrap around a bottle, can or handle. Thumb opposes the "
        "finger group rather than closing into the palm.",
        critical_fingers=(Finger.THUMB, Finger.INDEX, Finger.MIDDLE),
    ),
    GraspType.SPHERICAL: GripPreset(
        grasp=GraspType.SPHERICAL,
        pose=_pose(0.62, 0.66, 0.68, 0.66, 0.62),
        force=0.50,
        speed=0.9,
        preshape=_pose(0.05, 0.05, 0.05, 0.05, 0.05),
        description="Cage a ball or fruit. Even closure across all digits so "
        "contact points are distributed.",
        critical_fingers=(Finger.THUMB, Finger.INDEX, Finger.MIDDLE, Finger.RING),
    ),
    GraspType.PRECISION_PINCH: GripPreset(
        grasp=GraspType.PRECISION_PINCH,
        pose=_pose(0.78, 0.80, 0.30, 0.20, 0.15),
        force=0.32,
        speed=0.7,
        preshape=_pose(0.30, 0.25, 0.30, 0.20, 0.15),
        description="Thumb-to-index pinch for small, light or fragile objects.",
        critical_fingers=(Finger.THUMB, Finger.INDEX),
    ),
    GraspType.TRIPOD: GripPreset(
        grasp=GraspType.TRIPOD,
        pose=_pose(0.74, 0.76, 0.74, 0.22, 0.18),
        force=0.40,
        speed=0.8,
        preshape=_pose(0.28, 0.25, 0.25, 0.20, 0.15),
        description="Thumb with index and middle — pens, small tools, cutlery.",
        critical_fingers=(Finger.THUMB, Finger.INDEX, Finger.MIDDLE),
    ),
    GraspType.LATERAL_KEY: GripPreset(
        grasp=GraspType.LATERAL_KEY,
        pose=_pose(0.66, 0.88, 0.88, 0.86, 0.84),
        force=0.45,
        speed=0.85,
        description="Thumb presses against the side of the curled index — keys, "
        "cards, a zip pull.",
        critical_fingers=(Finger.THUMB, Finger.INDEX),
    ),
    GraspType.HOOK: GripPreset(
        grasp=GraspType.HOOK,
        pose=_pose(0.05, 0.88, 0.90, 0.88, 0.84),
        force=0.70,
        speed=1.0,
        description="Four-finger hook with the thumb clear — carrier bags, "
        "handles. Load is carried by the fingers, not a pinch.",
        critical_fingers=(Finger.INDEX, Finger.MIDDLE, Finger.RING),
    ),
    GraspType.POINT: GripPreset(
        grasp=GraspType.POINT,
        pose=_pose(0.55, 0.00, 0.90, 0.90, 0.88),
        force=0.35,
        speed=1.1,
        description="Index extended — touchscreens, buttons, keyboards.",
        critical_fingers=(Finger.INDEX,),
    ),
    GraspType.FIST: GripPreset(
        grasp=GraspType.FIST,
        pose=_pose(0.90, 0.98, 0.98, 0.98, 0.96),
        force=0.80,
        speed=1.05,
        description="Full closure. Maximum force; not used by the planner, "
        "available manually.",
        critical_fingers=(Finger.INDEX, Finger.MIDDLE, Finger.RING, Finger.PINKY),
    ),
}


class GripLibrary:
    """Collection of grip presets, loadable from configuration."""

    def __init__(self, presets: Mapping[GraspType, GripPreset] | None = None) -> None:
        self._presets: dict[GraspType, GripPreset] = dict(presets or BUILTIN_PRESETS)

    # -- access ---------------------------------------------------------------

    def get(self, grasp: GraspType) -> GripPreset:
        """Preset for ``grasp``, falling back to POWER then OPEN.

        Never raises: a missing preset must not prevent the hand from moving.
        """
        preset = self._presets.get(grasp)
        if preset is not None:
            return preset
        log.warning("no preset for grasp; using power grip", grasp=grasp.value)
        return self._presets.get(GraspType.POWER, BUILTIN_PRESETS[GraspType.OPEN])

    def pose(self, grasp: GraspType) -> HandPose:
        return self.get(grasp).pose

    def __contains__(self, grasp: GraspType) -> bool:
        return grasp in self._presets

    def __iter__(self) -> Iterator[GripPreset]:
        return iter(self._presets.values())

    def __len__(self) -> int:
        return len(self._presets)

    @property
    def available(self) -> tuple[GraspType, ...]:
        return tuple(self._presets)

    def add(self, preset: GripPreset) -> None:
        """Add or replace a preset at runtime (used by the calibration UI)."""
        self._presets[preset.grasp] = preset

    def scaled_for_mode(self, speed_scale: float, force_scale: float) -> GripLibrary:
        """Return a copy with every preset scaled — how modes retune the library."""
        return GripLibrary(
            {
                grasp: preset.scaled(
                    force=preset.force * force_scale, speed=preset.speed * speed_scale
                )
                for grasp, preset in self._presets.items()
            }
        )

    # -- construction ---------------------------------------------------------

    @classmethod
    def from_config(cls, config: Config) -> GripLibrary:
        """Load presets from a ``[grasps.<name>]`` configuration tree.

        Unknown grasp names are logged and skipped rather than raising: a config
        file written for a newer build must not stop this one from starting.
        """
        presets = dict(BUILTIN_PRESETS)
        for name, section in config.sections("grasps").items():
            try:
                grasp = GraspType(name)
            except ValueError:
                log.warning("unknown grasp in configuration; ignoring", grasp=name)
                continue

            builtin = presets.get(grasp)
            pose_map = section.get("pose")
            if isinstance(pose_map, dict):
                pose = HandPose.from_mapping(pose_map)
            elif isinstance(pose_map, list):
                pose = HandPose.from_iterable(pose_map)
            elif builtin is not None:
                pose = builtin.pose
            else:
                log.warning("grasp has no pose and no built-in default", grasp=name)
                continue

            preshape = None
            preshape_map = section.get("preshape")
            if isinstance(preshape_map, dict):
                preshape = HandPose.from_mapping(preshape_map)
            elif isinstance(preshape_map, list):
                preshape = HandPose.from_iterable(preshape_map)
            elif builtin is not None:
                preshape = builtin.preshape

            critical = builtin.critical_fingers if builtin else (Finger.THUMB, Finger.INDEX)
            raw_critical = section.get("critical_fingers")
            if isinstance(raw_critical, list):
                critical = tuple(Finger.parse(f) for f in raw_critical)

            presets[grasp] = GripPreset(
                grasp=grasp,
                pose=pose,
                force=section.get_float("force", builtin.force if builtin else 0.55),
                speed=section.get_float("speed", builtin.speed if builtin else 1.0),
                preshape=preshape,
                description=section.get_str("description", builtin.description if builtin else ""),
                critical_fingers=critical,
            )

        return cls(presets)
