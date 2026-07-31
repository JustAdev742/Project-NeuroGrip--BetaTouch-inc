"""Core value types shared by every subsystem.

Everything in this module is a plain, immutable (``frozen=True``) data type with no
dependency on hardware, I/O or configuration. Keeping the vocabulary types free of
behaviour means the EMG code, the vision code and the firmware bridge can all talk
about "a hand pose" without depending on each other.

Conventions used throughout the stack:

* **Finger closure** is a normalised scalar in ``[0, 1]`` where ``0.0`` is fully
  open (extended) and ``1.0`` is fully closed (flexed). Servo angles, tendon
  travel and pulse widths are hardware details handled in
  :mod:`neurogrip.control.kinematics`.
* **Confidence** is a probability-like scalar in ``[0, 1]``.
* **Time** is monotonic seconds from :class:`neurogrip.core.clock.Clock`, never
  ``time.time()``, so that simulation and replay are deterministic.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum, IntEnum

__all__ = [
    "FINGER_COUNT",
    "Finger",
    "FingerVector",
    "GraspType",
    "HandPose",
    "IntentKind",
    "ModeId",
    "clamp",
    "lerp",
    "normalise",
]


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp ``value`` into ``[low, high]``.

    Used pervasively; a single implementation avoids subtly different clamping
    behaviour between the control loop and the UI.
    """
    if value < low:
        return low
    if value > high:
        return high
    return value


def lerp(a: float, b: float, t: float) -> float:
    """Linearly interpolate from ``a`` to ``b`` by ``t`` (``t`` is clamped)."""
    return a + (b - a) * clamp(t)


def normalise(value: float, low: float, high: float) -> float:
    """Map ``value`` from ``[low, high]`` onto ``[0, 1]``, clamped.

    Returns ``0.0`` for a degenerate range rather than raising, because sensor
    calibration can legitimately produce ``low == high`` before the user has
    completed a calibration run.
    """
    span = high - low
    if abs(span) < 1e-12:
        return 0.0
    return clamp((value - low) / span)


class Finger(IntEnum):
    """The five actuated digits, ordered thumb-to-pinky.

    The integer values are part of the wire protocol shared with the ESP32
    firmware (see ``docs/protocol.md``) and must not be reordered.
    """

    THUMB = 0
    INDEX = 1
    MIDDLE = 2
    RING = 3
    PINKY = 4

    @property
    def label(self) -> str:
        """Human-readable name for UI surfaces."""
        return self.name.capitalize()

    @classmethod
    def parse(cls, value: int | str | Finger) -> Finger:
        """Coerce an int, name or :class:`Finger` into a :class:`Finger`."""
        if isinstance(value, Finger):
            return value
        if isinstance(value, int):
            return cls(value)
        return cls[value.strip().upper()]


FINGER_COUNT = len(Finger)

#: A per-finger closure vector in thumb-to-pinky order, each element in ``[0, 1]``.
FingerVector = tuple[float, float, float, float, float]


@dataclass(frozen=True, slots=True)
class HandPose:
    """An immutable, validated per-finger closure vector.

    ``HandPose`` is the single currency for "where the hand should be" and "where
    the hand is". Trajectory generation, grip presets, grasp plans and the UI all
    exchange poses rather than raw servo angles.
    """

    values: FingerVector = (0.0, 0.0, 0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if len(self.values) != FINGER_COUNT:
            raise ValueError(f"HandPose requires {FINGER_COUNT} values, got {len(self.values)}")
        clamped = tuple(clamp(float(v)) for v in self.values)
        if clamped != self.values:
            # frozen dataclass: bypass the setattr guard for normalisation only.
            object.__setattr__(self, "values", clamped)

    # -- construction helpers -------------------------------------------------

    @classmethod
    def open_hand(cls) -> HandPose:
        """Fully extended hand — the mechanically safe default pose."""
        return cls((0.0, 0.0, 0.0, 0.0, 0.0))

    @classmethod
    def closed_hand(cls) -> HandPose:
        """Fully flexed hand."""
        return cls((1.0, 1.0, 1.0, 1.0, 1.0))

    @classmethod
    def uniform(cls, value: float) -> HandPose:
        """All fingers at the same closure."""
        v = clamp(value)
        return cls((v, v, v, v, v))

    @classmethod
    def from_iterable(cls, values: Iterable[float]) -> HandPose:
        """Build a pose from any 5-element iterable."""
        seq = tuple(float(v) for v in values)
        if len(seq) != FINGER_COUNT:
            raise ValueError(f"expected {FINGER_COUNT} values, got {len(seq)}")
        return cls(seq)  # type: ignore[arg-type]

    @classmethod
    def from_mapping(cls, mapping: dict[str, float], default: float = 0.0) -> HandPose:
        """Build a pose from a ``{"thumb": 0.8, ...}`` mapping (config-friendly)."""
        values = []
        for finger in Finger:
            key = finger.name.lower()
            values.append(float(mapping.get(key, mapping.get(finger.name, default))))
        return cls(tuple(values))  # type: ignore[arg-type]

    # -- accessors ------------------------------------------------------------

    def __getitem__(self, finger: int | Finger) -> float:
        return self.values[int(finger)]

    def __iter__(self) -> Iterator[float]:
        return iter(self.values)

    def __len__(self) -> int:
        return FINGER_COUNT

    def as_dict(self) -> dict[str, float]:
        """Serialise to a ``{"thumb": 0.0, ...}`` mapping."""
        return {f.name.lower(): self.values[int(f)] for f in Finger}

    # -- algebra --------------------------------------------------------------

    def with_finger(self, finger: int | Finger, value: float) -> HandPose:
        """Return a copy with a single finger changed."""
        values = list(self.values)
        values[int(finger)] = clamp(value)
        return HandPose(tuple(values))  # type: ignore[arg-type]

    def blend(self, other: HandPose, t: float) -> HandPose:
        """Interpolate towards ``other`` by ``t`` in ``[0, 1]``."""
        return HandPose(
            tuple(lerp(a, b, t) for a, b in zip(self.values, other.values))  # type: ignore[arg-type]
        )

    def scaled(self, factor: float) -> HandPose:
        """Scale every finger closure by ``factor`` (clamped)."""
        return HandPose(tuple(clamp(v * factor) for v in self.values))  # type: ignore[arg-type]

    def masked(self, fingers: Sequence[Finger], other: HandPose) -> HandPose:
        """Take ``fingers`` from ``other`` and keep the rest from ``self``.

        Used by finger-isolation training exercises and by partial grasps.
        """
        values = list(self.values)
        for finger in fingers:
            values[int(finger)] = other[finger]
        return HandPose(tuple(values))  # type: ignore[arg-type]

    def max_difference(self, other: HandPose) -> float:
        """Largest per-finger absolute difference — the natural "distance to target"."""
        return max(abs(a - b) for a, b in zip(self.values, other.values))

    def mean_difference(self, other: HandPose) -> float:
        """Mean per-finger absolute difference, used for accuracy scoring."""
        return sum(abs(a - b) for a, b in zip(self.values, other.values)) / FINGER_COUNT

    def is_close(self, other: HandPose, tolerance: float = 0.02) -> bool:
        """True when every finger is within ``tolerance`` of ``other``."""
        return self.max_difference(other) <= tolerance

    @property
    def aperture(self) -> float:
        """Approximate opening of the hand in ``[0, 1]`` (1.0 = wide open).

        Derived from the index/middle pair and the thumb, which dominate the
        functional opening between the fingertips.
        """
        grip = (self[Finger.INDEX] + self[Finger.MIDDLE]) * 0.5
        return clamp(1.0 - max(grip, self[Finger.THUMB]))

    def __str__(self) -> str:  # pragma: no cover - display helper
        return "[" + " ".join(f"{v:.2f}" for v in self.values) + "]"


class GraspType(str, Enum):
    """Taxonomy of supported grasps.

    The set is a pragmatic subset of the Cutkosky/Feix grasp taxonomies, chosen
    for what five independently tendon-driven fingers can actually achieve
    without an actuated thumb-opposition joint. New grasps are added here and in
    ``config/grasps.toml``; no other module needs to change.
    """

    OPEN = "open"
    RELAXED = "relaxed"
    POWER = "power"
    CYLINDRICAL = "cylindrical"
    SPHERICAL = "spherical"
    PRECISION_PINCH = "precision_pinch"
    TRIPOD = "tripod"
    LATERAL_KEY = "lateral_key"
    HOOK = "hook"
    POINT = "point"
    FIST = "fist"

    @property
    def label(self) -> str:
        """Human-readable label for the touchscreen."""
        return self.value.replace("_", " ").title()

    @property
    def is_precision(self) -> bool:
        """Precision grasps use lower force and slower approach by default."""
        return self in (GraspType.PRECISION_PINCH, GraspType.TRIPOD, GraspType.LATERAL_KEY)


class IntentKind(str, Enum):
    """What the user's muscles are asking for.

    Intent is deliberately coarse. Fine-grained decisions ("which grasp?") belong
    to the AI layer; the EMG layer only answers "does the user want to do
    something, and roughly what?".
    """

    #: No meaningful muscle activity. Never triggers motion.
    REST = "rest"
    #: Flexion — the user wants to close/grasp.
    CLOSE = "close"
    #: Extension — the user wants to open/release.
    OPEN = "open"
    #: Sustained activation holding the current pose.
    HOLD = "hold"
    #: Deliberate co-contraction: abort whatever is happening, immediately.
    CANCEL = "cancel"
    #: Double-pulse gesture used to cycle modes from the wrist, hands-free.
    TOGGLE = "toggle"
    #: Signal present but not classifiable (bad contact, noise, novel pattern).
    UNKNOWN = "unknown"

    @property
    def requests_motion(self) -> bool:
        """True for intents that legitimately command hand movement."""
        return self in (IntentKind.CLOSE, IntentKind.OPEN, IntentKind.HOLD)


class ModeId(str, Enum):
    """Identifiers of the operating modes.

    Kept as a closed enum (rather than free strings) so that the mode manager,
    the UI and the configuration schema cannot drift apart.
    """

    MANUAL = "manual"
    AI_ASSIST = "ai_assist"
    SPORTS = "sports"
    TRAINING = "training"

    @property
    def label(self) -> str:
        return self.value.replace("_", " ").title()

    @property
    def ai_enabled(self) -> bool:
        """Whether the assistive pipeline may contribute in this mode.

        The UI reads this to render the prominent "AI DISABLED" banner required
        by the manual-mode specification.
        """
        return self in (ModeId.AI_ASSIST, ModeId.SPORTS)


@dataclass(frozen=True, slots=True)
class Range:
    """An inclusive numeric range with clamping helpers.

    Used for calibration bounds, servo limits and training difficulty windows.
    """

    low: float
    high: float

    def __post_init__(self) -> None:
        if self.high < self.low:
            object.__setattr__(self, "low", self.high)
            object.__setattr__(self, "high", self.low)

    @property
    def span(self) -> float:
        return self.high - self.low

    def clamp(self, value: float) -> float:
        return min(max(value, self.low), self.high)

    def contains(self, value: float) -> bool:
        return self.low <= value <= self.high

    def normalise(self, value: float) -> float:
        return normalise(value, self.low, self.high)

    def expanded(self, value: float) -> Range:
        """Return a range grown to include ``value`` (running min/max tracking)."""
        return Range(min(self.low, value), max(self.high, value))


@dataclass(frozen=True, slots=True)
class Annotated:
    """A value carried alongside the reasoning that produced it.

    Explainability is a first-class requirement for a shared-control device: the
    user must always be able to see *why* the hand did what it did. Components
    that make decisions attach ``reasons`` so the UI and the black-box recorder
    can surface them verbatim.
    """

    value: float
    reasons: tuple[str, ...] = field(default_factory=tuple)

    def with_reason(self, reason: str) -> Annotated:
        return replace(self, reasons=(*self.reasons, reason))
