"""Vision value types.

Deliberately backend-agnostic. A detector, a segmenter and a grasp network all
report into the same :class:`VisionResult`, so swapping the model changes what
*fills* these structures, never what reads them.

Image coordinates are **normalised** (``0..1``, origin top-left). Normalised
coordinates survive a change of camera resolution or a letterbox pad, which raw
pixels do not — and this stack has to run on a 160×120 synthetic scene, a 640×480
webcam and a 1280×720 Pi camera without any of the consumers noticing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, Flag, auto

from ..core.types import clamp

__all__ = [
    "BoundingBox",
    "DepthEstimate",
    "Detection",
    "GraspApproach",
    "GraspCandidate",
    "VisionCapability",
    "VisionResult",
]


class VisionCapability(Flag):
    """What a backend can produce.

    Consumers check capabilities instead of the backend's type, so the pipeline
    can ask "can anyone give me a grasp?" without knowing who is loaded.
    """

    NONE = 0
    DETECTION = auto()
    CLASSIFICATION = auto()
    DEPTH = auto()
    SEGMENTATION = auto()
    GRASP = auto()
    GESTURE = auto()
    TRACKING = auto()

    def describe(self) -> str:
        if self is VisionCapability.NONE:
            return "none"
        return "+".join(flag.name.lower() for flag in VisionCapability if flag & self and flag.name)


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Axis-aligned box in normalised image coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        # Normalise ordering so downstream maths never has to defend against it.
        # Note the temporaries: assigning through object.__setattr__ in place
        # would overwrite the first value before the second read it.
        if self.x2 < self.x1:
            low, high = self.x2, self.x1
            object.__setattr__(self, "x1", low)
            object.__setattr__(self, "x2", high)
        if self.y2 < self.y1:
            low, high = self.y2, self.y1
            object.__setattr__(self, "y1", low)
            object.__setattr__(self, "y2", high)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def aspect_ratio(self) -> float:
        """Width over height. Distinguishes an upright bottle from a lying pen."""
        return self.width / self.height if self.height > 1e-6 else 0.0

    @property
    def is_upright(self) -> bool:
        return self.aspect_ratio < 0.85

    def iou(self, other: BoundingBox) -> float:
        """Intersection over union — the tracker's association metric."""
        ix1, iy1 = max(self.x1, other.x1), max(self.y1, other.y1)
        ix2, iy2 = min(self.x2, other.x2), min(self.y2, other.y2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        intersection = (ix2 - ix1) * (iy2 - iy1)
        union = self.area + other.area - intersection
        return intersection / union if union > 1e-9 else 0.0

    def contains(self, x: float, y: float) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

    def distance_from_center(self) -> float:
        """Distance of the box centre from the image centre, in normalised units.

        The hand-mounted camera points where the user is reaching, so a box near
        the image centre is far more likely to be the intended target. This is
        the primary target-selection cue.
        """
        cx, cy = self.center
        return math.hypot(cx - 0.5, cy - 0.5)

    def scaled(self, factor: float) -> BoundingBox:
        """Grow or shrink about the centre (used for approach margins)."""
        cx, cy = self.center
        hw, hh = self.width * factor / 2, self.height * factor / 2
        return BoundingBox(
            clamp(cx - hw), clamp(cy - hh), clamp(cx + hw), clamp(cy + hh)
        )

    def to_pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        return (
            int(self.x1 * width),
            int(self.y1 * height),
            int(self.x2 * width),
            int(self.y2 * height),
        )


@dataclass(frozen=True, slots=True)
class Detection:
    """One detected object."""

    label: str
    confidence: float
    bbox: BoundingBox
    #: Stable identity across frames, assigned by the tracker (-1 = untracked).
    track_id: int = -1
    #: Frames this track has been continuously observed — a stability measure.
    age: int = 0
    #: Backend-specific extras (segmentation mask id, pose, colour, …).
    attributes: dict[str, float | str] = field(default_factory=dict)

    @property
    def is_stable(self) -> bool:
        """Whether the track has been seen long enough to be trusted.

        Acting on a single-frame detection is how a prosthesis ends up choosing a
        grasp from a motion-blurred flicker.
        """
        return self.age >= 3

    def with_track(self, track_id: int, age: int) -> Detection:
        return Detection(
            label=self.label,
            confidence=self.confidence,
            bbox=self.bbox,
            track_id=track_id,
            age=age,
            attributes=dict(self.attributes),
        )


class GraspApproach(str, Enum):
    """Direction from which the hand approaches the object."""

    TOP_DOWN = "top_down"
    SIDE = "side"
    FRONTAL = "frontal"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class GraspCandidate:
    """A proposed grasp, in image space plus whatever metric data is available.

    HGGD-style networks predict grasps directly; classical planners derive them
    from a detection. Both produce this type, so
    :mod:`neurogrip.ai.grasp` consumes one shape regardless of source.
    """

    #: Grasp centre in normalised image coordinates.
    center_x: float
    center_y: float
    #: In-plane rotation of the gripper axis, radians, 0 = horizontal.
    angle: float
    #: Required opening as a fraction of the hand's maximum aperture.
    width: float
    #: Confidence/quality score in ``[0, 1]``.
    quality: float
    #: Distance to the grasp point, metres. ``None`` when unknown.
    depth_m: float | None = None
    #: Metric gripper opening in metres, when depth is known.
    width_m: float | None = None
    approach: GraspApproach = GraspApproach.UNKNOWN
    #: Which backend/head produced this candidate.
    source: str = ""
    #: Label of the object this grasp belongs to, when the backend knows it.
    label: str = ""
    #: Unit approach vector in camera coordinates (+X right, +Y down, +Z forward),
    #: for backends that predict a full 6-DoF pose rather than a planar grasp.
    #: ``None`` from planar backends such as HGGD-MCU. Consumers must treat its
    #: absence as "unknown", never as "straight ahead" — see
    #: :mod:`neurogrip.ai.grasp.anygrasp` for why that distinction matters on a
    #: hand with no powered wrist.
    approach_vector: tuple[float, float, float] | None = None

    @property
    def center(self) -> tuple[float, float]:
        return (self.center_x, self.center_y)

    @property
    def angle_degrees(self) -> float:
        return math.degrees(self.angle)

    @property
    def is_centered(self) -> bool:
        """Whether the grasp point is near the image centre (i.e. where the user is aiming)."""
        return math.hypot(self.center_x - 0.5, self.center_y - 0.5) < 0.25

    def distance_to(self, other: GraspCandidate) -> float:
        """Combined spatial + angular distance, used for non-maximum suppression."""
        spatial = math.hypot(self.center_x - other.center_x, self.center_y - other.center_y)
        # Grasps are symmetric under 180° rotation, so wrap the angle difference
        # into [0, π/2] before comparing.
        angular = abs(self.angle - other.angle) % math.pi
        angular = min(angular, math.pi - angular)
        return spatial + angular / math.pi * 0.5


@dataclass(frozen=True, slots=True)
class DepthEstimate:
    """Distance to the target, with provenance.

    Provenance matters: a stereo measurement and a size-prior guess have very
    different error bars, and the fusion layer weights them differently.
    """

    distance_m: float
    confidence: float
    #: ``"sensor"``, ``"size_prior"``, ``"disparity"``, ``"learned"``, ``"default"``.
    method: str = "unknown"
    #: Estimated relative error, e.g. 0.3 = ±30 %.
    relative_error: float = 0.5

    @property
    def is_reachable(self) -> bool:
        """Within the working envelope of a hand at the end of a human arm."""
        return 0.05 <= self.distance_m <= 0.85


@dataclass(frozen=True, slots=True)
class VisionResult:
    """Everything the vision system knows about one frame."""

    timestamp: float
    frame_index: int = 0
    detections: tuple[Detection, ...] = field(default_factory=tuple)
    grasps: tuple[GraspCandidate, ...] = field(default_factory=tuple)
    depth: DepthEstimate | None = None
    #: Latency from frame capture to result, in milliseconds.
    latency_ms: float = 0.0
    backend: str = ""
    capabilities: VisionCapability = VisionCapability.NONE
    #: Set when the backend failed for this frame; the pipeline keeps running.
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error

    @property
    def primary(self) -> Detection | None:
        """The detection the user is most likely aiming at.

        Ranked by a blend of confidence, centrality and track stability, not by
        confidence alone — the most confident detection in the frame is often a
        large background object the user has no interest in.
        """
        if not self.detections:
            return None
        return max(self.detections, key=_target_score)

    @property
    def best_grasp(self) -> GraspCandidate | None:
        if not self.grasps:
            return None
        return max(self.grasps, key=lambda g: g.quality)

    @property
    def object_confidence(self) -> float:
        """Confidence that we understand what is in front of the hand."""
        primary = self.primary
        if primary is None:
            return 0.0
        stability = min(1.0, primary.age / 5.0) if primary.age else 0.4
        return clamp(primary.confidence * (0.6 + 0.4 * stability))

    def age(self, now: float) -> float:
        return now - self.timestamp

    def is_fresh(self, now: float, max_age: float = 0.5) -> bool:
        """Vision older than ``max_age`` must not influence a grasp decision."""
        return self.age(now) <= max_age

    @classmethod
    def empty(cls, timestamp: float, backend: str = "", error: str = "") -> VisionResult:
        return cls(timestamp=timestamp, backend=backend, error=error)


def _target_score(detection: Detection) -> float:
    """Rank a detection by how likely it is to be the user's target."""
    centrality = 1.0 - clamp(detection.bbox.distance_from_center() / 0.7)
    stability = clamp(detection.age / 5.0) if detection.age else 0.3
    # Very small boxes are usually distant background clutter.
    size = clamp(detection.bbox.area * 8.0)
    return detection.confidence * 0.45 + centrality * 0.35 + stability * 0.1 + size * 0.1
