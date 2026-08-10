"""AnyGrasp backend adapter.

AnyGrasp (Fang et al., T-RO 2023) predicts 6-DoF grasp poses from a point cloud.
It is a strong model and a poor fit for *this* hardware, for two reasons worth
stating plainly rather than discovering during a demonstration:

1. **It needs a depth sensor.** The reference build has a single RGB camera.
   Monocular depth from size priors (:mod:`neurogrip.vision.depth`) is good
   enough to choose a grip; it is not a point cloud.
2. **It needs a licence and a CUDA runtime.** Neither belongs in a repository
   that must run on a battery-powered SBC and in CI with no install step.

So this module is an *adapter*, not an implementation. It exists to prove — and
keep proving, because it is exercised by the tests — that the vision layer is not
built around HGGD-MCU. Anything that can produce grasp candidates plugs in here
with no change to fusion, planning, or control.

Behaviour when the runtime is absent is deliberate: constructing the backend
raises :class:`~neurogrip.core.errors.ModelLoadError` with an actionable message,
and the factory falls back to the configured alternative. It does *not* silently
return zero grasps, which would look identical to "the camera sees nothing" and
would be diagnosed as a hardware fault.

To supply a real implementation, provide an object with a ``predict`` method
matching :class:`PointCloudGraspModel` and pass it as ``model``; the conversion
from its output to :class:`~neurogrip.vision.types.GraspCandidate` below is
complete and tested against a stub.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ...core.config import Config
from ...core.errors import ModelLoadError
from ...core.logging import get_logger
from ...core.types import clamp
from ...hal.camera.base import Frame
from ..backend import BackendInfo, register_backend
from ..types import (
    GraspApproach,
    GraspCandidate,
    VisionCapability,
    VisionResult,
)

__all__ = ["AnyGraspBackend", "AnyGraspSettings", "PointCloudGraspModel", "SixDofGrasp"]

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class SixDofGrasp:
    """One grasp as a 6-DoF model reports it.

    Mirrors the AnyGrasp output record. Kept as a plain dataclass so an
    integrator can construct it from any SDK without importing ours into theirs.
    """

    #: Grasp centre in camera coordinates, metres.
    position: tuple[float, float, float]
    #: Unit approach vector in camera coordinates.
    approach: tuple[float, float, float]
    #: Gripper opening, metres.
    width_m: float
    score: float
    label: str = ""


@runtime_checkable
class PointCloudGraspModel(Protocol):
    """What this backend needs from a 6-DoF grasp model."""

    def predict(self, points, colors=None) -> list[SixDofGrasp]:
        """Return grasp poses for an ``(N, 3)`` point cloud in camera frame."""
        ...


@dataclass(frozen=True, slots=True)
class AnyGraspSettings:
    """Configuration for the AnyGrasp adapter."""

    #: Minimum model score to emit a candidate.
    min_score: float = 0.30
    #: Maximum candidates returned per frame, best first.
    max_grasps: int = 8
    #: Working volume, metres. Grasps outside it belong to the background.
    min_distance_m: float = 0.05
    max_distance_m: float = 1.20
    #: Camera intrinsics, needed to project a 3-D grasp back into image space so
    #: the rest of the stack can reason about it in the same terms as HGGD.
    focal_length_px: float = 533.0
    image_width: int = 640
    image_height: int = 480

    @classmethod
    def from_config(cls, config: Config) -> AnyGraspSettings:
        d = cls()
        section = config.section("vision.anygrasp")
        return cls(
            min_score=section.get_float("min_score", d.min_score),
            max_grasps=section.get_int("max_grasps", d.max_grasps),
            min_distance_m=section.get_float("min_distance_m", d.min_distance_m),
            max_distance_m=section.get_float("max_distance_m", d.max_distance_m),
            focal_length_px=section.get_float("focal_length_px", d.focal_length_px),
            image_width=config.get_int("camera.width", d.image_width),
            image_height=config.get_int("camera.height", d.image_height),
        )


class AnyGraspBackend:
    """Turns 6-DoF model output into the stack's common grasp representation."""

    def __init__(
        self,
        settings: AnyGraspSettings | None = None,
        *,
        model: PointCloudGraspModel | None = None,
    ) -> None:
        self._settings = settings or AnyGraspSettings()
        if model is None:
            raise ModelLoadError(
                "AnyGrasp needs a depth sensor and its licensed runtime, neither of "
                "which ships with this project. Set vision.backend = 'hggd_mcu' for "
                "the bundled model, or pass a PointCloudGraspModel to use AnyGrasp.",
                context={"backend": "anygrasp"},
            )
        self._model = model
        self.frames = 0

    @property
    def name(self) -> str:
        return "anygrasp"

    @property
    def capabilities(self) -> VisionCapability:
        # No classification head: AnyGrasp predicts where to grasp, not what the
        # object is. Declaring DETECTION here would make the fusion layer expect
        # a label that never arrives.
        return VisionCapability.GRASP | VisionCapability.DEPTH

    def info(self) -> BackendInfo:
        return BackendInfo(
            name="anygrasp",
            version="adapter-1",
            capabilities=self.capabilities,
            runtime="external",
        )

    def initialize(self) -> None:
        """Nothing to load: the model was supplied already constructed."""

    def shutdown(self) -> None:
        closer = getattr(self._model, "close", None)
        if callable(closer):
            closer()

    def process(self, frame: Frame) -> VisionResult:
        """Run the model and convert its output.

        The point cloud is expected on the frame as ``frame.point_cloud``; a
        frame without one yields an empty result rather than an error, because an
        RGB camera producing RGB frames is not a fault.
        """
        self.frames += 1
        points = getattr(frame, "point_cloud", None)
        if points is None:
            return VisionResult.empty(frame.timestamp, self.name, "no depth data on this frame")

        grasps = self._model.predict(points)
        return VisionResult(
            timestamp=frame.timestamp,
            backend=self.name,
            capabilities=self.capabilities,
            grasps=self._convert(grasps),
        )

    # -- conversion -----------------------------------------------------------

    def _convert(self, grasps: list[SixDofGrasp]) -> tuple[GraspCandidate, ...]:
        settings = self._settings
        converted: list[GraspCandidate] = []

        for grasp in grasps:
            if grasp.score < settings.min_score:
                continue
            x, y, z = grasp.position
            if not settings.min_distance_m <= z <= settings.max_distance_m:
                continue

            # Pinhole projection back into normalised image coordinates, so
            # downstream code can treat these exactly like planar candidates.
            u = (x * settings.focal_length_px / z) + settings.image_width / 2.0
            v = (y * settings.focal_length_px / z) + settings.image_height / 2.0
            center_x = clamp(u / settings.image_width)
            center_y = clamp(v / settings.image_height)

            # Apparent opening, for backends and planners that reason in image
            # space rather than metres.
            width_px = grasp.width_m * settings.focal_length_px / z
            width = clamp(width_px / settings.image_width)

            converted.append(
                GraspCandidate(
                    center_x=center_x,
                    center_y=center_y,
                    angle=0.0,
                    width=width,
                    quality=clamp(grasp.score),
                    depth_m=z,
                    width_m=grasp.width_m,
                    approach=_approach_from_vector(grasp.approach),
                    source="anygrasp",
                    label=grasp.label,
                    approach_vector=grasp.approach,
                )
            )

        converted.sort(key=lambda c: c.quality, reverse=True)
        return tuple(converted[: settings.max_grasps])


def _approach_from_vector(vector: tuple[float, float, float]) -> GraspApproach:
    """Classify an approach vector into the coarse enum the UI displays."""
    magnitude = sum(component * component for component in vector) ** 0.5
    if magnitude < 1e-9:
        return GraspApproach.UNKNOWN
    _, down, forward = (component / magnitude for component in vector)
    if down > 0.6:
        return GraspApproach.TOP_DOWN
    if forward > 0.6:
        return GraspApproach.FRONTAL
    return GraspApproach.SIDE


def _build(config: Config) -> AnyGraspBackend:
    return AnyGraspBackend(AnyGraspSettings.from_config(config))


register_backend("anygrasp", _build)
