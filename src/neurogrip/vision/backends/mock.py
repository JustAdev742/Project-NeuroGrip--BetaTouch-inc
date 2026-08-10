"""Mock vision backend for simulation and tests.

Reads the ground truth that :class:`~neurogrip.hal.camera.simulated.SimulatedCamera`
attaches to each frame and emits detections and grasps from it, with configurable
degradation:

* ``confidence_noise`` — jitter on reported confidence;
* ``false_negative_rate`` — frames on which the object is simply not seen;
* ``label_error_rate`` — frames on which it is misclassified;
* ``latency_ms`` — simulated inference cost.

Perfect vision is the least useful thing to test against. Fusion, target
selection and the "AI unsure → user still in control" paths only get exercised
when vision is *sometimes wrong*, and these knobs are how the integration tests
create those situations deterministically.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from ...core.config import Config
from ...core.types import clamp
from ...hal.camera.base import Frame
from ..backend import BackendInfo, register_backend
from ..types import (
    BoundingBox,
    DepthEstimate,
    Detection,
    GraspApproach,
    GraspCandidate,
    VisionCapability,
    VisionResult,
)

__all__ = ["MockSettings", "MockVisionBackend"]

#: Labels the mock may substitute when simulating a misclassification. Chosen to
#: be plausible confusions (a can looks like a cup), because that is what a real
#: classifier gets wrong — not bottle-for-book.
_CONFUSIONS = {
    "bottle": "can",
    "can": "cup",
    "cup": "can",
    "ball": "fruit",
    "fruit": "ball",
    "box": "book",
    "book": "box",
    "pen": "key",
    "key": "pen",
}


@dataclass(frozen=True, slots=True)
class MockSettings:
    """Degradation model for the mock backend."""

    confidence_noise: float = 0.05
    false_negative_rate: float = 0.0
    label_error_rate: float = 0.0
    latency_ms: float = 4.0
    base_confidence: float = 0.86
    seed: int = 991

    @classmethod
    def from_config(cls, config: Config) -> MockSettings:
        d = cls()
        section = config.section("vision.mock")
        return cls(
            confidence_noise=section.get_float("confidence_noise", d.confidence_noise),
            false_negative_rate=section.get_float("false_negative_rate", d.false_negative_rate),
            label_error_rate=section.get_float("label_error_rate", d.label_error_rate),
            latency_ms=section.get_float("latency_ms", d.latency_ms),
            base_confidence=section.get_float("base_confidence", d.base_confidence),
            seed=section.get_int("seed", d.seed),
        )


class MockVisionBackend:
    """Ground-truth-driven backend with configurable, deterministic errors."""

    def __init__(self, config: Config | None = None, settings: MockSettings | None = None) -> None:
        self._settings = settings or (
            MockSettings.from_config(config) if config is not None else MockSettings()
        )
        self._random = random.Random(self._settings.seed)
        self._initialised = False

    def initialize(self) -> None:
        self._random = random.Random(self._settings.seed)
        self._initialised = True

    def shutdown(self) -> None:
        self._initialised = False

    @property
    def capabilities(self) -> VisionCapability:
        return (
            VisionCapability.DETECTION
            | VisionCapability.CLASSIFICATION
            | VisionCapability.GRASP
            | VisionCapability.DEPTH
        )

    def info(self) -> BackendInfo:
        return BackendInfo(
            name="mock",
            version="1.0",
            capabilities=self.capabilities,
            runtime="mock",
            degraded_reason="simulation backend — not for hardware use",
        )

    def process(self, frame: Frame) -> VisionResult:
        if not self._initialised:
            return VisionResult.empty(frame.timestamp, "mock", "backend not initialised")

        scene = frame.metadata.get("scene")
        if not isinstance(scene, dict) or not scene.get("visible", False):
            return VisionResult(
                timestamp=frame.timestamp,
                frame_index=frame.index,
                latency_ms=self._settings.latency_ms,
                backend="mock",
                capabilities=self.capabilities,
            )

        if self._random.random() < self._settings.false_negative_rate:
            return VisionResult(
                timestamp=frame.timestamp,
                frame_index=frame.index,
                latency_ms=self._settings.latency_ms,
                backend="mock",
                capabilities=self.capabilities,
            )

        x1, y1, x2, y2 = scene["bbox"]
        bbox = BoundingBox(clamp(x1), clamp(y1), clamp(x2), clamp(y2))

        label = str(scene.get("label", "unknown"))
        if self._random.random() < self._settings.label_error_rate:
            label = _CONFUSIONS.get(label, "unknown")

        confidence = clamp(
            self._settings.base_confidence + self._random.gauss(0.0, self._settings.confidence_noise)
        )

        detection = Detection(
            label=label,
            confidence=confidence,
            bbox=bbox,
            attributes={"source": "mock", "shape": str(scene.get("shape", ""))},
        )

        distance = float(scene.get("distance_m", 0.35))
        depth = DepthEstimate(
            distance_m=distance * (1.0 + self._random.gauss(0.0, 0.04)),
            confidence=0.8,
            method="sensor",
            relative_error=0.08,
        )

        grasp = self._grasp_for(scene, bbox, confidence, distance, label)
        return VisionResult(
            timestamp=frame.timestamp,
            frame_index=frame.index,
            detections=(detection,),
            grasps=(grasp,),
            depth=depth,
            latency_ms=self._settings.latency_ms,
            backend="mock",
            capabilities=self.capabilities,
        )

    def _grasp_for(
        self, scene: dict, bbox: BoundingBox, confidence: float, distance: float, label: str
    ) -> GraspCandidate:
        """Derive a plausible grasp from the ground-truth geometry."""
        cx, cy = bbox.center
        shape = str(scene.get("shape", "cylinder"))
        orientation = float(scene.get("orientation", 0.0))

        if shape == "sphere":
            # A sphere has no preferred axis; approach from above.
            angle = math.pi / 2
            approach = GraspApproach.TOP_DOWN
            width = max(bbox.width, bbox.height)
        elif shape == "flat":
            angle = 0.0
            approach = GraspApproach.TOP_DOWN
            width = min(bbox.width, bbox.height)
        else:
            # Cylinder: close across the narrow axis, perpendicular to its length.
            angle = (orientation + (0.0 if bbox.is_upright else math.pi / 2)) % math.pi
            approach = GraspApproach.SIDE
            width = min(bbox.width, bbox.height)

        return GraspCandidate(
            center_x=cx,
            center_y=cy,
            angle=angle,
            width=clamp(width),
            quality=clamp(confidence * 0.95),
            depth_m=distance,
            width_m=width * distance * 0.9,  # rough pinhole scaling
            approach=approach,
            source="mock",
            label=label,
        )


def _factory(config: Config | None = None, **_: object) -> MockVisionBackend:
    return MockVisionBackend(config)


register_backend("mock", _factory)
