"""Computer vision: object understanding and grasp proposal.

Structure::

    CameraSource (HAL)
        │  Frame (raw bytes + format)
        ▼
    VisionPipeline
        ├─ preprocess    letterbox, greyscale, coordinate bookkeeping
        ├─ VisionBackend ← swappable: hggd_mcu | onnx_detector | mock | null
        ├─ ObjectTracker temporal identity + label voting
        └─ depth         monocular size-prior distance estimate
        ▼
    VisionResult (detections, grasps, depth, capabilities)

The pipeline knows nothing about which model is loaded. Backends declare
:class:`~neurogrip.vision.types.VisionCapability` flags, and consumers query
those — so adding segmentation or gesture recognition later means shipping a
backend that sets the flag, not editing the pipeline.

The configured backend for this build is ``hggd_mcu``; see
:mod:`neurogrip.vision.backends.hggd_mcu`.
"""

from __future__ import annotations

from .backend import (
    BackendInfo,
    VisionBackend,
    available_backends,
    create_backend,
    register_backend,
)
from .depth import OBJECT_SIZE_PRIORS, MonocularDepthEstimator, SizePrior
from .pipeline import VisionPipeline, VisionStats
from .tracking import ObjectTracker, Track
from .types import (
    BoundingBox,
    DepthEstimate,
    Detection,
    GraspApproach,
    GraspCandidate,
    VisionCapability,
    VisionResult,
)

__all__ = [
    "OBJECT_SIZE_PRIORS",
    "BackendInfo",
    "BoundingBox",
    "DepthEstimate",
    "Detection",
    "GraspApproach",
    "GraspCandidate",
    "MonocularDepthEstimator",
    "ObjectTracker",
    "SizePrior",
    "Track",
    "VisionBackend",
    "VisionCapability",
    "VisionPipeline",
    "VisionResult",
    "VisionStats",
    "available_backends",
    "create_backend",
    "register_backend",
]
