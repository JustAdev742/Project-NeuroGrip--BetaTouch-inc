"""Bundled vision backends.

Importing this package registers every built-in backend with
:mod:`neurogrip.vision.backend`. Third-party backends register themselves the
same way — call ``register_backend(name, factory)`` at import time and add the
module to ``[vision] extra_backends`` in configuration.
"""

from __future__ import annotations

from .anygrasp import AnyGraspBackend, AnyGraspSettings, SixDofGrasp
from .hggd_mcu import HggdMcuBackend, HggdMcuSettings
from .mock import MockSettings, MockVisionBackend
from .null import NullVisionBackend
from .onnx_detector import OnnxDetectorBackend, OnnxDetectorSettings
from .replay import ReplaySettings, ReplayVisionBackend, VisionRecorder

__all__ = [
    "AnyGraspBackend",
    "AnyGraspSettings",
    "HggdMcuBackend",
    "HggdMcuSettings",
    "MockSettings",
    "MockVisionBackend",
    "NullVisionBackend",
    "OnnxDetectorBackend",
    "OnnxDetectorSettings",
    "ReplaySettings",
    "ReplayVisionBackend",
    "SixDofGrasp",
    "VisionRecorder",
]
