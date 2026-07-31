"""Generic ONNX object detector (YOLO-style output).

Provided as the second concrete example of the backend interface, proving the
abstraction is not shaped around HGGD-MCU alone. A plain detector declares only
``DETECTION | CLASSIFICATION`` — no ``GRASP`` — which makes
:mod:`neurogrip.ai.grasp` fall back to the affordance-driven heuristic planner.
That path is exercised by ``tests/unit/test_grasp_planner.py``.

Expected output: ``(1, N, 5 + num_classes)`` with rows
``[cx, cy, w, h, objectness, class scores...]`` in letterboxed input coordinates
— the layout emitted by YOLOv5/v8-style exports.

TODO(models): the class-name list is read from a sibling ``.names`` file when
present; otherwise classes are reported as ``class_<i>``, which the affordance
database maps to ``unknown`` and therefore to a conservative default grasp.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from ...core.config import Config
from ...core.errors import ModelLoadError
from ...core.logging import get_logger
from ...core.types import clamp
from ...hal.camera.base import Frame
from ..backend import BackendInfo, register_backend
from ..preprocess import frame_to_model_input
from ..types import BoundingBox, Detection, VisionCapability, VisionResult

__all__ = ["OnnxDetectorBackend", "OnnxDetectorSettings"]

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class OnnxDetectorSettings:
    """Configuration from ``[vision.onnx_detector]``."""

    model_path: str = "models/detector/detector.onnx"
    input_width: int = 320
    input_height: int = 320
    score_threshold: float = 0.35
    iou_threshold: float = 0.45
    max_detections: int = 16
    providers: tuple[str, ...] = ("CPUExecutionProvider",)

    @classmethod
    def from_config(cls, config: Config) -> OnnxDetectorSettings:
        d = cls()
        section = config.section("vision.onnx_detector")
        return cls(
            model_path=section.get_str("model_path", d.model_path),
            input_width=section.get_int("input_width", d.input_width),
            input_height=section.get_int("input_height", d.input_height),
            score_threshold=section.get_float("score_threshold", d.score_threshold),
            iou_threshold=section.get_float("iou_threshold", d.iou_threshold),
            max_detections=section.get_int("max_detections", d.max_detections),
            providers=tuple(section.get_list("providers", list(d.providers))),
        )


class OnnxDetectorBackend:
    """Detection-only backend."""

    def __init__(
        self, config: Config | None = None, settings: OnnxDetectorSettings | None = None
    ) -> None:
        self._settings = settings or (
            OnnxDetectorSettings.from_config(config) if config is not None else OnnxDetectorSettings()
        )
        self._session = None
        self._input_name = ""
        self._classes: list[str] = []
        self._degraded = ""

    def initialize(self) -> None:
        path = Path(self._settings.model_path)
        try:
            if not path.exists():
                raise ModelLoadError(f"detector weights not found: {path}")
            import onnxruntime as ort  # type: ignore[import-not-found]

            self._session = ort.InferenceSession(
                str(path), providers=list(self._settings.providers)
            )
            self._input_name = self._session.get_inputs()[0].name
            self._classes = _load_class_names(path)
            self._degraded = ""
            log.info("ONNX detector loaded", path=str(path), classes=len(self._classes))
        except (ModelLoadError, ImportError) as exc:
            # Detection-only backends have no meaningful fallback; report the
            # degradation and return empty results so the system still runs.
            self._degraded = str(exc)
            log.warning("ONNX detector unavailable", error=str(exc))

    def shutdown(self) -> None:
        self._session = None

    @property
    def capabilities(self) -> VisionCapability:
        if self._session is None:
            return VisionCapability.NONE
        return VisionCapability.DETECTION | VisionCapability.CLASSIFICATION

    def info(self) -> BackendInfo:
        return BackendInfo(
            name="onnx_detector",
            version="1.0",
            capabilities=self.capabilities,
            runtime="onnxruntime" if self._session else "unavailable",
            model_path=self._settings.model_path,
            input_width=self._settings.input_width,
            input_height=self._settings.input_height,
            degraded_reason=self._degraded,
        )

    def process(self, frame: Frame) -> VisionResult:
        if self._session is None:
            return VisionResult(
                timestamp=frame.timestamp,
                frame_index=frame.index,
                backend="onnx_detector",
                capabilities=VisionCapability.NONE,
            )

        started = time.perf_counter()
        try:
            import numpy as np

            pixels, letterbox_info = frame_to_model_input(
                frame, self._settings.input_width, self._settings.input_height
            )
            array = (
                np.asarray(pixels, dtype=np.float32).reshape(
                    1, 1, self._settings.input_height, self._settings.input_width
                )
                / 255.0
            )
            # Replicate the single channel to RGB; most exported detectors expect 3.
            array = np.repeat(array, 3, axis=1)
            raw = self._session.run(None, {self._input_name: array})[0]
            detections = self._decode(np.asarray(raw), letterbox_info)
        except Exception as exc:
            log.throttled(
                "onnx-detector", "error", "detector inference failed", now=frame.timestamp, error=str(exc)
            )
            return VisionResult.empty(frame.timestamp, "onnx_detector", str(exc))

        return VisionResult(
            timestamp=frame.timestamp,
            frame_index=frame.index,
            detections=tuple(detections),
            latency_ms=(time.perf_counter() - started) * 1000.0,
            backend="onnx_detector",
            capabilities=self.capabilities,
        )

    def _decode(self, raw, letterbox_info) -> list[Detection]:
        """Decode YOLO-style rows and apply class-wise NMS."""
        rows = raw.reshape(-1, raw.shape[-1])
        candidates: list[tuple[float, str, BoundingBox]] = []

        for row in rows:
            objectness = float(row[4])
            if objectness < self._settings.score_threshold:
                continue
            scores = row[5:]
            if len(scores) == 0:
                continue
            best = int(scores.argmax())
            confidence = objectness * float(scores[best])
            if confidence < self._settings.score_threshold:
                continue

            cx, cy, w, h = (float(v) for v in row[:4])
            # Model space is pixels in the padded image; normalise, then un-pad.
            nx = cx / self._settings.input_width
            ny = cy / self._settings.input_height
            nw = w / self._settings.input_width
            nh = h / self._settings.input_height
            sx, sy = letterbox_info.to_source(nx, ny)
            sw = letterbox_info.scale_length(nw)
            sh = letterbox_info.scale_length(nh)

            label = self._classes[best] if best < len(self._classes) else f"class_{best}"
            candidates.append(
                (
                    confidence,
                    label,
                    BoundingBox(
                        clamp(sx - sw / 2), clamp(sy - sh / 2), clamp(sx + sw / 2), clamp(sy + sh / 2)
                    ),
                )
            )

        candidates.sort(key=lambda c: c[0], reverse=True)
        kept: list[Detection] = []
        for confidence, label, bbox in candidates:
            if any(
                bbox.iou(existing.bbox) > self._settings.iou_threshold and existing.label == label
                for existing in kept
            ):
                continue
            kept.append(Detection(label=label, confidence=confidence, bbox=bbox))
            if len(kept) >= self._settings.max_detections:
                break
        return kept


def _load_class_names(model_path: Path) -> list[str]:
    """Read a sibling ``<model>.names`` file, one class per line."""
    names_path = model_path.with_suffix(".names")
    if not names_path.exists():
        return []
    return [line.strip() for line in names_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _factory(config: Config | None = None, **_: object) -> OnnxDetectorBackend:
    return OnnxDetectorBackend(config)


register_backend("onnx_detector", _factory)
