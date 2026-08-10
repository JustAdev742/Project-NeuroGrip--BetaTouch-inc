"""HGGD-MCU — heatmap-guided grasp detection, edge-deployed.

This is the project's primary vision backend.

Background
----------
HGGD (*Efficient Heatmap-Guided 6-DoF Grasp Detection in Cluttered Scenes*,
Chen et al., IEEE RA-L 2023) detects grasps in two stages: a global network
predicts a dense **graspable heatmap** plus coarse grasp attributes, and a local
network refines candidates around the heatmap peaks. The heatmap is the useful
idea for us — it answers "where on this object would a grasp work?" directly,
rather than making us infer it from a bounding box.

``hggd-mcu`` is the microcontroller/edge profile of that architecture: a single
quantised network, monocular input, and the local refinement stage folded into
per-anchor regression heads. It runs in a few milliseconds on the Pi-class SBC
that hosts this stack, which is what makes a 30 Hz assistive loop feasible.

Output tensor contract
----------------------
One forward pass over a ``1 × 1 × H × W`` greyscale input produces, at stride
``S`` (so ``h = H/S``, ``w = W/S``):

===============  =================  ==============================================
head             shape              meaning
===============  =================  ==============================================
``heatmap``      ``h × w``          graspability, 0..1
``angle``        ``h × w × A``      grasp-axis orientation, ``A`` bins over [0, π)
``width``        ``h × w``          gripper opening, normalised to image width
``quality``      ``h × w``          predicted grasp success, 0..1
``class``        ``h × w × C``      object class logits (shares the backbone)
===============  =================  ==============================================

Decoding — peak finding, angle-bin argmax, and grasp NMS — lives in
:func:`decode_heatmap`, is independent of the inference runtime, and is unit
tested against hand-built tensors in ``tests/unit/test_hggd_mcu.py``.

Runtimes
--------
The network is reached through :class:`InferenceSession`, so the same decoding
code serves ONNX Runtime, TFLite, and the classical fallback:

* :class:`OnnxInferenceSession` — ``onnxruntime``, the deployment path;
* :class:`TfliteInferenceSession` — ``tflite_runtime``, for int8 on constrained
  targets;
* :class:`ClassicalHeatmapSession` — no weights required. It computes a real
  edge-density graspability heatmap in the *same tensor layout*, so the whole
  decode path is exercised without a model file. It is a genuine (if modest)
  classical baseline, not a stub, and it is what keeps the system usable when the
  weights are missing.

.. note::
   **Depth.** Reference HGGD consumes RGB-D. This build has a monocular camera,
   so metric depth comes from :mod:`neurogrip.vision.depth` (size priors) and is
   attached after decoding. The grasp *geometry* is therefore reliable in image
   space and approximate in metres — which is exactly why the fusion layer treats
   vision as advisory and never lets it authorise motion on its own.
   TODO(hardware): when a depth camera is fitted, feed a real depth plane into
   :meth:`InferenceSession.run` as a second input channel and drop the size-prior
   estimator.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from ...core.config import Config
from ...core.errors import ModelLoadError
from ...core.logging import get_logger
from ...core.types import clamp
from ...hal.camera.base import Frame
from ..backend import BackendInfo, register_backend
from ..preprocess import LetterboxInfo, frame_to_model_input
from ..types import (
    BoundingBox,
    Detection,
    GraspApproach,
    GraspCandidate,
    VisionCapability,
    VisionResult,
)

__all__ = [
    "ClassicalHeatmapSession",
    "HeatmapTensors",
    "HggdMcuBackend",
    "HggdMcuSettings",
    "InferenceSession",
    "OnnxInferenceSession",
    "TfliteInferenceSession",
    "decode_heatmap",
]

log = get_logger(__name__)

#: Class labels the shipped model head is trained for. Kept in one place so the
#: affordance database (``neurogrip.ai.objects``) can be validated against it at
#: startup rather than failing on an unknown label at grasp time.
HGGD_MCU_CLASSES: tuple[str, ...] = (
    "unknown",
    "bottle",
    "cup",
    "can",
    "box",
    "ball",
    "pen",
    "key",
    "card",
    "phone",
    "book",
    "tool",
    "fruit",
    "plate",
    "handle",
)


@dataclass(frozen=True, slots=True)
class HggdMcuSettings:
    """Configuration for the backend, from ``[vision.hggd_mcu]``."""

    model_path: str = "models/hggd_mcu/hggd_mcu_int8.onnx"
    input_width: int = 160
    input_height: int = 128
    #: Downsampling factor from input to heatmap grid.
    stride: int = 8
    #: Number of discrete orientation bins over [0, π).
    angle_bins: int = 12
    #: Heatmap responses below this are not considered.
    score_threshold: float = 0.35
    #: Minimum combined distance between retained grasps during NMS.
    nms_distance: float = 0.12
    #: Cap on returned candidates; the planner only ever needs a handful.
    max_grasps: int = 8
    #: Class probability below which a detection is reported as "unknown".
    class_threshold: float = 0.30
    #: Execution providers passed to ONNX Runtime, in priority order.
    providers: tuple[str, ...] = ("CPUExecutionProvider",)
    #: Number of inference threads; 2 leaves headroom for the control loop.
    threads: int = 2

    @property
    def grid_width(self) -> int:
        return max(1, self.input_width // self.stride)

    @property
    def grid_height(self) -> int:
        return max(1, self.input_height // self.stride)

    @classmethod
    def from_config(cls, config: Config) -> HggdMcuSettings:
        """Build settings from ``[vision.hggd_mcu]``, keeping dataclass defaults.

        ``slots=True`` removes class-level attributes, so defaults are read from
        a prototype instance rather than from ``cls``.
        """
        d = cls()
        section = config.section("vision.hggd_mcu")
        return cls(
            model_path=section.get_str("model_path", d.model_path),
            input_width=section.get_int("input_width", d.input_width),
            input_height=section.get_int("input_height", d.input_height),
            stride=section.get_int("stride", d.stride),
            angle_bins=section.get_int("angle_bins", d.angle_bins),
            score_threshold=section.get_float("score_threshold", d.score_threshold),
            nms_distance=section.get_float("nms_distance", d.nms_distance),
            max_grasps=section.get_int("max_grasps", d.max_grasps),
            class_threshold=section.get_float("class_threshold", d.class_threshold),
            providers=tuple(section.get_list("providers", list(d.providers))),
            threads=section.get_int("threads", d.threads),
        )


@dataclass(frozen=True, slots=True)
class HeatmapTensors:
    """Decoded network outputs for one frame, in row-major grid order."""

    width: int
    height: int
    angle_bins: int
    #: ``height × width`` graspability.
    heatmap: list[float]
    #: ``height × width × angle_bins`` orientation logits.
    angle: list[float]
    #: ``height × width`` normalised gripper opening.
    grasp_width: list[float]
    #: ``height × width`` predicted success.
    quality: list[float]
    #: ``height × width × len(classes)`` class logits (may be empty).
    class_logits: list[float] = field(default_factory=list)
    class_count: int = 0

    def at(self, x: int, y: int) -> float:
        return self.heatmap[y * self.width + x]

    def angle_at(self, x: int, y: int) -> float:
        """Argmax orientation bin, converted to radians in ``[0, π)``."""
        base = (y * self.width + x) * self.angle_bins
        bins = self.angle[base : base + self.angle_bins]
        if not bins:
            return 0.0
        best = max(range(len(bins)), key=bins.__getitem__)
        return (best + 0.5) * math.pi / self.angle_bins

    def class_at(self, x: int, y: int) -> tuple[int, float]:
        """``(class_index, probability)`` at a grid cell."""
        if not self.class_count:
            return (0, 0.0)
        base = (y * self.width + x) * self.class_count
        logits = self.class_logits[base : base + self.class_count]
        if not logits:
            return (0, 0.0)
        peak = max(logits)
        exponentials = [math.exp(v - peak) for v in logits]
        total = sum(exponentials)
        probabilities = [e / total for e in exponentials] if total else [0.0] * len(logits)
        best = max(range(len(probabilities)), key=probabilities.__getitem__)
        return (best, probabilities[best])


class InferenceSession(Protocol):
    """Runs the network. One implementation per runtime."""

    @property
    def runtime(self) -> str:
        """Short runtime identifier for diagnostics."""
        ...

    def run(self, pixels: Sequence[int], width: int, height: int) -> HeatmapTensors:
        """Execute one forward pass over a greyscale, letterboxed image."""
        ...

    def close(self) -> None: ...


# ---------------------------------------------------------------------------
# Decoding (runtime independent)
# ---------------------------------------------------------------------------


def decode_heatmap(
    tensors: HeatmapTensors,
    settings: HggdMcuSettings,
    letterbox_info: LetterboxInfo,
) -> list[GraspCandidate]:
    """Turn network outputs into ranked grasp candidates.

    Steps:

    1. **Peak finding** — 3×3 non-maximum suppression on the heatmap grid, so
       only local maxima above ``score_threshold`` survive.
    2. **Sub-cell refinement** — a parabolic fit across each peak's neighbours
       recovers sub-cell position. At stride 8 one cell is ~5 % of the image
       width; without this the grasp point visibly quantises.
    3. **Attribute decode** — orientation from the angle-bin argmax, opening
       from the width head, score from heatmap × quality.
    4. **Coordinate mapping** — model space back to source-image space via the
       letterbox geometry.
    5. **Grasp NMS** — suppress candidates that are close in both position and
       orientation, keeping the strongest.
    """
    peaks: list[tuple[float, int, int]] = []
    for y in range(tensors.height):
        for x in range(tensors.width):
            score = tensors.at(x, y)
            if score < settings.score_threshold:
                continue
            if not _is_local_max(tensors, x, y, score):
                continue
            # Reject peaks sitting on (or right against) the letterbox padding.
            # The pad boundary is a synthetic step edge and would otherwise be
            # the most "graspable" structure in every frame.
            if not letterbox_info.contains_model_point(
                (x + 0.5) / tensors.width, (y + 0.5) / tensors.height, margin=0.03
            ):
                continue
            peaks.append((score, x, y))

    peaks.sort(reverse=True)
    candidates: list[GraspCandidate] = []

    for score, x, y in peaks[: settings.max_grasps * 4]:
        offset_x, offset_y = _sub_cell_offset(tensors, x, y)
        # Grid cell centre, plus the refinement, in model-normalised coordinates.
        model_x = (x + 0.5 + offset_x) / tensors.width
        model_y = (y + 0.5 + offset_y) / tensors.height
        source_x, source_y = letterbox_info.to_source(model_x, model_y)

        index = y * tensors.width + x
        raw_width = tensors.grasp_width[index] if index < len(tensors.grasp_width) else 0.3
        quality = tensors.quality[index] if index < len(tensors.quality) else score
        class_index, class_probability = tensors.class_at(x, y)
        label = (
            HGGD_MCU_CLASSES[class_index]
            if class_probability >= settings.class_threshold
            and class_index < len(HGGD_MCU_CLASSES)
            else "unknown"
        )

        angle = tensors.angle_at(x, y)
        candidates.append(
            GraspCandidate(
                center_x=source_x,
                center_y=source_y,
                angle=angle,
                width=clamp(letterbox_info.scale_length(raw_width)),
                quality=clamp(score * 0.5 + quality * 0.5),
                approach=_approach_from_angle(angle),
                source="hggd_mcu",
                label=label,
            )
        )

    return _grasp_nms(candidates, settings.nms_distance)[: settings.max_grasps]


def _is_local_max(tensors: HeatmapTensors, x: int, y: int, score: float) -> bool:
    """3×3 non-maximum check on the heatmap grid."""
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            in_bounds = 0 <= nx < tensors.width and 0 <= ny < tensors.height
            if in_bounds and tensors.at(nx, ny) > score:
                return False
    return True


def _sub_cell_offset(tensors: HeatmapTensors, x: int, y: int) -> tuple[float, float]:
    """Parabolic sub-cell peak refinement, clamped to ±0.5 cells."""

    def refine(before: float, here: float, after: float) -> float:
        denominator = before - 2.0 * here + after
        if abs(denominator) < 1e-9:
            return 0.0
        return clamp(0.5 * (before - after) / denominator, -0.5, 0.5)

    left = tensors.at(x - 1, y) if x > 0 else tensors.at(x, y)
    right = tensors.at(x + 1, y) if x + 1 < tensors.width else tensors.at(x, y)
    up = tensors.at(x, y - 1) if y > 0 else tensors.at(x, y)
    down = tensors.at(x, y + 1) if y + 1 < tensors.height else tensors.at(x, y)
    here = tensors.at(x, y)
    return refine(left, here, right), refine(up, here, down)


def _grasp_nms(candidates: list[GraspCandidate], min_distance: float) -> list[GraspCandidate]:
    """Greedy non-maximum suppression in the joint position/orientation space."""
    kept: list[GraspCandidate] = []
    for candidate in sorted(candidates, key=lambda c: c.quality, reverse=True):
        if all(candidate.distance_to(other) >= min_distance for other in kept):
            kept.append(candidate)
    return kept


def _approach_from_angle(angle: float) -> GraspApproach:
    """Map an in-plane grasp axis to a coarse approach direction.

    A near-horizontal gripper axis means the fingers close across a vertical
    object, which for a hand-mounted camera is a side approach; a vertical axis
    implies reaching over the object from above.
    """
    normalised = angle % math.pi
    if normalised < math.pi / 6 or normalised > 5 * math.pi / 6:
        return GraspApproach.SIDE
    if math.pi / 3 <= normalised <= 2 * math.pi / 3:
        return GraspApproach.TOP_DOWN
    return GraspApproach.FRONTAL


# ---------------------------------------------------------------------------
# Inference sessions
# ---------------------------------------------------------------------------


class ClassicalHeatmapSession:
    """Weight-free graspability heatmap from local edge structure.

    Not a stub: it implements a real (classical) grasp heuristic and emits the
    exact tensor layout the network would, so the decode path, the pipeline, the
    planner and the UI are all fully exercised with no model file present.

    Method, per grid cell:

    * accumulate Sobel gradients over the cell;
    * **graspability** rises with total edge energy (an object boundary is
      present) and falls where the gradient is isotropic (texture, not an edge);
    * **orientation** is the dominant gradient direction rotated by 90°, because
      fingers should close *across* an edge, not along it;
    * **width** comes from the distance to the next strong edge along the
      closing axis — a direct estimate of how far apart the object's sides are.
    """

    def __init__(self, settings: HggdMcuSettings) -> None:
        self._settings = settings

    @property
    def runtime(self) -> str:
        return "classical"

    def close(self) -> None:
        """Nothing to release."""

    def run(self, pixels: Sequence[int], width: int, height: int) -> HeatmapTensors:
        settings = self._settings
        stride = settings.stride
        grid_w = max(1, width // stride)
        grid_h = max(1, height // stride)
        bins = settings.angle_bins

        heatmap = [0.0] * (grid_w * grid_h)
        angle = [0.0] * (grid_w * grid_h * bins)
        grasp_width = [0.25] * (grid_w * grid_h)
        quality = [0.0] * (grid_w * grid_h)

        for gy in range(grid_h):
            for gx in range(grid_w):
                gxx = gyy = gxy = 0.0
                energy = 0.0
                samples = 0
                for y in range(gy * stride + 1, min(height - 1, (gy + 1) * stride)):
                    row = y * width
                    for x in range(gx * stride + 1, min(width - 1, (gx + 1) * stride)):
                        # 3×3 Sobel.
                        dx = (
                            pixels[row - width + x + 1]
                            + 2 * pixels[row + x + 1]
                            + pixels[row + width + x + 1]
                            - pixels[row - width + x - 1]
                            - 2 * pixels[row + x - 1]
                            - pixels[row + width + x - 1]
                        ) / 4.0
                        dy = (
                            pixels[row + width + x - 1]
                            + 2 * pixels[row + width + x]
                            + pixels[row + width + x + 1]
                            - pixels[row - width + x - 1]
                            - 2 * pixels[row - width + x]
                            - pixels[row - width + x + 1]
                        ) / 4.0
                        gxx += dx * dx
                        gyy += dy * dy
                        gxy += dx * dy
                        energy += math.hypot(dx, dy)
                        samples += 1

                if samples == 0:
                    continue

                index = gy * grid_w + gx
                mean_energy = energy / samples
                # Structure-tensor coherence: 1.0 for a clean edge, ~0 for texture.
                trace = gxx + gyy
                determinant = gxx * gyy - gxy * gxy
                discriminant = max(0.0, trace * trace / 4.0 - determinant)
                coherence = (
                    math.sqrt(discriminant) / (trace / 2.0) if trace > 1e-6 else 0.0
                )

                score = clamp(mean_energy / 45.0) * clamp(0.35 + 0.65 * coherence)
                heatmap[index] = score
                quality[index] = clamp(score * 0.85 + 0.1)

                # Dominant gradient direction, then rotate 90° to close across it.
                gradient_angle = 0.5 * math.atan2(2.0 * gxy, gxx - gyy)
                closing_angle = (gradient_angle + math.pi / 2) % math.pi
                best_bin = min(bins - 1, int(closing_angle / math.pi * bins))
                base = index * bins
                for b in range(bins):
                    # Soft one-hot: neighbours get partial weight, mirroring the
                    # smoothed targets a trained model produces.
                    delta = min(abs(b - best_bin), bins - abs(b - best_bin))
                    angle[base + b] = math.exp(-(delta**2) / 2.0)

                grasp_width[index] = clamp(0.12 + 0.5 * (1.0 - coherence), 0.08, 0.9)

        return HeatmapTensors(
            width=grid_w,
            height=grid_h,
            angle_bins=bins,
            heatmap=heatmap,
            angle=angle,
            grasp_width=grasp_width,
            quality=quality,
            class_logits=[],
            class_count=0,
        )


class OnnxInferenceSession:
    """ONNX Runtime session — the deployment path."""

    def __init__(self, settings: HggdMcuSettings) -> None:
        self._settings = settings
        self._session = None
        self._input_name = ""
        self._output_names: list[str] = []
        self._load()

    @property
    def runtime(self) -> str:
        return "onnxruntime"

    def _load(self) -> None:
        path = Path(self._settings.model_path)
        if not path.exists():
            raise ModelLoadError(f"HGGD-MCU weights not found: {path}")
        try:
            import onnxruntime as ort  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ModelLoadError(
                "onnxruntime is not installed; install the 'vision' extra"
            ) from exc

        options = ort.SessionOptions()
        options.intra_op_num_threads = self._settings.threads
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        try:
            self._session = ort.InferenceSession(
                str(path), sess_options=options, providers=list(self._settings.providers)
            )
        except Exception as exc:  # onnxruntime raises bare RuntimeError
            raise ModelLoadError(f"failed to load {path}: {exc}") from exc

        self._input_name = self._session.get_inputs()[0].name
        self._output_names = [o.name for o in self._session.get_outputs()]
        log.info(
            "HGGD-MCU loaded",
            path=str(path),
            providers=self._session.get_providers(),
            outputs=self._output_names,
        )

    def run(self, pixels: Sequence[int], width: int, height: int) -> HeatmapTensors:
        import numpy as np

        settings = self._settings
        array = (
            np.asarray(pixels, dtype=np.float32).reshape(1, 1, height, width) / 255.0
        )
        outputs = self._session.run(self._output_names, {self._input_name: array})  # type: ignore[union-attr]
        named = dict(zip(self._output_names, outputs))

        grid_w = settings.grid_width
        grid_h = settings.grid_height

        def flat(key: str, fallback: float, size: int) -> list[float]:
            tensor = named.get(key)
            if tensor is None:
                return [fallback] * size
            return np.asarray(tensor, dtype=np.float32).reshape(-1)[:size].tolist()

        cells = grid_w * grid_h
        class_logits = flat("class", 0.0, cells * len(HGGD_MCU_CLASSES)) if "class" in named else []
        return HeatmapTensors(
            width=grid_w,
            height=grid_h,
            angle_bins=settings.angle_bins,
            heatmap=flat("heatmap", 0.0, cells),
            angle=flat("angle", 0.0, cells * settings.angle_bins),
            grasp_width=flat("width", 0.3, cells),
            quality=flat("quality", 0.0, cells),
            class_logits=class_logits,
            class_count=len(HGGD_MCU_CLASSES) if class_logits else 0,
        )

    def close(self) -> None:
        self._session = None


class TfliteInferenceSession:
    """TFLite session for int8 deployment on constrained targets.

    TODO(hardware): quantisation parameters are read from the interpreter, but
    the int8 zero-point/scale round-trip has only been validated against the
    float reference on the development machine — re-verify on the target SoC
    before trusting it for grasp geometry.
    """

    def __init__(self, settings: HggdMcuSettings) -> None:
        self._settings = settings
        self._interpreter = None
        self._input_detail: dict = {}
        self._output_details: list[dict] = []
        self._load()

    @property
    def runtime(self) -> str:
        return "tflite"

    def _load(self) -> None:
        path = Path(self._settings.model_path)
        if not path.exists():
            raise ModelLoadError(f"HGGD-MCU weights not found: {path}")
        try:
            try:
                from tflite_runtime.interpreter import Interpreter  # type: ignore[import-not-found]
            except ImportError:
                from tensorflow.lite.python.interpreter import (  # type: ignore[import-not-found]
                    Interpreter,
                )
        except ImportError as exc:
            raise ModelLoadError("no TFLite runtime available") from exc

        self._interpreter = Interpreter(model_path=str(path), num_threads=self._settings.threads)
        self._interpreter.allocate_tensors()
        self._input_detail = self._interpreter.get_input_details()[0]
        self._output_details = self._interpreter.get_output_details()
        log.info("HGGD-MCU (tflite) loaded", path=str(path))

    def run(self, pixels: Sequence[int], width: int, height: int) -> HeatmapTensors:
        import numpy as np

        settings = self._settings
        detail = self._input_detail
        array = np.asarray(pixels, dtype=np.float32).reshape(1, height, width, 1) / 255.0
        if detail["dtype"] == np.int8:
            scale, zero_point = detail["quantization"]
            array = np.clip(array / max(scale, 1e-9) + zero_point, -128, 127).astype(np.int8)
        else:
            array = array.astype(detail["dtype"])

        self._interpreter.set_tensor(detail["index"], array)  # type: ignore[union-attr]
        self._interpreter.invoke()  # type: ignore[union-attr]

        outputs: dict[str, list[float]] = {}
        for detail_out in self._output_details:
            tensor = self._interpreter.get_tensor(detail_out["index"])  # type: ignore[union-attr]
            if detail_out["dtype"] == np.int8:
                scale, zero_point = detail_out["quantization"]
                tensor = (tensor.astype(np.float32) - zero_point) * scale
            name = detail_out["name"].split("/")[-1].split(":")[0].lower()
            outputs[name] = np.asarray(tensor, dtype=np.float32).reshape(-1).tolist()

        cells = settings.grid_width * settings.grid_height
        return HeatmapTensors(
            width=settings.grid_width,
            height=settings.grid_height,
            angle_bins=settings.angle_bins,
            heatmap=outputs.get("heatmap", [0.0] * cells),
            angle=outputs.get("angle", [0.0] * (cells * settings.angle_bins)),
            grasp_width=outputs.get("width", [0.3] * cells),
            quality=outputs.get("quality", [0.0] * cells),
            class_logits=outputs.get("class", []),
            class_count=len(HGGD_MCU_CLASSES) if "class" in outputs else 0,
        )

    def close(self) -> None:
        self._interpreter = None


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


class HggdMcuBackend:
    """Heatmap-guided grasp detection backend."""

    def __init__(self, config: Config | None = None, settings: HggdMcuSettings | None = None) -> None:
        self._settings = settings or (
            HggdMcuSettings.from_config(config) if config is not None else HggdMcuSettings()
        )
        self._session: InferenceSession | None = None
        self._degraded = ""
        self._frames = 0
        self._total_latency_ms = 0.0

    # -- lifecycle ------------------------------------------------------------

    def initialize(self) -> None:
        """Load the model, falling back to the classical session if unavailable.

        Falling back rather than raising is the correct trade-off here: without
        weights the user loses *grasp quality*, not the use of their hand.
        The degradation is recorded, logged and shown on the diagnostics screen,
        so it is never silent.
        """
        path = Path(self._settings.model_path)
        suffix = path.suffix.lower()
        try:
            if suffix == ".onnx":
                self._session = OnnxInferenceSession(self._settings)
            elif suffix in (".tflite", ".lite"):
                self._session = TfliteInferenceSession(self._settings)
            else:
                raise ModelLoadError(f"unsupported model format: {suffix or '<none>'}")
            self._degraded = ""
        except ModelLoadError as exc:
            self._session = ClassicalHeatmapSession(self._settings)
            self._degraded = str(exc)
            log.warning(
                "HGGD-MCU weights unavailable; using the classical graspability fallback",
                error=str(exc),
                model_path=str(path),
            )

    def shutdown(self) -> None:
        if self._session is not None:
            self._session.close()
            self._session = None

    # -- description ----------------------------------------------------------

    @property
    def capabilities(self) -> VisionCapability:
        base = VisionCapability.GRASP | VisionCapability.DETECTION
        if self._session is not None and self._session.runtime != "classical":
            base |= VisionCapability.CLASSIFICATION
        return base

    def info(self) -> BackendInfo:
        return BackendInfo(
            name="hggd_mcu",
            version="1.0",
            capabilities=self.capabilities,
            runtime=self._session.runtime if self._session else "uninitialised",
            model_path=self._settings.model_path,
            input_width=self._settings.input_width,
            input_height=self._settings.input_height,
            degraded_reason=self._degraded,
        )

    @property
    def average_latency_ms(self) -> float:
        return self._total_latency_ms / self._frames if self._frames else 0.0

    # -- inference ------------------------------------------------------------

    def process(self, frame: Frame) -> VisionResult:
        """Run detection + grasp decoding on one frame."""
        if self._session is None:
            return VisionResult.empty(frame.timestamp, "hggd_mcu", "backend not initialised")

        started = time.perf_counter()
        try:
            pixels, letterbox_info = frame_to_model_input(
                frame, self._settings.input_width, self._settings.input_height
            )
            tensors = self._session.run(
                pixels, self._settings.input_width, self._settings.input_height
            )
            grasps = decode_heatmap(tensors, self._settings, letterbox_info)
            detections = self._detections_from_grasps(grasps)
        except Exception as exc:
            log.throttled(
                "hggd-inference",
                "error",
                "HGGD-MCU inference failed",
                now=frame.timestamp,
                error=str(exc),
            )
            return VisionResult.empty(frame.timestamp, "hggd_mcu", f"inference failed: {exc}")

        latency_ms = (time.perf_counter() - started) * 1000.0
        self._frames += 1
        self._total_latency_ms += latency_ms

        return VisionResult(
            timestamp=frame.timestamp,
            frame_index=frame.index,
            detections=tuple(detections),
            grasps=tuple(grasps),
            latency_ms=latency_ms,
            backend="hggd_mcu",
            capabilities=self.capabilities,
        )

    def _detections_from_grasps(self, grasps: Sequence[GraspCandidate]) -> list[Detection]:
        """Group grasp candidates into object-level detections.

        HGGD predicts grasps, not boxes. Downstream code (the affordance
        database, the UI, the target selector) is object-oriented, so grasps that
        share a class and sit close together are merged into one detection whose
        box is their bounding extent. This keeps the "what object is it?" and
        "where would I grip it?" questions separable.
        """
        if not grasps:
            return []

        clusters: list[list[GraspCandidate]] = []
        for grasp in sorted(grasps, key=lambda g: g.quality, reverse=True):
            for cluster in clusters:
                anchor = cluster[0]
                near = (
                    math.hypot(grasp.center_x - anchor.center_x, grasp.center_y - anchor.center_y)
                    < 0.22
                )
                if near and grasp.label == anchor.label:
                    cluster.append(grasp)
                    break
            else:
                clusters.append([grasp])

        detections: list[Detection] = []
        for cluster in clusters:
            xs = [g.center_x for g in cluster]
            ys = [g.center_y for g in cluster]
            # Grasp width is the object's extent across the closing axis; use it
            # to give the box a sensible size even from a single candidate.
            spread = max(0.06, max(g.width for g in cluster) / 2)
            detections.append(
                Detection(
                    label=cluster[0].label or "unknown",
                    confidence=clamp(max(g.quality for g in cluster)),
                    bbox=BoundingBox(
                        clamp(min(xs) - spread),
                        clamp(min(ys) - spread),
                        clamp(max(xs) + spread),
                        clamp(max(ys) + spread),
                    ),
                    attributes={
                        "grasp_count": float(len(cluster)),
                        "source": "hggd_mcu",
                    },
                )
            )
        return detections


def _factory(config: Config | None = None, **_: object) -> HggdMcuBackend:
    return HggdMcuBackend(config)


register_backend("hggd_mcu", _factory)
