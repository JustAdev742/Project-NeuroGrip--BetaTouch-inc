"""Recorded-perception replay, and the recorder that produces it.

The simulated camera generates scenes procedurally, which is right for testing
*logic* — it can produce any situation on demand. It is the wrong tool for
testing *perception quality*, because the ground truth is whatever the generator
decided, so a detector regression cannot show up as a difference.

Replay closes that gap. A recording is a fixed sequence of what the vision system
actually reported, captured once from real hardware or from a known-good build,
and replayed identically on every run. That makes two otherwise impossible things
routine:

* **Regression testing.** Change the backend, replay the same recording, and any
  difference in detections or grasps is attributable to the change rather than to
  a different random scene.
* **Reproducing field reports.** "It kept trying to pinch my mug" becomes a file
  someone can replay at a desk, rather than a description to be re-enacted.

The format is JSON Lines: one :class:`~neurogrip.vision.types.VisionResult` per
line, in capture order. Line-oriented so a recording truncated by a power loss
loses only its last line, and text so it can be inspected, diffed, and trimmed by
hand — recordings are evidence, and evidence that needs a special tool to read is
evidence nobody reads.

Replay is driven by *frame count*, not wall-clock time. A recording played back
under a simulated clock running 50× real time must yield the same sequence, and
matching on timestamps would silently skip frames.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ...core.config import Config
from ...core.errors import VisionError
from ...core.logging import get_logger
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

__all__ = ["ReplaySettings", "ReplayVisionBackend", "VisionRecorder", "load_recording"]

log = get_logger(__name__)

#: Bumped when the on-disk shape changes incompatibly. Readers refuse a version
#: they do not understand rather than silently misreading fields.
FORMAT_VERSION = 1


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def _detection_to_dict(detection: Detection) -> dict:
    return {
        "label": detection.label,
        "confidence": detection.confidence,
        "bbox": [detection.bbox.x1, detection.bbox.y1, detection.bbox.x2, detection.bbox.y2],
        "track_id": detection.track_id,
        "age": detection.age,
        "attributes": dict(detection.attributes),
    }


def _detection_from_dict(data: dict) -> Detection:
    x1, y1, x2, y2 = data["bbox"]
    return Detection(
        label=str(data["label"]),
        confidence=float(data["confidence"]),
        bbox=BoundingBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2)),
        track_id=int(data.get("track_id", -1)),
        age=int(data.get("age", 0)),
        attributes=dict(data.get("attributes", {})),
    )


def _grasp_to_dict(grasp: GraspCandidate) -> dict:
    return {
        "center_x": grasp.center_x,
        "center_y": grasp.center_y,
        "angle": grasp.angle,
        "width": grasp.width,
        "quality": grasp.quality,
        "depth_m": grasp.depth_m,
        "width_m": grasp.width_m,
        "approach": grasp.approach.value,
        "source": grasp.source,
        "label": grasp.label,
        "approach_vector": list(grasp.approach_vector) if grasp.approach_vector else None,
    }


def _grasp_from_dict(data: dict) -> GraspCandidate:
    vector = data.get("approach_vector")
    return GraspCandidate(
        center_x=float(data["center_x"]),
        center_y=float(data["center_y"]),
        angle=float(data.get("angle", 0.0)),
        width=float(data.get("width", 0.0)),
        quality=float(data.get("quality", 0.0)),
        depth_m=data.get("depth_m"),
        width_m=data.get("width_m"),
        approach=GraspApproach(data.get("approach", "unknown")),
        source=str(data.get("source", "")),
        label=str(data.get("label", "")),
        approach_vector=tuple(vector) if vector else None,
    )


def _result_to_dict(result: VisionResult) -> dict:
    depth = result.depth
    return {
        "timestamp": result.timestamp,
        "frame_index": result.frame_index,
        "backend": result.backend,
        "latency_ms": result.latency_ms,
        "error": result.error,
        "detections": [_detection_to_dict(d) for d in result.detections],
        "grasps": [_grasp_to_dict(g) for g in result.grasps],
        "depth": None
        if depth is None
        else {
            "distance_m": depth.distance_m,
            "confidence": depth.confidence,
            "method": depth.method,
            "relative_error": depth.relative_error,
        },
    }


def _result_from_dict(data: dict) -> VisionResult:
    depth_data = data.get("depth")
    depth = None
    if depth_data:
        depth = DepthEstimate(
            distance_m=float(depth_data["distance_m"]),
            confidence=float(depth_data.get("confidence", 0.0)),
            method=str(depth_data.get("method", "unknown")),
            relative_error=float(depth_data.get("relative_error", 0.5)),
        )
    detections = tuple(_detection_from_dict(d) for d in data.get("detections", []))
    grasps = tuple(_grasp_from_dict(g) for g in data.get("grasps", []))

    capabilities = VisionCapability.NONE
    if detections:
        capabilities |= VisionCapability.DETECTION | VisionCapability.CLASSIFICATION
    if grasps:
        capabilities |= VisionCapability.GRASP
    if depth is not None:
        capabilities |= VisionCapability.DEPTH

    return VisionResult(
        timestamp=float(data.get("timestamp", 0.0)),
        frame_index=int(data.get("frame_index", 0)),
        detections=detections,
        grasps=grasps,
        depth=depth,
        latency_ms=float(data.get("latency_ms", 0.0)),
        backend=str(data.get("backend", "replay")),
        capabilities=capabilities,
        error=str(data.get("error", "")),
    )


def load_recording(path: Path | str) -> tuple[VisionResult, ...]:
    """Read a recording, skipping lines that do not parse.

    A malformed line is logged and dropped rather than aborting the load: a
    recording is usually the only copy of something that already happened, and
    losing all of it because one line was truncated would be the wrong trade.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise VisionError(f"vision recording not found: {file_path}")

    results: list[VisionResult] = []
    header_checked = False
    for number, line in enumerate(file_path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError as exc:
            log.warning("skipping malformed line in recording", line=number, error=str(exc))
            continue
        if not header_checked and "format_version" in data:
            version = int(data["format_version"])
            if version != FORMAT_VERSION:
                raise VisionError(
                    f"recording {file_path} is format v{version}; this build reads v{FORMAT_VERSION}"
                )
            header_checked = True
            continue
        try:
            results.append(_result_from_dict(data))
        except (KeyError, TypeError, ValueError) as exc:
            log.warning("skipping unreadable frame in recording", line=number, error=str(exc))

    if not results:
        raise VisionError(f"recording {file_path} contains no usable frames")
    return tuple(results)


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------


class VisionRecorder:
    """Appends vision results to a recording file.

    Flushes every frame. A recording is normally being made *because* something
    is going wrong, and the frames worth having are the ones just before it goes
    wrong — exactly the ones a buffer would lose.
    """

    def __init__(self, path: Path | str, *, backend: str = "", note: str = "") -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self._path.open("w", encoding="utf-8")
        self._handle.write(
            json.dumps({"format_version": FORMAT_VERSION, "backend": backend, "note": note}) + "\n"
        )
        self._handle.flush()
        self.frames = 0

    @property
    def path(self) -> Path:
        return self._path

    def write(self, result: VisionResult) -> None:
        self._handle.write(json.dumps(_result_to_dict(result)) + "\n")
        self._handle.flush()
        self.frames += 1

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()
            log.info("vision recording saved", path=str(self._path), frames=self.frames)

    def __enter__(self) -> VisionRecorder:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ReplaySettings:
    """Configuration for the replay backend."""

    path: str = ""
    #: Restart from the beginning at the end of the recording.
    loop: bool = True
    #: Emit each recorded frame this many times, to replay a recording captured
    #: at one rate through a pipeline running at another.
    repeat: int = 1

    @classmethod
    def from_config(cls, config: Config) -> ReplaySettings:
        d = cls()
        section = config.section("vision.replay")
        return cls(
            path=section.get_str("path", config.get_str("vision.recording", d.path)),
            loop=section.get_bool("loop", d.loop),
            repeat=max(1, section.get_int("repeat", d.repeat)),
        )


class ReplayVisionBackend:
    """Replays a recording instead of running a model."""

    def __init__(self, settings: ReplaySettings) -> None:
        if not settings.path:
            raise VisionError(
                "the replay backend needs vision.replay.path (or vision.recording) "
                "set to a recording file"
            )
        self._settings = settings
        self._results = load_recording(settings.path)
        self._index = 0
        self._emitted = 0
        self.frames = 0
        self.wrapped = 0

    @property
    def name(self) -> str:
        return "replay"

    @property
    def capabilities(self) -> VisionCapability:
        """Union of what the recording actually contains.

        Derived from the data rather than declared, so replaying a
        detection-only recording does not advertise grasp planning the recording
        cannot deliver.
        """
        capabilities = VisionCapability.NONE
        for result in self._results:
            capabilities |= result.capabilities
        return capabilities

    def info(self) -> BackendInfo:
        return BackendInfo(
            name="replay",
            version=str(FORMAT_VERSION),
            capabilities=self.capabilities,
            runtime="recording",
            model_path=self._settings.path,
            degraded_reason="replaying a recording — not live perception",
        )

    def initialize(self) -> None:
        self._index = 0
        self._emitted = 0
        log.info(
            "replaying vision recording",
            path=self._settings.path,
            frames=len(self._results),
            loop=self._settings.loop,
        )

    def shutdown(self) -> None:
        self._index = 0

    @property
    def exhausted(self) -> bool:
        """True when a non-looping recording has been played to the end."""
        return not self._settings.loop and self._index >= len(self._results)

    def process(self, frame: Frame) -> VisionResult:
        """Return the next recorded result, re-stamped to the current frame.

        The timestamp is replaced rather than preserved: everything downstream
        judges staleness against the clock, and a result carrying a timestamp
        from a recording made last week is stale by any measure.
        """
        self.frames += 1
        if self.exhausted:
            return VisionResult.empty(frame.timestamp, self.name, "recording exhausted")

        result = self._results[self._index % len(self._results)]
        self._emitted += 1
        if self._emitted >= self._settings.repeat:
            self._emitted = 0
            self._index += 1
            if self._settings.loop and self._index >= len(self._results):
                self._index = 0
                self.wrapped += 1

        return VisionResult(
            timestamp=frame.timestamp,
            frame_index=frame.index,
            detections=result.detections,
            grasps=result.grasps,
            depth=result.depth,
            latency_ms=result.latency_ms,
            backend=self.name,
            capabilities=result.capabilities,
            error=result.error,
        )


def _build(config: Config) -> ReplayVisionBackend:
    return ReplayVisionBackend(ReplaySettings.from_config(config))


register_backend("replay", _build)
