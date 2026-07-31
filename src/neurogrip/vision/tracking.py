"""Multi-object tracking and temporal label smoothing.

Per-frame detections flicker: an object is seen at 0.9 confidence, missed
entirely on the next frame, then reported as a different class on the one after.
Choosing a grasp from that would produce a hand that changes its mind mid-reach.

The tracker provides:

* **identity** across frames via greedy IoU association;
* **persistence** — a track survives a configurable number of missed frames, so a
  brief occlusion (by the user's own hand, typically) does not reset it;
* **label voting** — the reported class is the majority over the track's recent
  history, not the current frame's guess;
* **age**, which the fusion layer uses as a stability term: a two-frame-old track
  is not trusted the way a twenty-frame-old one is.

Greedy IoU (not Hungarian assignment, not a Kalman filter) is the deliberate
choice: with a handful of objects and a hand-mounted camera, association is
rarely ambiguous, and the simpler algorithm is easier to reason about when
something does go wrong.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field

from .types import BoundingBox, Detection

__all__ = ["ObjectTracker", "Track"]


@dataclass(slots=True)
class Track:
    """A tracked object's state across frames."""

    track_id: int
    bbox: BoundingBox
    label: str
    confidence: float
    #: Frames since this track was created.
    age: int = 0
    #: Consecutive frames without a matching detection.
    missed: int = 0
    #: Recent labels, for majority voting.
    label_history: deque[str] = field(default_factory=lambda: deque(maxlen=15))
    confidence_history: deque[float] = field(default_factory=lambda: deque(maxlen=15))
    last_seen: float = 0.0
    #: Normalised per-frame motion, for "is the object still?" checks.
    velocity: tuple[float, float] = (0.0, 0.0)

    @property
    def voted_label(self) -> str:
        """Majority label over the recent history."""
        if not self.label_history:
            return self.label
        return Counter(self.label_history).most_common(1)[0][0]

    @property
    def label_agreement(self) -> float:
        """Fraction of recent frames agreeing with the voted label.

        Low agreement means the classifier is oscillating between classes, which
        the fusion layer treats as low vision confidence regardless of the
        per-frame score.
        """
        if not self.label_history:
            return 0.0
        counts = Counter(self.label_history)
        return counts.most_common(1)[0][1] / len(self.label_history)

    @property
    def smoothed_confidence(self) -> float:
        if not self.confidence_history:
            return self.confidence
        return sum(self.confidence_history) / len(self.confidence_history)

    @property
    def is_moving(self) -> bool:
        """True when the object is translating quickly in the image."""
        return (self.velocity[0] ** 2 + self.velocity[1] ** 2) ** 0.5 > 0.03

    def to_detection(self) -> Detection:
        """Export as a stabilised detection."""
        return Detection(
            label=self.voted_label,
            confidence=self.smoothed_confidence * (0.5 + 0.5 * self.label_agreement),
            bbox=self.bbox,
            track_id=self.track_id,
            age=self.age,
            attributes={
                "label_agreement": self.label_agreement,
                "missed": float(self.missed),
                "moving": 1.0 if self.is_moving else 0.0,
            },
        )


class ObjectTracker:
    """Greedy IoU tracker with label voting."""

    def __init__(
        self,
        *,
        iou_threshold: float = 0.3,
        max_missed: int = 5,
        min_confidence: float = 0.25,
        position_smoothing: float = 0.55,
    ) -> None:
        self._iou_threshold = iou_threshold
        self._max_missed = max_missed
        self._min_confidence = min_confidence
        #: Exponential smoothing factor for box position; higher = more responsive.
        self._alpha = position_smoothing
        self._tracks: dict[int, Track] = {}
        self._next_id = 1

    def update(self, detections: tuple[Detection, ...], timestamp: float) -> tuple[Detection, ...]:
        """Associate ``detections`` with existing tracks and return stabilised output."""
        candidates = [d for d in detections if d.confidence >= self._min_confidence]

        # Greedy association: strongest detections claim their best-matching track.
        unmatched_tracks = set(self._tracks)
        matched: dict[int, Detection] = {}
        for detection in sorted(candidates, key=lambda d: d.confidence, reverse=True):
            best_id = -1
            best_iou = self._iou_threshold
            for track_id in unmatched_tracks:
                iou = self._tracks[track_id].bbox.iou(detection.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_id = track_id
            if best_id >= 0:
                matched[best_id] = detection
                unmatched_tracks.discard(best_id)
            else:
                self._create(detection, timestamp)

        for track_id, detection in matched.items():
            self._update_track(self._tracks[track_id], detection, timestamp)

        for track_id in list(unmatched_tracks):
            track = self._tracks[track_id]
            track.missed += 1
            track.age += 1
            if track.missed > self._max_missed:
                del self._tracks[track_id]

        return tuple(
            track.to_detection() for track in self._tracks.values() if track.missed == 0
        )

    def _create(self, detection: Detection, timestamp: float) -> None:
        track = Track(
            track_id=self._next_id,
            bbox=detection.bbox,
            label=detection.label,
            confidence=detection.confidence,
            age=1,
            last_seen=timestamp,
        )
        track.label_history.append(detection.label)
        track.confidence_history.append(detection.confidence)
        self._tracks[self._next_id] = track
        self._next_id += 1

    def _update_track(self, track: Track, detection: Detection, timestamp: float) -> None:
        old_cx, old_cy = track.bbox.center
        new_cx, new_cy = detection.bbox.center
        dt = max(1e-3, timestamp - track.last_seen)
        track.velocity = ((new_cx - old_cx) / dt, (new_cy - old_cy) / dt)

        a = self._alpha
        track.bbox = BoundingBox(
            track.bbox.x1 * (1 - a) + detection.bbox.x1 * a,
            track.bbox.y1 * (1 - a) + detection.bbox.y1 * a,
            track.bbox.x2 * (1 - a) + detection.bbox.x2 * a,
            track.bbox.y2 * (1 - a) + detection.bbox.y2 * a,
        )
        track.label = detection.label
        track.confidence = detection.confidence
        track.label_history.append(detection.label)
        track.confidence_history.append(detection.confidence)
        track.age += 1
        track.missed = 0
        track.last_seen = timestamp

    def reset(self) -> None:
        """Forget everything (mode change, camera restart)."""
        self._tracks.clear()

    @property
    def active_tracks(self) -> int:
        return sum(1 for t in self._tracks.values() if t.missed == 0)

    def track(self, track_id: int) -> Track | None:
        return self._tracks.get(track_id)
