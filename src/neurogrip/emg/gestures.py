"""Gesture classification from processed EMG.

Two implementations ship, behind one interface:

* :class:`ThresholdGestureClassifier` — deterministic, explainable rules over
  flexor/extensor activation. This is the **default**, and deliberately so: for a
  device a person depends on, a classifier whose behaviour can be predicted,
  explained on screen and reproduced exactly is worth more than a few points of
  offline accuracy. It also needs no training data, so a new user is up and
  running after a 20-second calibration.
* :class:`LinearGestureClassifier` — a linear discriminant over the Hudgins
  feature set, with weights loaded from a file. This is the hook for
  pattern-recognition control (more gestures, better separation) once per-user
  training data exists.

Adding a third (an on-device MLP, a temporal CNN) means implementing
:class:`GestureClassifier` and registering it. Nothing else changes.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from ..core.errors import ModelLoadError
from ..core.logging import get_logger
from ..core.types import IntentKind, clamp
from .pipeline import EmgFrame

__all__ = [
    "GestureClassifier",
    "GestureResult",
    "LinearGestureClassifier",
    "ThresholdGestureClassifier",
    "ThresholdSettings",
    "create_classifier",
    "register_classifier",
]

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class GestureResult:
    """Classifier output for one frame."""

    kind: IntentKind
    confidence: float
    #: Proportional effort in ``[0, 1]``: how hard the user is pushing.
    strength: float = 0.0
    #: Per-class scores, for the diagnostics view and for debugging misfires.
    scores: dict[IntentKind, float] = field(default_factory=dict)
    #: Human-readable justification, surfaced in the UI's "why?" panel.
    reason: str = ""

    @property
    def is_actionable(self) -> bool:
        return self.kind is not IntentKind.REST and self.kind is not IntentKind.UNKNOWN


@runtime_checkable
class GestureClassifier(Protocol):
    """Maps a processed EMG frame onto a gesture."""

    @property
    def name(self) -> str: ...

    def classify(self, frame: EmgFrame) -> GestureResult:
        """Classify one frame. Must be pure with respect to hand state."""
        ...

    def reset(self) -> None:
        """Clear any internal history (mode change, recalibration)."""
        ...


@dataclass(frozen=True, slots=True)
class ThresholdSettings:
    """Tuning for :class:`ThresholdGestureClassifier`.

    Defaults are conservative. Every threshold here trades false activations
    (hand moves when the user did not mean it) against missed activations (user
    has to try twice). For a prosthesis, false activations are much worse, so the
    defaults sit on the cautious side and Training Mode helps users build the
    control to lower them.
    """

    #: Activation needed to register intent.
    onset: float = 0.22
    #: Activation below which an active intent releases (hysteresis band).
    offset: float = 0.12
    #: Both groups above this simultaneously means "cancel".
    co_contraction: float = 0.35
    #: Minimum separation between flexor and extensor for a directional gesture.
    separation: float = 0.10
    #: Activation above which the gesture is reported as a strong/held effort.
    hold: float = 0.55


class ThresholdGestureClassifier:
    """Rule-based flexor/extensor classifier with hysteresis.

    Decision order matters and is safety-driven:

    1. **Cancel** (co-contraction) is tested first and wins unconditionally. The
       user's abort signal must never be shadowed by another interpretation.
    2. **Direction** (close vs. open) from whichever group dominates, requiring a
       minimum separation so a sloppy contraction does not flip-flop.
    3. **Rest** otherwise.

    Note what is *not* here: any notion of "hold". Sustained effort is a
    temporal property, and it is interpreted by
    :class:`~neurogrip.emg.intent.IntentEngine`, which owns the clock.

    Hysteresis (``onset`` > ``offset``) prevents chattering at the threshold,
    which would otherwise produce a visibly twitching hand.
    """

    def __init__(self, settings: ThresholdSettings | None = None) -> None:
        self._settings = settings or ThresholdSettings()
        self._active = IntentKind.REST

    @property
    def name(self) -> str:
        return "threshold"

    @property
    def settings(self) -> ThresholdSettings:
        return self._settings

    def reset(self) -> None:
        self._active = IntentKind.REST

    def classify(self, frame: EmgFrame) -> GestureResult:
        s = self._settings
        flexor = frame.flexor
        extensor = frame.extensor
        co = frame.co_contraction
        # Once a gesture is active, use the lower release threshold.
        threshold = s.offset if self._active is not IntentKind.REST else s.onset

        scores = {
            IntentKind.CLOSE: flexor,
            IntentKind.OPEN: extensor,
            IntentKind.CANCEL: co,
            IntentKind.REST: 1.0 - max(flexor, extensor),
        }

        # 1. Cancel takes absolute priority.
        if co >= s.co_contraction:
            self._active = IntentKind.CANCEL
            return GestureResult(
                kind=IntentKind.CANCEL,
                confidence=clamp(0.6 + (co - s.co_contraction) * 1.5),
                strength=co,
                scores=scores,
                reason=f"co-contraction {co:.2f} ≥ {s.co_contraction:.2f}",
            )

        # 2. Direction.
        dominant = flexor - extensor
        magnitude = max(flexor, extensor)

        if magnitude < threshold:
            self._active = IntentKind.REST
            return GestureResult(
                kind=IntentKind.REST,
                confidence=clamp(1.0 - magnitude / max(1e-6, threshold)),
                strength=0.0,
                scores=scores,
                reason=f"activation {magnitude:.2f} < threshold {threshold:.2f}",
            )

        if abs(dominant) < s.separation:
            # Both groups active but neither dominant, and not enough for cancel:
            # ambiguous. Report UNKNOWN rather than guessing — the intent engine
            # treats it as "do nothing", which is the safe interpretation.
            self._active = IntentKind.UNKNOWN
            return GestureResult(
                kind=IntentKind.UNKNOWN,
                confidence=0.3,
                strength=magnitude,
                scores=scores,
                reason=f"ambiguous: flexor {flexor:.2f} vs extensor {extensor:.2f}",
            )

        # Direction only. Whether a sustained effort means "hold" is a temporal
        # judgement and belongs to the intent engine, which owns the clock —
        # promoting to HOLD here would restart the engine's dwell timer on the
        # very next frame, because the reported gesture would have changed.
        kind = IntentKind.CLOSE if dominant > 0 else IntentKind.OPEN
        self._active = kind
        separation_confidence = clamp(abs(dominant) / max(1e-6, s.separation * 3.0))
        margin_confidence = clamp((magnitude - threshold) / max(1e-6, 1.0 - threshold))
        return GestureResult(
            kind=kind,
            confidence=clamp(0.45 + 0.3 * separation_confidence + 0.25 * margin_confidence),
            strength=magnitude,
            scores=scores,
            reason=(
                f"{'flexor' if dominant > 0 else 'extensor'} dominant "
                f"({magnitude:.2f}, margin {abs(dominant):.2f})"
            ),
        )


class LinearGestureClassifier:
    """Linear discriminant over the Hudgins feature vector.

    Model file format (JSON)::

        {
          "classes": ["rest", "close", "open", "cancel"],
          "feature_order": ["mav", "rms", "wl", "zc", "ssc", "var"],
          "channels": [0, 1],
          "mean":  [...],   # per-feature standardisation
          "scale": [...],
          "weights": [[...], ...],   # one row per class
          "bias": [...]
        }

    TODO(training): no weights ship with the repository — a linear model trained
    on one person's electrodes is worthless on another's. ``tools/train_gestures.py``
    (see ``docs/emg.md``) collects labelled windows with
    :class:`~neurogrip.emg.recorder.EmgRecorder` and fits the model per user.
    Until a model exists, :func:`create_classifier` falls back to the threshold
    classifier and says so in the log rather than failing to start.
    """

    def __init__(self, model_path: Path | str) -> None:
        self._path = Path(model_path)
        self._classes: list[IntentKind] = []
        self._channels: list[int] = []
        self._mean: list[float] = []
        self._scale: list[float] = []
        self._weights: list[list[float]] = []
        self._bias: list[float] = []
        self._load()

    @property
    def name(self) -> str:
        return f"linear({self._path.name})"

    def _load(self) -> None:
        if not self._path.exists():
            raise ModelLoadError(f"gesture model not found: {self._path}")
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            self._classes = [IntentKind(c) for c in data["classes"]]
            self._channels = list(data.get("channels", [0, 1]))
            self._mean = [float(v) for v in data["mean"]]
            self._scale = [max(1e-12, float(v)) for v in data["scale"]]
            self._weights = [[float(w) for w in row] for row in data["weights"]]
            self._bias = [float(b) for b in data["bias"]]
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ModelLoadError(f"invalid gesture model {self._path}: {exc}") from exc

        expected = len(self._mean)
        if any(len(row) != expected for row in self._weights):
            raise ModelLoadError("gesture model weight shape does not match feature count")
        if len(self._weights) != len(self._classes) or len(self._bias) != len(self._classes):
            raise ModelLoadError("gesture model class count is inconsistent")

    def reset(self) -> None:
        """Stateless — nothing to reset."""

    def classify(self, frame: EmgFrame) -> GestureResult:
        features: list[float] = []
        for channel_index in self._channels:
            channel = next((c for c in frame.channels if c.index == channel_index), None)
            if channel is None or channel.features is None:
                return GestureResult(
                    kind=IntentKind.UNKNOWN, confidence=0.0, reason="features not ready"
                )
            features.extend(channel.features.as_tuple())

        if len(features) != len(self._mean):
            return GestureResult(
                kind=IntentKind.UNKNOWN,
                confidence=0.0,
                reason=f"feature length {len(features)} != model {len(self._mean)}",
            )

        standardised = [
            (value - mu) / sigma for value, mu, sigma in zip(features, self._mean, self._scale)
        ]
        logits = [
            sum(w * x for w, x in zip(row, standardised)) + bias
            for row, bias in zip(self._weights, self._bias)
        ]
        probabilities = _softmax(logits)
        best = max(range(len(probabilities)), key=probabilities.__getitem__)
        scores = {kind: probabilities[i] for i, kind in enumerate(self._classes)}
        return GestureResult(
            kind=self._classes[best],
            confidence=probabilities[best],
            strength=frame.total_activation,
            scores=scores,
            reason=f"linear model p={probabilities[best]:.2f}",
        )


def _softmax(values: list[float]) -> list[float]:
    """Numerically stable softmax."""
    if not values:
        return []
    peak = max(values)
    exponentials = [math.exp(v - peak) for v in values]
    total = sum(exponentials)
    return [e / total for e in exponentials] if total > 0 else [1.0 / len(values)] * len(values)


#: Registry of available classifiers, keyed by configuration name.
_REGISTRY: dict[str, Callable[..., GestureClassifier]] = {}


def register_classifier(name: str, factory: Callable[..., GestureClassifier]) -> None:
    """Register a classifier implementation under ``name``."""
    _REGISTRY[name] = factory


def create_classifier(
    name: str = "threshold",
    *,
    settings: ThresholdSettings | None = None,
    model_path: Path | str | None = None,
) -> GestureClassifier:
    """Instantiate the configured classifier.

    Falls back to the threshold classifier with a warning when a learned model is
    requested but unavailable. Refusing to start would be the wrong trade: an
    explainable classifier that works is better than no hand at all.
    """
    if name == "threshold":
        return ThresholdGestureClassifier(settings)
    if name == "linear":
        if model_path is None:
            log.warning("linear classifier requested without a model path; using threshold rules")
            return ThresholdGestureClassifier(settings)
        try:
            return LinearGestureClassifier(model_path)
        except ModelLoadError as exc:
            log.warning(
                "gesture model unavailable; falling back to threshold rules", error=str(exc)
            )
            return ThresholdGestureClassifier(settings)
    factory = _REGISTRY.get(name)
    if factory is None:
        log.warning("unknown gesture classifier; using threshold rules", requested=name)
        return ThresholdGestureClassifier(settings)
    return factory(settings=settings, model_path=model_path)
