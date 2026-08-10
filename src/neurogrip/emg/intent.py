"""Intent estimation — the user's half of shared control.

The gesture classifier answers "what does this instant look like?". The intent
engine answers the question the rest of the system actually needs: **"does the
user want something, how sure are we, and how strongly?"**

Everything the AI is allowed to do hangs off the output of this module. If no
:class:`IntentEstimate` with ``requests_motion`` is present and fresh, the fusion
layer cannot produce a motion decision — that is the structural expression of
"the AI never moves the hand on its own".

Three mechanisms turn a noisy per-frame classification into a trustworthy intent:

* **Dwell** — a gesture must persist before it counts, which removes transients
  and classifier flicker. Cancel has a much shorter dwell, because an abort must
  feel instant.
* **Hysteresis** — releasing an intent uses a lower threshold than starting one,
  so a sustained grasp does not drop out during natural effort dips.
* **Confidence shaping** — the reported confidence folds in signal quality,
  dwell satisfaction and classifier margin, so downstream code gets one honest
  number instead of having to re-derive trust from raw parts.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from ..core.clock import Clock
from ..core.logging import get_logger
from ..core.types import IntentKind, clamp
from .gestures import GestureClassifier, GestureResult
from .pipeline import EmgFrame
from .quality import SignalQuality

__all__ = ["IntentEngine", "IntentEstimate", "IntentSettings"]

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class IntentEstimate:
    """What the user is asking for, with the evidence behind it."""

    kind: IntentKind
    #: Overall trust in this estimate, in ``[0, 1]``.
    confidence: float
    #: Proportional effort, in ``[0, 1]`` — drives speed and grip force.
    strength: float
    timestamp: float
    #: Seconds this intent has been continuously held.
    duration: float = 0.0
    quality: SignalQuality = SignalQuality.GOOD
    #: Signed flexor-minus-extensor value for direct proportional control.
    differential: float = 0.0
    activations: tuple[float, ...] = field(default_factory=tuple)
    reasons: tuple[str, ...] = field(default_factory=tuple)
    #: True while the intent is still accumulating dwell time.
    provisional: bool = False

    def age(self, now: float) -> float:
        return now - self.timestamp

    def is_fresh(self, now: float, max_age: float = 0.3) -> bool:
        """Whether this estimate is recent enough to act on.

        Stale intent must never authorise motion: if EMG stops arriving, the
        hand must stop, not continue on the last command it saw.
        """
        return self.age(now) <= max_age

    @property
    def requests_motion(self) -> bool:
        return self.kind.requests_motion and not self.provisional

    @property
    def is_cancel(self) -> bool:
        return self.kind is IntentKind.CANCEL

    @classmethod
    def resting(cls, timestamp: float, quality: SignalQuality = SignalQuality.GOOD) -> IntentEstimate:
        return cls(
            kind=IntentKind.REST,
            confidence=1.0,
            strength=0.0,
            timestamp=timestamp,
            quality=quality,
            reasons=("no muscle activity",),
        )

    def __str__(self) -> str:  # pragma: no cover - display helper
        tag = "~" if self.provisional else ""
        return f"{tag}{self.kind.value}({self.confidence:.2f}, s={self.strength:.2f})"


@dataclass(frozen=True, slots=True)
class IntentSettings:
    """Timing and confidence tuning for :class:`IntentEngine`."""

    #: Seconds a directional gesture must persist before it counts as intent.
    dwell_s: float = 0.12
    #: Dwell for the cancel gesture — short, because aborting must feel immediate.
    cancel_dwell_s: float = 0.04
    #: Seconds an intent survives after the gesture stops (hysteresis in time).
    release_s: float = 0.15
    #: Maximum age at which an estimate may still authorise motion.
    max_age_s: float = 0.30
    #: Two flexor pulses shorter than this, within the window below, mean TOGGLE.
    pulse_max_s: float = 0.35
    toggle_window_s: float = 0.9
    #: Quality below this hard-blocks intent (a detached electrode says nothing).
    min_quality: SignalQuality = SignalQuality.POOR
    #: Weight given to signal quality when shaping confidence.
    quality_weight: float = 0.35
    #: Confidence below this is reported but never treated as actionable.
    min_confidence: float = 0.35
    #: A confirmed CLOSE held at or above ``hold_strength`` for this long is
    #: reported as HOLD, which tells the control layer to maintain the grip
    #: rather than keep driving further closed.
    hold_after_s: float = 0.5
    hold_strength: float = 0.55


class IntentEngine:
    """Turns per-frame gestures into a stable, timed intent estimate."""

    def __init__(
        self,
        classifier: GestureClassifier,
        clock: Clock,
        settings: IntentSettings | None = None,
    ) -> None:
        self._classifier = classifier
        self._clock = clock
        self._settings = settings or IntentSettings()

        self._candidate: IntentKind = IntentKind.REST
        self._candidate_since = 0.0
        self._confirmed: IntentKind = IntentKind.REST
        self._confirmed_since = 0.0
        self._last_active_at = 0.0
        self._pulse_times: deque[float] = deque(maxlen=4)
        self._pulse_started = 0.0
        self._in_pulse = False
        self._latest = IntentEstimate.resting(0.0)
        #: Counters for the diagnostics screen.
        self.transitions = 0
        self.cancels = 0
        self.toggles = 0

    # -- configuration --------------------------------------------------------

    @property
    def settings(self) -> IntentSettings:
        return self._settings

    def set_settings(self, settings: IntentSettings) -> None:
        """Swap tuning at runtime — Sports Mode shortens dwell, for instance."""
        self._settings = settings

    def set_classifier(self, classifier: GestureClassifier) -> None:
        self._classifier = classifier
        self.reset()

    def reset(self) -> None:
        """Clear all timing state (mode change, calibration, e-stop recovery)."""
        self._classifier.reset()
        self._candidate = IntentKind.REST
        self._confirmed = IntentKind.REST
        self._pulse_times.clear()
        self._in_pulse = False
        self._latest = IntentEstimate.resting(self._clock.monotonic())

    # -- estimation -----------------------------------------------------------

    @property
    def latest(self) -> IntentEstimate:
        """Most recent estimate, without recomputation."""
        return self._latest

    def update(self, frame: EmgFrame) -> IntentEstimate:
        """Process one EMG frame and return the current intent estimate."""
        now = frame.timestamp
        gesture = self._classifier.classify(frame)
        reasons: list[str] = [gesture.reason] if gesture.reason else []

        # Hard gate: an unusable signal cannot express intent, whatever the
        # classifier says about it.
        if frame.quality < self._settings.min_quality:
            self._candidate = IntentKind.REST
            self._confirmed = IntentKind.REST
            estimate = IntentEstimate(
                kind=IntentKind.REST,
                confidence=0.0,
                strength=0.0,
                timestamp=now,
                quality=frame.quality,
                activations=frame.activations(),
                reasons=("signal quality too low for intent", *frame.reasons[:2]),
            )
            self._latest = estimate
            return estimate

        self._track_pulses(frame, now)

        # Cancel bypasses the normal dwell machinery entirely.
        if gesture.kind is IntentKind.CANCEL:
            return self._emit_cancel(frame, gesture, now, reasons)

        toggle = self._detect_toggle(now)
        if toggle:
            return self._emit_toggle(frame, now)

        # Dwell accumulation.
        if gesture.kind != self._candidate:
            self._candidate = gesture.kind
            self._candidate_since = now
        held = now - self._candidate_since

        required = self._settings.dwell_s
        confirmed = held >= required

        if confirmed and gesture.kind is not IntentKind.REST:
            if self._confirmed is not gesture.kind:
                self._confirmed = gesture.kind
                self._confirmed_since = now
                self.transitions += 1
            self._last_active_at = now
        elif gesture.kind is IntentKind.REST:
            # Time-domain hysteresis: hold the previous intent briefly so a dip
            # in effort does not drop an in-progress grasp.
            if now - self._last_active_at <= self._settings.release_s and self._confirmed.requests_motion:
                reasons.append("holding through release window")
                estimate = self._build(
                    self._confirmed,
                    frame,
                    gesture,
                    now,
                    confidence_scale=0.8,
                    reasons=reasons,
                )
                self._latest = estimate
                return estimate
            self._confirmed = IntentKind.REST
            self._confirmed_since = now

        kind = self._confirmed if confirmed else IntentKind.REST
        provisional = (not confirmed) and gesture.kind is not IntentKind.REST
        if provisional:
            reasons.append(f"dwell {held * 1000:.0f}/{required * 1000:.0f} ms")
            kind = gesture.kind
        elif (
            kind is IntentKind.CLOSE
            and gesture.strength >= self._settings.hold_strength
            and now - self._confirmed_since >= self._settings.hold_after_s
        ):
            kind = IntentKind.HOLD
            reasons.append("sustained effort — holding")

        estimate = self._build(
            kind, frame, gesture, now, reasons=reasons, provisional=provisional
        )
        self._latest = estimate
        return estimate

    # -- helpers --------------------------------------------------------------

    def _build(
        self,
        kind: IntentKind,
        frame: EmgFrame,
        gesture: GestureResult,
        now: float,
        *,
        confidence_scale: float = 1.0,
        reasons: list[str] | None = None,
        provisional: bool = False,
    ) -> IntentEstimate:
        """Assemble an estimate, shaping confidence by quality and dwell."""
        quality_factor = clamp(frame.quality_score)
        weight = self._settings.quality_weight
        confidence = gesture.confidence * (1.0 - weight) + gesture.confidence * quality_factor * weight
        confidence *= confidence_scale
        if provisional:
            confidence *= 0.6

        if kind is IntentKind.REST:
            confidence = clamp(max(confidence, 1.0 - frame.total_activation))

        return IntentEstimate(
            kind=kind,
            confidence=clamp(confidence),
            strength=clamp(gesture.strength),
            timestamp=now,
            duration=now - self._confirmed_since if kind is self._confirmed else 0.0,
            quality=frame.quality,
            differential=frame.differential,
            activations=frame.activations(),
            reasons=tuple(reasons or ()),
            provisional=provisional,
        )

    def _emit_cancel(
        self, frame: EmgFrame, gesture: GestureResult, now: float, reasons: list[str]
    ) -> IntentEstimate:
        """Cancel path: minimal dwell, maximum priority."""
        if self._candidate is not IntentKind.CANCEL:
            self._candidate = IntentKind.CANCEL
            self._candidate_since = now
        held = now - self._candidate_since
        if held < self._settings.cancel_dwell_s:
            reasons.append("cancel dwell")
            estimate = self._build(
                IntentKind.REST, frame, gesture, now, reasons=reasons, provisional=True
            )
            self._latest = estimate
            return estimate

        if self._confirmed is not IntentKind.CANCEL:
            self.cancels += 1
            self._confirmed_since = now
            log.info("user cancel detected", co_contraction=round(frame.co_contraction, 3))
        self._confirmed = IntentKind.CANCEL
        self._last_active_at = now
        estimate = IntentEstimate(
            kind=IntentKind.CANCEL,
            # Cancel is reported at high confidence on purpose: refusing to abort
            # because we were unsure is not an acceptable failure mode.
            confidence=clamp(max(0.9, gesture.confidence)),
            strength=gesture.strength,
            timestamp=now,
            duration=now - self._confirmed_since,
            quality=frame.quality,
            differential=frame.differential,
            activations=frame.activations(),
            reasons=(*reasons, "co-contraction abort"),
        )
        self._latest = estimate
        return estimate

    def _track_pulses(self, frame: EmgFrame, now: float) -> None:
        """Record short flexor pulses used for the hands-free mode toggle."""
        active = frame.flexor > self._settings.min_confidence
        if active and not self._in_pulse:
            self._in_pulse = True
            self._pulse_started = now
        elif not active and self._in_pulse:
            self._in_pulse = False
            if now - self._pulse_started <= self._settings.pulse_max_s:
                self._pulse_times.append(now)

    def _detect_toggle(self, now: float) -> bool:
        """Two short pulses inside the toggle window."""
        if len(self._pulse_times) < 2:
            return False
        first, second = self._pulse_times[-2], self._pulse_times[-1]
        if second - first > self._settings.toggle_window_s:
            return False
        return now - second <= self._settings.toggle_window_s

    def _emit_toggle(self, frame: EmgFrame, now: float) -> IntentEstimate:
        self._pulse_times.clear()
        self.toggles += 1
        estimate = IntentEstimate(
            kind=IntentKind.TOGGLE,
            confidence=0.8,
            strength=frame.total_activation,
            timestamp=now,
            quality=frame.quality,
            activations=frame.activations(),
            reasons=("double pulse detected",),
        )
        self._latest = estimate
        return estimate
