"""The EMG processing pipeline.

Turns raw :class:`~neurogrip.hal.emg.base.EmgSample` batches into
:class:`EmgFrame` objects: filtered waveforms, envelopes, normalised activations,
feature vectors and a quality assessment. Everything downstream — the gesture
classifier, the intent engine, the muscle visualisation, the training exercises —
consumes ``EmgFrame`` and nothing else.

The pipeline is *pull*-driven and stateless between calls apart from its filter
state, so it can be run over a live source, a recording, or a synthetic signal
with identical results.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from ..core.ringbuffer import RingBuffer
from ..core.types import clamp
from ..hal.emg.base import EmgChannelSpec, EmgSample
from .calibration import EmgCalibration
from .features import FeatureVector, extract_features
from .filters import FilterChain
from .quality import ChannelQuality, QualityEstimator, SignalQuality

__all__ = ["ChannelFrame", "EmgFrame", "EmgPipeline", "PipelineSettings"]


@dataclass(frozen=True, slots=True)
class ChannelFrame:
    """Processed state of one channel at one instant."""

    index: int
    name: str
    role: str
    #: Latest band-pass filtered sample (for the waveform display).
    filtered: float = 0.0
    #: Smoothed envelope in volts.
    envelope: float = 0.0
    #: Windowed RMS in volts.
    rms: float = 0.0
    #: Calibrated activation in ``[0, 1]`` — the number control actually uses.
    activation: float = 0.0
    #: Rate of change of activation, per second. Positive = contracting.
    slope: float = 0.0
    features: FeatureVector | None = None
    quality: ChannelQuality | None = None

    @property
    def is_active(self) -> bool:
        return self.activation > 0.0

    @property
    def quality_score(self) -> float:
        return self.quality.score if self.quality else 1.0


@dataclass(frozen=True, slots=True)
class EmgFrame:
    """One processed multi-channel observation."""

    timestamp: float
    channels: tuple[ChannelFrame, ...]
    quality: SignalQuality = SignalQuality.GOOD
    quality_score: float = 1.0
    sample_count: int = 0
    #: Accumulated dropouts reported by the acquisition device.
    dropped_samples: int = 0
    reasons: tuple[str, ...] = field(default_factory=tuple)

    # -- role-based accessors -------------------------------------------------

    def by_role(self, role: str) -> tuple[ChannelFrame, ...]:
        return tuple(c for c in self.channels if c.role == role)

    @property
    def flexor(self) -> float:
        """Strongest flexor activation."""
        return max((c.activation for c in self.channels if c.role == "flexor"), default=0.0)

    @property
    def extensor(self) -> float:
        """Strongest extensor activation."""
        return max((c.activation for c in self.channels if c.role == "extensor"), default=0.0)

    @property
    def co_contraction(self) -> float:
        """Simultaneous flexor+extensor activation in ``[0, 1]``.

        Defined as the *minimum* of the two, so it only rises when both groups
        are genuinely active — the signature of the deliberate cancel gesture.
        """
        return min(self.flexor, self.extensor)

    @property
    def differential(self) -> float:
        """Flexor minus extensor in ``[-1, 1]``: the proportional control signal."""
        return self.flexor - self.extensor

    @property
    def total_activation(self) -> float:
        return max((c.activation for c in self.channels), default=0.0)

    @property
    def is_resting(self) -> bool:
        """True when every channel is below its onset threshold."""
        return self.total_activation <= 0.0

    def channel(self, index: int) -> ChannelFrame:
        return self.channels[index]

    def activations(self) -> tuple[float, ...]:
        return tuple(c.activation for c in self.channels)


@dataclass(frozen=True, slots=True)
class PipelineSettings:
    """Tunable pipeline parameters, loaded from ``[emg]`` configuration."""

    sample_rate_hz: float = 1000.0
    mains_hz: float = 50.0
    band_low_hz: float = 20.0
    band_high_hz: float = 400.0
    envelope_attack_s: float = 0.03
    envelope_release_s: float = 0.15
    median_size: int = 5
    rms_window_s: float = 0.2
    #: Feature window length; 200 ms is the usual latency/accuracy compromise for
    #: myoelectric pattern recognition.
    feature_window_s: float = 0.2
    #: Recompute features at most this often (they are the expensive part).
    feature_interval_s: float = 0.05


class EmgPipeline:
    """Per-channel filtering, normalisation, feature extraction and quality."""

    def __init__(
        self,
        channels: Sequence[EmgChannelSpec],
        calibration: EmgCalibration,
        settings: PipelineSettings | None = None,
    ) -> None:
        self._channels = tuple(channels)
        self._calibration = calibration
        self._settings = settings or PipelineSettings()
        rate = self._settings.sample_rate_hz

        self._chains = {
            spec.index: FilterChain(
                rate,
                mains_hz=self._settings.mains_hz,
                band_low_hz=self._settings.band_low_hz,
                band_high_hz=self._settings.band_high_hz,
                attack_s=self._settings.envelope_attack_s,
                release_s=self._settings.envelope_release_s,
                median_size=self._settings.median_size,
                rms_window_s=self._settings.rms_window_s,
            )
            for spec in self._channels
        }
        window_samples = max(16, int(self._settings.feature_window_s * rate))
        self._windows = {spec.index: RingBuffer(window_samples) for spec in self._channels}
        self._quality = {
            spec.index: QualityEstimator(
                spec.index,
                sample_rate_hz=rate,
                full_scale_v=spec.full_scale_v / 2000.0,
                mains_hz=self._settings.mains_hz,
            )
            for spec in self._channels
        }
        self._last_activation = {spec.index: 0.0 for spec in self._channels}
        self._last_features: dict[int, FeatureVector | None] = {
            spec.index: None for spec in self._channels
        }
        self._last_feature_time = 0.0
        self._last_frame_time = 0.0
        self._total_dropped = 0

    # -- configuration --------------------------------------------------------

    @property
    def calibration(self) -> EmgCalibration:
        return self._calibration

    def set_calibration(self, calibration: EmgCalibration) -> None:
        """Install a new calibration. Filter state is preserved deliberately —
        re-normalising does not invalidate the band-pass history."""
        self._calibration = calibration

    def reset(self) -> None:
        """Clear all filter and window state (electrode reseat, mode change)."""
        for chain in self._chains.values():
            chain.reset()
        for window in self._windows.values():
            window.clear()
        for estimator in self._quality.values():
            estimator.reset()
        self._last_activation = {spec.index: 0.0 for spec in self._channels}

    # -- processing -----------------------------------------------------------

    def process(self, samples: Sequence[EmgSample], *, dropped: int = 0) -> EmgFrame | None:
        """Process a batch of raw samples into a single frame.

        Returns ``None`` for an empty batch. The frame reflects the *last* sample
        in the batch: the pipeline runs every sample through the filters (so no
        signal is lost) but publishes state at the batch rate, which is the rate
        the control loop can actually consume.
        """
        if not samples:
            return None

        self._total_dropped += dropped
        timestamp = samples[-1].timestamp
        dt = timestamp - self._last_frame_time if self._last_frame_time else 0.0
        self._last_frame_time = timestamp

        latest: dict[int, tuple[float, float, float]] = {}
        for sample in samples:
            for spec in self._channels:
                if spec.index >= len(sample.values):
                    continue
                chain = self._chains[spec.index]
                filtered, envelope, rms = chain.process(sample.values[spec.index])
                self._windows[spec.index].append(filtered)
                self._quality[spec.index].add_sample(filtered, chain.prenotch)
                latest[spec.index] = (filtered, envelope, rms)

        recompute_features = (
            timestamp - self._last_feature_time >= self._settings.feature_interval_s
        )
        if recompute_features:
            self._last_feature_time = timestamp

        channel_frames: list[ChannelFrame] = []
        reasons: list[str] = []
        worst = SignalQuality.EXCELLENT
        score_sum = 0.0

        for spec in self._channels:
            filtered, envelope, rms = latest.get(spec.index, (0.0, 0.0, 0.0))
            channel_cal = self._calibration.get(spec.index)
            activation = channel_cal.activation(envelope)

            slope = 0.0
            if dt > 1e-4:
                slope = (activation - self._last_activation[spec.index]) / dt
            self._last_activation[spec.index] = activation

            if recompute_features:
                self._last_features[spec.index] = extract_features(
                    spec.index, self._windows[spec.index].to_list()
                )

            self._quality[spec.index].note_dropouts(dropped, len(samples))
            quality = self._quality[spec.index].evaluate(
                rest_noise_floor=channel_cal.rest_mean, active=activation > 0.05
            )
            worst = min(worst, quality.quality)
            score_sum += quality.score
            reasons.extend(f"{spec.name}: {reason}" for reason in quality.reasons)

            channel_frames.append(
                ChannelFrame(
                    index=spec.index,
                    name=spec.name,
                    role=spec.role,
                    filtered=filtered,
                    envelope=envelope,
                    rms=rms,
                    activation=activation,
                    slope=slope,
                    features=self._last_features[spec.index],
                    quality=quality,
                )
            )

        return EmgFrame(
            timestamp=timestamp,
            channels=tuple(channel_frames),
            quality=worst,
            quality_score=clamp(score_sum / len(self._channels)) if self._channels else 0.0,
            sample_count=len(samples),
            dropped_samples=self._total_dropped,
            reasons=tuple(reasons),
        )

    @property
    def channels(self) -> tuple[EmgChannelSpec, ...]:
        return self._channels

    @property
    def settings(self) -> PipelineSettings:
        return self._settings
