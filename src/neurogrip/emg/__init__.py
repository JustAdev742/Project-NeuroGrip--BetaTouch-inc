"""EMG acquisition-to-intent processing.

Pipeline overview::

    EmgSource (HAL)
        │  raw samples, volts
        ▼
    EmgPipeline ── FilterChain ─ DC block → notch → band-pass → envelope
        │          Calibration ─ normalise to activation in [0, 1]
        │          Features    ─ MAV / RMS / WL / ZC / SSC
        │          Quality     ─ saturation, noise, mains, dropouts
        ▼
    EmgFrame
        ▼
    GestureClassifier ── threshold rules (default) or learned linear model
        ▼
    IntentEngine ── dwell, hysteresis, cancel fast-path, confidence shaping
        ▼
    IntentEstimate  →  DecisionFusion

Only :class:`~neurogrip.emg.intent.IntentEstimate` crosses into the rest of the
system; nothing downstream sees microvolts.
"""

from __future__ import annotations

from .calibration import (
    CalibrationPhase,
    CalibrationProgress,
    CalibrationWizard,
    ChannelCalibration,
    EmgCalibration,
)
from .features import FeatureVector, extract_features
from .filters import Biquad, BiquadCascade, EnvelopeFollower, FilterChain
from .gestures import (
    GestureClassifier,
    GestureResult,
    ThresholdGestureClassifier,
    ThresholdSettings,
    create_classifier,
)
from .intent import IntentEngine, IntentEstimate, IntentSettings
from .pipeline import ChannelFrame, EmgFrame, EmgPipeline, PipelineSettings
from .quality import ChannelQuality, QualityEstimator, SignalQuality
from .recorder import AutoRecalibrator, EmgRecorder, RecalibrationEvent

__all__ = [
    "AutoRecalibrator",
    "Biquad",
    "BiquadCascade",
    "CalibrationPhase",
    "CalibrationProgress",
    "CalibrationWizard",
    "ChannelCalibration",
    "ChannelFrame",
    "ChannelQuality",
    "EmgCalibration",
    "EmgFrame",
    "EmgPipeline",
    "EmgRecorder",
    "EnvelopeFollower",
    "FeatureVector",
    "FilterChain",
    "GestureClassifier",
    "GestureResult",
    "IntentEngine",
    "IntentEstimate",
    "IntentSettings",
    "PipelineSettings",
    "QualityEstimator",
    "RecalibrationEvent",
    "SignalQuality",
    "ThresholdGestureClassifier",
    "ThresholdSettings",
    "create_classifier",
    "extract_features",
]
