"""EMG: filters, features, calibration, quality, pipeline, gestures, intent."""

from __future__ import annotations

import math

import pytest

from neurogrip.core.types import IntentKind
from neurogrip.emg.calibration import (
    CalibrationPhase,
    CalibrationWizard,
    ChannelCalibration,
    EmgCalibration,
)
from neurogrip.emg.features import extract_features, mav, rms, zero_crossings
from neurogrip.emg.filters import Biquad, DCBlocker, EnvelopeFollower, FilterChain, MovingRms
from neurogrip.emg.gestures import ThresholdGestureClassifier, ThresholdSettings
from neurogrip.emg.intent import IntentEngine, IntentSettings
from neurogrip.emg.pipeline import ChannelFrame, EmgFrame, EmgPipeline, PipelineSettings
from neurogrip.emg.quality import SignalQuality
from neurogrip.emg.recorder import AutoRecalibrator, EmgRecorder
from neurogrip.hal.emg.replay import ReplayEmgSource
from neurogrip.hal.emg.simulated import DEFAULT_CHANNELS

RATE = 1000.0


def _frame(flexor: float, extensor: float, timestamp: float = 0.0, quality=SignalQuality.GOOD) -> EmgFrame:
    """Build an EmgFrame directly, bypassing acquisition."""
    return EmgFrame(
        timestamp=timestamp,
        channels=(
            ChannelFrame(index=0, name="Flexor", role="flexor", activation=flexor),
            ChannelFrame(index=1, name="Extensor", role="extensor", activation=extensor),
        ),
        quality=quality,
        quality_score=1.0 if quality >= SignalQuality.GOOD else 0.4,
    )


class TestFilters:
    def test_lowpass_passes_dc_and_blocks_high_frequencies(self):
        filt = Biquad.lowpass(50.0, RATE)
        assert filt.magnitude_at(1.0, RATE) == pytest.approx(1.0, abs=0.02)
        assert filt.magnitude_at(400.0, RATE) < 0.05

    def test_highpass_blocks_dc(self):
        filt = Biquad.highpass(20.0, RATE)
        assert filt.magnitude_at(0.5, RATE) < 0.01
        assert filt.magnitude_at(200.0, RATE) == pytest.approx(1.0, abs=0.05)

    def test_notch_removes_the_mains_tone_but_keeps_its_neighbours(self):
        notch = Biquad.notch(50.0, RATE, q=30.0)
        assert notch.magnitude_at(50.0, RATE) < 0.01
        # The EMG band sits right next to the notch and must survive it.
        assert notch.magnitude_at(80.0, RATE) > 0.9
        assert notch.magnitude_at(150.0, RATE) > 0.95

    def test_dc_blocker_removes_a_constant_offset(self):
        blocker = DCBlocker(0.5, RATE)
        output = 0.0
        for _ in range(4000):
            output = blocker.process(1.0)
        assert abs(output) < 0.02

    def test_envelope_follower_attacks_faster_than_it_releases(self):
        follower = EnvelopeFollower(attack_s=0.01, release_s=0.5, sample_rate_hz=RATE)
        for _ in range(50):  # 50 ms of signal
            follower.process(1.0)
        risen = follower.value
        for _ in range(50):  # 50 ms of silence
            follower.process(0.0)
        assert risen > 0.9
        assert follower.value > 0.5  # released far more slowly than it attacked

    def test_moving_rms_matches_the_analytic_value(self):
        meter = MovingRms(100)
        value = 0.0
        for i in range(1000):
            value = meter.process(math.sin(2 * math.pi * 10 * i / RATE))
        assert value == pytest.approx(1 / math.sqrt(2), abs=0.05)

    def test_chain_rejects_mains_and_passes_emg_band(self):
        chain = FilterChain(RATE, mains_hz=50.0)
        mains_energy = 0.0
        for i in range(2000):
            filtered, _, _ = chain.process(math.sin(2 * math.pi * 50 * i / RATE))
            if i > 1000:
                mains_energy += filtered * filtered

        chain.reset()
        emg_energy = 0.0
        for i in range(2000):
            filtered, _, _ = chain.process(math.sin(2 * math.pi * 120 * i / RATE))
            if i > 1000:
                emg_energy += filtered * filtered

        assert emg_energy > mains_energy * 50


class TestFeatures:
    def test_mav_and_rms_of_a_known_signal(self):
        samples = [1.0, -1.0] * 50
        assert mav(samples) == pytest.approx(1.0)
        assert rms(samples) == pytest.approx(1.0)

    def test_zero_crossings_ignore_sub_threshold_noise(self):
        noisy = [1e-9 * (1 if i % 2 else -1) for i in range(100)]
        assert zero_crossings(noisy) == 0

    def test_zero_crossings_count_real_alternation(self):
        signal = [1.0 if i % 2 else -1.0 for i in range(100)]
        assert zero_crossings(signal) == 99

    def test_feature_vector_order_is_stable(self):
        vector = extract_features(0, [0.1, -0.2, 0.3, -0.4])
        assert vector.as_tuple()[0] == pytest.approx(vector.mav)
        assert len(vector.as_tuple()) == 6


class TestCalibration:
    def test_activation_is_zero_at_rest(self):
        channel = ChannelCalibration(channel=0, rest_mean=1e-5, rest_std=1e-6, full_scale=5e-4)
        assert channel.activation(1e-5) == 0.0
        assert channel.activation(channel.onset_threshold) == 0.0

    def test_activation_saturates_at_full_scale(self):
        channel = ChannelCalibration(channel=0, rest_mean=1e-5, rest_std=1e-6, full_scale=5e-4)
        assert channel.activation(5e-4) == pytest.approx(1.0)
        assert channel.activation(1.0) == pytest.approx(1.0)

    def test_low_snr_calibration_is_rejected(self):
        bad = ChannelCalibration(
            channel=0, rest_mean=1e-4, rest_std=1e-5, mvc=2e-4, full_scale=1.5e-4, samples=100
        )
        assert not bad.is_valid

    def test_missing_channel_falls_back_conservatively(self):
        fallback = EmgCalibration().get(3)
        # The default must be *harder* to trigger than a real calibration.
        assert fallback.onset_threshold > 0
        assert fallback.activation(3e-5) == 0.0

    def test_round_trips_through_json(self, tmp_path, calibration):
        path = tmp_path / "cal.json"
        calibration.save(path)
        loaded = EmgCalibration.load(path)
        assert loaded.channels.keys() == calibration.channels.keys()
        assert loaded.get(0).mvc == pytest.approx(calibration.get(0).mvc)

    def test_wizard_produces_a_valid_calibration(self, clock, emg_source):
        pipeline = EmgPipeline(DEFAULT_CHANNELS, EmgCalibration(), PipelineSettings())
        wizard = CalibrationWizard(DEFAULT_CHANNELS, clock)
        wizard.start()

        drives = {
            CalibrationPhase.REST: (0.0, 0.0),
            CalibrationPhase.FLEXOR_MAX: (0.9, 0.0),
            CalibrationPhase.EXTENSOR_MAX: (0.0, 0.9),
            CalibrationPhase.CO_CONTRACTION: (0.7, 0.7),
        }
        for _ in range(4000):
            if wizard.progress().finished:
                break
            flexor, extensor = drives.get(wizard.phase, (0.0, 0.0))
            emg_source.set_flexor(flexor)
            emg_source.set_extensor(extensor)
            clock.advance(0.01)
            frame = pipeline.process(emg_source.read())
            if frame is not None:
                wizard.update([c.envelope for c in frame.channels])

        assert wizard.phase is CalibrationPhase.COMPLETE
        result = wizard.result
        assert result is not None and result.is_valid
        assert result.get(0).snr_ratio > 4.0

    def test_wizard_is_not_active_before_it_is_started(self, clock):
        wizard = CalibrationWizard(DEFAULT_CHANNELS, clock)
        # Regression: an idle wizard must not look like a running one, or the
        # EMG service would suppress intent estimation for ever.
        assert not wizard.active
        assert not wizard.progress().finished


class TestPipeline:
    def test_produces_activation_proportional_to_drive(self, clock, emg_source, calibration):
        pipeline = EmgPipeline(DEFAULT_CHANNELS, calibration, PipelineSettings())
        results = {}
        for drive in (0.0, 0.3, 0.9):
            emg_source.set_flexor(drive)
            for _ in range(80):
                clock.advance(0.01)
                frame = pipeline.process(emg_source.read())
            results[drive] = frame.flexor

        assert results[0.0] == 0.0
        assert results[0.3] > 0.0
        assert results[0.9] > results[0.3]

    def test_reports_quality_and_detects_a_detached_electrode(self, clock, emg_source, calibration):
        pipeline = EmgPipeline(DEFAULT_CHANNELS, calibration, PipelineSettings())
        for _ in range(150):
            clock.advance(0.01)
            frame = pipeline.process(emg_source.read())
        assert frame.quality >= SignalQuality.FAIR

        emg_source.set_contact_quality(0, 0.0)
        for _ in range(200):
            clock.advance(0.01)
            frame = pipeline.process(emg_source.read())
        assert frame.channels[0].quality is not None
        assert not frame.channels[0].quality.contact_ok

    def test_empty_batch_returns_none(self, calibration):
        pipeline = EmgPipeline(DEFAULT_CHANNELS, calibration, PipelineSettings())
        assert pipeline.process([]) is None

    def test_co_contraction_is_the_minimum_of_both_groups(self):
        assert _frame(0.8, 0.2).co_contraction == pytest.approx(0.2)
        assert _frame(0.8, 0.9).co_contraction == pytest.approx(0.8)


class TestThresholdClassifier:
    def test_rest_below_threshold(self):
        classifier = ThresholdGestureClassifier()
        assert classifier.classify(_frame(0.05, 0.0)).kind is IntentKind.REST

    def test_flexion_reads_as_close(self):
        classifier = ThresholdGestureClassifier()
        assert classifier.classify(_frame(0.6, 0.05)).kind is IntentKind.CLOSE

    def test_extension_reads_as_open(self):
        classifier = ThresholdGestureClassifier()
        assert classifier.classify(_frame(0.05, 0.6)).kind is IntentKind.OPEN

    def test_co_contraction_beats_a_stronger_directional_signal(self):
        classifier = ThresholdGestureClassifier()
        # Flexor dominates numerically, but both groups are active: the user is
        # asking to abort, and that must win.
        result = classifier.classify(_frame(0.9, 0.5))
        assert result.kind is IntentKind.CANCEL

    def test_ambiguous_activation_is_unknown_not_a_guess(self):
        classifier = ThresholdGestureClassifier(
            ThresholdSettings(onset=0.2, separation=0.2, co_contraction=0.9)
        )
        assert classifier.classify(_frame(0.30, 0.28)).kind is IntentKind.UNKNOWN

    def test_hysteresis_keeps_an_active_gesture_alive(self):
        classifier = ThresholdGestureClassifier(ThresholdSettings(onset=0.4, offset=0.15))
        assert classifier.classify(_frame(0.5, 0.0)).kind is IntentKind.CLOSE
        # Between offset and onset: an already-active gesture must persist.
        assert classifier.classify(_frame(0.25, 0.0)).kind is IntentKind.CLOSE
        assert classifier.classify(_frame(0.05, 0.0)).kind is IntentKind.REST


class TestIntentEngine:
    def test_dwell_is_required_before_intent_is_actionable(self, clock):
        engine = IntentEngine(ThresholdGestureClassifier(), clock, IntentSettings(dwell_s=0.2))
        estimate = engine.update(_frame(0.7, 0.0, timestamp=0.0))
        assert estimate.provisional
        assert not estimate.requests_motion

        estimate = engine.update(_frame(0.7, 0.0, timestamp=0.25))
        assert not estimate.provisional
        assert estimate.requests_motion
        assert estimate.kind is IntentKind.CLOSE

    def test_cancel_bypasses_the_normal_dwell(self, clock):
        engine = IntentEngine(
            ThresholdGestureClassifier(), clock, IntentSettings(dwell_s=0.5, cancel_dwell_s=0.02)
        )
        engine.update(_frame(0.7, 0.7, timestamp=0.0))
        estimate = engine.update(_frame(0.7, 0.7, timestamp=0.03))
        assert estimate.kind is IntentKind.CANCEL
        assert estimate.confidence >= 0.9

    def test_unusable_signal_yields_no_intent(self, clock):
        engine = IntentEngine(ThresholdGestureClassifier(), clock)
        estimate = engine.update(_frame(0.9, 0.0, timestamp=0.0, quality=SignalQuality.UNUSABLE))
        assert estimate.kind is IntentKind.REST
        assert estimate.confidence == 0.0
        assert not estimate.requests_motion

    def test_stale_intent_is_not_fresh(self, clock):
        engine = IntentEngine(ThresholdGestureClassifier(), clock, IntentSettings(dwell_s=0.05))
        engine.update(_frame(0.7, 0.0, timestamp=0.0))
        estimate = engine.update(_frame(0.7, 0.0, timestamp=0.1))
        assert estimate.is_fresh(0.2, max_age=0.3)
        assert not estimate.is_fresh(1.0, max_age=0.3)

    def test_release_window_holds_intent_through_a_brief_dip(self, clock):
        engine = IntentEngine(
            ThresholdGestureClassifier(), clock, IntentSettings(dwell_s=0.05, release_s=0.3)
        )
        engine.update(_frame(0.7, 0.0, timestamp=0.0))
        engine.update(_frame(0.7, 0.0, timestamp=0.1))
        held = engine.update(_frame(0.0, 0.0, timestamp=0.2))
        assert held.requests_motion  # still within the release window
        dropped = engine.update(_frame(0.0, 0.0, timestamp=1.0))
        assert dropped.kind is IntentKind.REST


class TestRecordingAndRecalibration:
    def test_recording_round_trips_through_replay(self, tmp_path, clock, emg_source):
        path = tmp_path / "session.emg"
        recorder = EmgRecorder(path, DEFAULT_CHANNELS, sample_rate_hz=1000.0, subject="tester")
        recorder.start()
        recorder.set_label("close")
        for _ in range(20):
            clock.advance(0.01)
            recorder.write(emg_source.read())
        info = recorder.stop()
        assert info.samples > 100

        replay = ReplayEmgSource(path, clock)
        replay.open()
        assert replay.sample_rate_hz == pytest.approx(1000.0)
        assert len(replay.channels) == 2
        clock.advance(1.0)
        assert len(replay.read()) > 0

    def test_auto_recalibration_tracks_a_drifting_baseline(self, clock, calibration):
        recalibrator = AutoRecalibrator(
            calibration, clock, rest_window_s=1.0, min_interval_s=0.0, blend=1.0
        )
        original = calibration.get(0).rest_mean
        drifted = original * 1.5

        events = []
        for step in range(300):
            timestamp = step * 0.01
            frame = EmgFrame(
                timestamp=timestamp,
                channels=(
                    ChannelFrame(index=0, name="Flexor", role="flexor", envelope=drifted),
                    ChannelFrame(index=1, name="Extensor", role="extensor", envelope=drifted),
                ),
            )
            events.extend(recalibrator.update(frame))

        assert events
        assert calibration.get(0).rest_mean == pytest.approx(drifted, rel=0.05)

    def test_implausible_drift_is_refused(self, clock, calibration):
        recalibrator = AutoRecalibrator(
            calibration, clock, rest_window_s=1.0, min_interval_s=0.0, max_adjust_ratio=2.0
        )
        original = calibration.get(0).rest_mean
        absurd = original * 50  # an electrode fault, not drift

        for step in range(300):
            frame = EmgFrame(
                timestamp=step * 0.01,
                channels=(
                    ChannelFrame(index=0, name="Flexor", role="flexor", envelope=absurd),
                ),
            )
            recalibrator.update(frame)

        assert calibration.get(0).rest_mean == pytest.approx(original)
