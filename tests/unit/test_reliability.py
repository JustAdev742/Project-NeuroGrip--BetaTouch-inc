"""Reliability and bring-up: the mechanisms added by the production-readiness audit.

Each class here corresponds to a defect the audit found in the integrated system
rather than in an individual module — a mechanism that existed but was wired to
nothing, or a failure mode nothing handled. The tests are written to fail if that
wiring is ever removed again, which is the only thing that stops the same gap
reappearing.
"""

from __future__ import annotations

import json

import pytest

from neurogrip.control.controller import HandController
from neurogrip.control.servo_calibration import (
    MAX_ACCEPTABLE_SLACK,
    ServoCalibrationPhase,
    ServoCalibrationSet,
    ServoCalibrationWizard,
)
from neurogrip.core.clock import SimulatedClock
from neurogrip.core.config import ConfigLoader
from neurogrip.core.errors import (
    CalibrationError,
    CommunicationError,
    ConfigurationError,
    DeviceNotAvailableError,
)
from neurogrip.core.profiles import ProfileStore, UserProfile
from neurogrip.core.runstate import RunMarker, ShutdownReason
from neurogrip.core.types import Finger, HandPose
from neurogrip.core.validation import (
    Rule,
    ValidationSeverity,
    validate_config,
)
from neurogrip.diagnostics.bringup import EstopTester, LinkTester, RangeTester
from neurogrip.diagnostics.selftest import TestOutcome
from neurogrip.hal.base import DeviceInfo, DeviceKind
from neurogrip.hal.protocol import encode_set_calibration
from neurogrip.hal.servo.base import ServoCalibration
from neurogrip.hal.servo.simulated import SimulatedServoBus, TendonModel
from neurogrip.hal.transport.reconnecting import ReconnectingTransport
from neurogrip.safety.estop import EmergencyStop
from neurogrip.safety.monitor import SafetyMonitor
from neurogrip.safety.watchdog import WatchdogGroup

# ---------------------------------------------------------------------------
# Servo calibration
# ---------------------------------------------------------------------------


def _hand(clock, tendons=None, bus=None):
    """A started, enabled controller over a simulated bus with known tendons."""
    from neurogrip.core.events import EventBus

    servo = SimulatedServoBus(clock, tendons=tendons)
    controller = HandController(servo, clock, bus or EventBus(clock))
    controller.start()
    controller.enable()
    for _ in range(50):
        clock.advance(0.005)
        controller.tick()
    return controller, servo


def _run_wizard(wizard, controller, clock, limit=400_000):
    for _ in range(limit):
        clock.advance(0.005)
        wizard.update()
        controller.tick()
        if wizard.progress().finished:
            return True
    return False


class TestServoCalibration:
    def test_measures_the_slack_the_hand_actually_has(self, clock: SimulatedClock):
        truth = [0.20, 0.10, 0.15, 0.05, 0.25]
        controller, _ = _hand(clock, TendonModel(slack=list(truth)))
        wizard = ServoCalibrationWizard(controller, clock)
        wizard.start()

        assert _run_wizard(wizard, controller, clock)
        assert wizard.phase is ServoCalibrationPhase.COMPLETE
        result = wizard.result
        assert result is not None
        for finger in Finger:
            measured = result.get(finger).slack
            # The bias is the filter's response time at the creep rate; anything
            # beyond a few percent means the detection is wrong, not merely late.
            assert measured == pytest.approx(truth[int(finger)], abs=0.03)

    def test_reports_a_tendon_too_loose_to_calibrate(self, clock: SimulatedClock):
        controller, _ = _hand(clock, TendonModel(slack=[0.75, 0.0, 0.0, 0.0, 0.0]))
        wizard = ServoCalibrationWizard(controller, clock)
        wizard.start()
        assert _run_wizard(wizard, controller, clock)

        assert wizard.phase is ServoCalibrationPhase.FAILED
        thumb = next(r for r in wizard.results if r.finger is Finger.THUMB)
        assert not thumb.ok
        assert "re-string" in " ".join(thumb.problems)
        # The other four are still measured: one bad tendon must not cost the
        # operator the whole run.
        assert all(r.ok for r in wizard.results if r.finger is not Finger.THUMB)

    def test_refuses_to_run_while_stopped(self, clock: SimulatedClock):
        controller, _ = _hand(clock)
        controller.emergency_stop("test")
        controller.tick()
        wizard = ServoCalibrationWizard(controller, clock)
        with pytest.raises(CalibrationError, match="emergency stop"):
            wizard.start()

    def test_calibrating_one_finger_keeps_the_others(self, clock: SimulatedClock):
        base = ServoCalibrationSet()
        for finger in Finger:
            base.set(ServoCalibration(finger=finger, slack=0.42))
        controller, _ = _hand(clock, TendonModel(slack=[0.10, 0.0, 0.0, 0.0, 0.0]))
        wizard = ServoCalibrationWizard(controller, clock, base=base)
        wizard.start((Finger.THUMB,))
        assert _run_wizard(wizard, controller, clock)

        result = wizard.result
        assert result is not None
        assert result.get(Finger.THUMB).slack == pytest.approx(0.10, abs=0.03)
        assert result.get(Finger.INDEX).slack == pytest.approx(0.42)

    def test_calibration_makes_commanded_closure_mean_finger_closure(
        self, clock: SimulatedClock
    ):
        """The point of the whole exercise, asserted directly."""
        truth = TendonModel(slack=[0.25] * 5)
        controller, _ = _hand(clock, truth)

        controller.move_to(HandPose.uniform(0.5), force=0.4, speed=0.6)
        for _ in range(600):
            clock.advance(0.005)
            controller.tick()
        uncalibrated = controller.state.pose[Finger.INDEX]

        # Apply the calibration the wizard would have produced.
        controller.set_calibration(
            [ServoCalibration(finger=f, slack=0.25) for f in Finger]
        )
        controller.move_to(HandPose.uniform(0.5), force=0.4, speed=0.6)
        for _ in range(600):
            clock.advance(0.005)
            controller.tick()
        calibrated = controller.state.pose[Finger.INDEX]

        assert uncalibrated < 0.40, "uncalibrated slack should cause under-travel"
        assert calibrated == pytest.approx(0.5, abs=0.05)

    def test_round_trips_through_a_file(self, tmp_path):
        original = ServoCalibrationSet(hand_id="unit-1")
        for finger in Finger:
            original.set(
                ServoCalibration(finger=finger, min_pulse_us=900, max_pulse_us=2100, slack=0.13)
            )
        path = tmp_path / "servo.json"
        original.save(path)
        restored = ServoCalibrationSet.load(path)

        assert restored.hand_id == "unit-1"
        assert restored.is_complete
        for finger in Finger:
            assert restored.get(finger).slack == pytest.approx(0.13)
            assert restored.get(finger).min_pulse_us == 900

    def test_slack_reaches_the_wire(self):
        """Slack was silently dropped by the protocol before the audit."""
        payload = encode_set_calibration(
            Finger.THUMB, min_pulse_us=1000, max_pulse_us=2000, inverted=False, slack=0.5
        )
        assert len(payload) == 7, "one byte per field plus the slack byte"
        assert payload[-1] == 128  # 0.5 * 255, rounded

    def test_missing_calibration_defaults_to_no_slack(self):
        # Under-travel is the safe direction to be wrong in.
        assert ServoCalibrationSet().get(Finger.RING).slack == 0.0
        assert MAX_ACCEPTABLE_SLACK < 1.0


# ---------------------------------------------------------------------------
# Reconnection
# ---------------------------------------------------------------------------


class FlakyTransport:
    """A transport that can be made to fail on demand."""

    def __init__(self) -> None:
        self.opened = False
        self.fail_open = False
        self.fail_io = False
        self.opens = 0

    def open(self) -> None:
        if self.fail_open:
            raise DeviceNotAvailableError("device absent")
        self.opened = True
        self.opens += 1

    def close(self) -> None:
        self.opened = False

    @property
    def is_open(self) -> bool:
        return self.opened

    def write(self, data: bytes) -> int:
        if self.fail_io:
            raise CommunicationError("cable pulled")
        return len(data)

    def read(self, max_bytes: int = 4096) -> bytes:
        if self.fail_io:
            raise CommunicationError("cable pulled")
        return b"payload"

    def info(self) -> DeviceInfo:
        return DeviceInfo(name="flaky", kind=DeviceKind.SERVO_BUS, driver="test")


class TestReconnectingTransport:
    def test_recovers_after_the_backoff(self, clock: SimulatedClock):
        inner = FlakyTransport()
        restored: list[float] = []
        link = ReconnectingTransport(
            inner, clock, on_reconnect=lambda: restored.append(clock.monotonic())
        )
        link.open()

        inner.fail_io = True
        assert link.read() == b""
        assert not link.connected
        inner.fail_io = False

        # Too soon: retrying every cycle would spin the control loop.
        assert link.read() == b""
        assert inner.opens == 1

        clock.advance(0.6)
        link.read()
        assert link.connected
        assert link.reconnects == 1
        assert len(restored) == 1, "the driver must be told to restore its state"

    def test_backoff_grows_and_is_capped(self, clock: SimulatedClock):
        inner = FlakyTransport()
        link = ReconnectingTransport(inner, clock, initial_backoff_s=0.5, max_backoff_s=4.0)
        link.open()
        inner.fail_io = True
        link.read()
        inner.fail_open = True

        for _ in range(8):
            clock.advance(10.0)
            link.read()

        assert link.failed_attempts == 8
        assert link._backoff == pytest.approx(4.0)

    def test_write_raises_rather_than_silently_dropping(self, clock: SimulatedClock):
        inner = FlakyTransport()
        link = ReconnectingTransport(inner, clock)
        link.open()
        inner.fail_io = True
        with pytest.raises(CommunicationError):
            link.write(b"x")
        # A dropped command must never look like a delivered one.
        with pytest.raises(DeviceNotAvailableError):
            link.write(b"x")

    def test_close_stops_reconnection(self, clock: SimulatedClock):
        inner = FlakyTransport()
        link = ReconnectingTransport(inner, clock)
        link.open()
        link.close()
        clock.advance(60.0)
        link.read()
        assert inner.opens == 1, "a deliberate close must not be undone"

    def test_a_device_absent_at_startup_still_recovers(self, clock: SimulatedClock):
        inner = FlakyTransport()
        inner.fail_open = True
        link = ReconnectingTransport(inner, clock)
        with pytest.raises(DeviceNotAvailableError):
            link.open()

        inner.fail_open = False
        clock.advance(1.0)
        link.read()
        assert link.connected


# ---------------------------------------------------------------------------
# Configuration validation
# ---------------------------------------------------------------------------


def _config(mapping):
    return ConfigLoader().add_mapping(mapping).build()


class TestConfigValidation:
    def test_shipped_configuration_is_valid(self):
        from neurogrip.runtime.bootstrap import load_configuration

        for profile in (None, "simulation", "hardware"):
            report = validate_config(load_configuration(profile=profile, profiles=False))
            assert report.ok, f"{profile}: {report.describe()}"
            assert not report.warnings, f"{profile}: {report.describe()}"

    def test_out_of_range_value_is_an_error(self):
        report = validate_config(_config({"servo": {"max_force": 2.5}}))
        assert not report.ok
        assert any("max_force" in str(i) for i in report.errors)

    def test_misspelled_key_is_reported(self):
        report = validate_config(_config({"servo": {"max_forse": 0.4}}))
        assert any(i.path == "servo.max_forse" for i in report.warnings)

    def test_watchdog_shorter_than_its_loop_is_refused(self):
        report = validate_config(
            _config({"runtime": {"control_hz": 200.0}, "safety": {"watchdogs": {"control_s": 0.006}}})
        )
        assert any("control_s" in i.path for i in report.errors)

    def test_inverted_emg_hysteresis_is_refused(self):
        report = validate_config(
            _config({"emg": {"onset_threshold": 0.1, "offset_threshold": 0.3}})
        )
        assert any("offset_threshold" in i.path for i in report.errors)

    def test_band_above_nyquist_is_refused(self):
        report = validate_config(
            _config({"emg": {"sample_rate_hz": 500.0, "band_high_hz": 400.0}})
        )
        assert any("band_high_hz" in i.path for i in report.errors)

    def test_host_limit_above_the_actuator_is_a_warning_not_an_error(self):
        report = validate_config(
            _config({"control": {"max_velocity": 5.0}, "servo": {"max_velocity": 2.0}})
        )
        assert report.ok, "a misleading limit must not stop the hand from running"
        assert any("max_velocity" in i.path for i in report.warnings)

    def test_bool_does_not_satisfy_a_numeric_rule(self):
        rule = Rule("x.y", float, 0.0, 1.0)
        issue = rule.check(_config({"x": {"y": True}}))
        assert issue is not None and "number" in issue.message

    def test_raise_if_invalid_reports_every_error_at_once(self):
        report = validate_config(
            _config({"servo": {"max_force": 5.0}, "ui": {"theme": "neon"}})
        )
        with pytest.raises(ConfigurationError) as excinfo:
            report.raise_if_invalid()
        assert "max_force" in str(excinfo.value)
        assert "theme" in str(excinfo.value)

    def test_warnings_alone_do_not_block_startup(self):
        report = validate_config(_config({"servo": {"nonsense": 1}}))
        assert report.ok
        report.raise_if_invalid()  # must not raise
        assert report.warnings[0].severity is ValidationSeverity.WARNING


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------


class TestProfiles:
    def test_settings_survive_a_restart(self, tmp_path):
        store = ProfileStore(tmp_path / "profiles")
        profile = store.active()
        assert profile is not None
        profile.set("ui.theme", "high_contrast")
        profile.set("ui.accessibility.font_scale", 1.6)
        store.save(profile)

        reopened = ProfileStore(tmp_path / "profiles")
        assert reopened.overlay() == {
            "ui": {"theme": "high_contrast", "accessibility": {"font_scale": 1.6}}
        }

    def test_switching_profiles_switches_settings(self, tmp_path):
        store = ProfileStore(tmp_path / "profiles")
        default = store.active()
        default.set("ui.theme", "dark")
        store.save(default)

        store.create("alice")
        store.set_active("alice")
        alice = store.active()
        alice.set("ui.theme", "light")
        store.save(alice)

        assert store.overlay()["ui"]["theme"] == "light"
        store.set_active("default")
        assert store.overlay()["ui"]["theme"] == "dark"

    def test_a_profile_cannot_carry_a_safety_limit(self, tmp_path):
        profile = UserProfile(name="test")
        with pytest.raises(ConfigurationError, match="cannot be stored"):
            profile.set("servo.max_force", 1.0)

        # And the store refuses one that got there another way.
        profile.settings["servo.max_force"] = 1.0
        with pytest.raises(ConfigurationError):
            ProfileStore(tmp_path).save(profile)

    def test_profile_names_cannot_escape_the_directory(self):
        for bad in ("../../etc/passwd", "with/slash", "UPPER", "", "x" * 40):
            with pytest.raises(ConfigurationError):
                UserProfile(name=bad)

    def test_an_unreadable_profile_does_not_stop_startup(self, tmp_path):
        store = ProfileStore(tmp_path / "profiles")
        store.create("broken")
        store.set_active("broken")
        (tmp_path / "profiles" / "broken.json").write_text("{ truncated")

        # Falls back rather than raising: a corrupt preferences file must not
        # prevent the hand from working.
        assert store.overlay() == {}

    def test_deleting_the_active_profile_clears_the_selection(self, tmp_path):
        store = ProfileStore(tmp_path / "profiles")
        store.create("temp")
        store.set_active("temp")
        store.delete("temp")
        assert store.active_name() is None


# ---------------------------------------------------------------------------
# Crash recovery
# ---------------------------------------------------------------------------


class TestRunMarker:
    def test_detects_a_run_that_never_finished(self, tmp_path):
        path = tmp_path / "run-state.json"
        first = RunMarker(path, version="1.0")
        assert first.begin().reason is ShutdownReason.UNKNOWN
        first.checkpoint(state="active", mode="ai_assist", moving=True)

        # No finish() — the process died.
        record = RunMarker(path, version="1.0").begin()
        assert record.crashed
        assert record.state == "active"
        assert record.moving
        assert "in motion" in record.describe()

    def test_a_clean_shutdown_is_not_a_crash(self, tmp_path):
        path = tmp_path / "run-state.json"
        marker = RunMarker(path, version="1.0")
        marker.begin()
        marker.finish()
        assert not RunMarker(path).begin().crashed

    def test_a_truncated_marker_counts_as_unclean(self, tmp_path):
        path = tmp_path / "run-state.json"
        path.write_text("{ half written")
        assert RunMarker(path).begin().crashed

    def test_an_unwritable_location_does_not_raise(self, tmp_path):
        blocker = tmp_path / "blocked"
        blocker.write_text("not a directory")
        marker = RunMarker(blocker / "run.json")
        marker.begin()  # must not raise
        marker.checkpoint(state="active")
        marker.finish()


# ---------------------------------------------------------------------------
# Bring-up tools
# ---------------------------------------------------------------------------


class TestBringUpTools:
    def test_link_tester_skips_a_simulated_bus_rather_than_passing_it(
        self, clock: SimulatedClock
    ):
        servo = SimulatedServoBus(clock)
        servo.open()
        report = LinkTester(servo, clock).run()
        assert report.ok
        assert report.results[0].outcome is TestOutcome.SKIP

    def test_range_tester_passes_a_healthy_hand(self, clock: SimulatedClock):
        controller, _ = _hand(clock)
        report = RangeTester(controller, clock).run()
        assert report.ok, report.describe()
        assert len(report.results) == 5

    def test_range_tester_flags_a_finger_that_cannot_travel(self, clock: SimulatedClock):
        # A fouled routing guide: the middle finger jams part-way through its
        # travel. Modelled as contact, because that is what the servo feels.
        from neurogrip.hal.servo.simulated import ContactModel

        controller, servo = _hand(clock)
        servo.set_contact(ContactModel(blocked_at=[1.0, 1.0, 0.30, 1.0, 1.0]))
        report = RangeTester(controller, clock).run()
        middle = next(r for r in report.results if r.name.startswith("Middle"))
        assert middle.outcome is not TestOutcome.PASS
        others = [r for r in report.results if not r.name.startswith("Middle")]
        assert all(r.outcome is TestOutcome.PASS for r in others)

    def test_estop_tester_verifies_the_latch(self, clock: SimulatedClock):
        from neurogrip.core.events import EventBus

        bus = EventBus(clock)
        estop = EmergencyStop(clock)
        watchdogs = WatchdogGroup(clock)
        safety = SafetyMonitor(clock, bus, estop, watchdogs)
        controller, _ = _hand(clock)
        estop.add_listener(lambda r: controller.emergency_stop(r.reason) if r.engaged else None)

        report = EstopTester(controller, safety, clock).run()
        assert report.ok, report.describe()
        names = {r.name: r.outcome for r in report.results}
        assert names["Motion stopped"] is TestOutcome.PASS
        assert names["Drive de-energised"] is TestOutcome.PASS
        assert names["Latch holds"] is TestOutcome.PASS

    def test_estop_tester_fails_when_nothing_is_listening(self, clock: SimulatedClock):
        """The regression guard: before the audit, nothing registered a listener."""
        from neurogrip.core.events import EventBus

        bus = EventBus(clock)
        safety = SafetyMonitor(clock, bus, EmergencyStop(clock), WatchdogGroup(clock))
        controller, _ = _hand(clock)

        report = EstopTester(controller, safety, clock).run()
        assert not report.ok
        assert any(r.name == "Motion stopped" for r in report.failures)


# ---------------------------------------------------------------------------
# Camera calibration
# ---------------------------------------------------------------------------


class TestCameraCalibration:
    def test_recovers_a_known_field_of_view(self):
        import math

        from neurogrip.vision.calibration import CameraCalibrationWizard

        width, true_fov = 640, 62.0
        focal = (width / 2) / math.tan(math.radians(true_fov) / 2)
        wizard = CameraCalibrationWizard(width, 480)
        for distance in (0.20, 0.35, 0.50, 0.70):
            wizard.add_measurement("card", distance, 0.0856 * focal / distance)

        calibration = wizard.solve()
        assert calibration.horizontal_fov_deg == pytest.approx(true_fov, abs=0.1)
        assert calibration.is_trustworthy

    def test_disagreeing_samples_are_reported_as_untrustworthy(self):
        import math

        from neurogrip.vision.calibration import CameraCalibrationWizard

        focal = (640 / 2) / math.tan(math.radians(62.0) / 2)
        wizard = CameraCalibrationWizard(640, 480)
        for distance, actual in ((0.20, 0.20), (0.35, 0.35), (0.50, 0.50), (0.70, 0.55)):
            wizard.add_measurement("card", distance, 0.0856 * focal / actual)

        calibration = wizard.solve()
        assert not calibration.is_trustworthy
        # The residual identifies which measurement was wrong.
        residuals = dict(wizard.residuals(calibration))
        worst = max(residuals, key=lambda k: abs(residuals[k]))
        assert "0.70" in worst

    def test_rejects_a_target_too_small_to_measure(self):
        from neurogrip.vision.calibration import CameraCalibrationWizard

        wizard = CameraCalibrationWizard(640, 480)
        with pytest.raises(CalibrationError, match="move the target closer"):
            wizard.add_measurement("card", 2.0, 12.0)


# ---------------------------------------------------------------------------
# Tendon model
# ---------------------------------------------------------------------------


class TestTendonModel:
    def test_ideal_tendons_behave_exactly_as_before(self, clock: SimulatedClock):
        """The plant change must not perturb any existing tuning."""
        controller, _ = _hand(clock, TendonModel.ideal())
        controller.move_to(HandPose.uniform(0.6), force=0.5, speed=1.0)
        for _ in range(600):
            clock.advance(0.005)
            controller.tick()
        assert controller.state.pose[Finger.INDEX] == pytest.approx(0.6, abs=0.02)

    def test_a_slack_tendon_draws_no_load_current(self, clock: SimulatedClock):
        controller, _ = _hand(clock, TendonModel.uniform(0.4))
        # Command inside the slack region: the servo turns, the finger does not.
        controller.move_to(HandPose.uniform(0.2), force=0.4, speed=0.5)
        for _ in range(400):
            clock.advance(0.005)
            controller.tick()

        state = controller.state
        assert state.pose[Finger.INDEX] == pytest.approx(0.0, abs=0.02)
        assert state.currents[int(Finger.INDEX)] < SimulatedServoBus.HOLDING_CURRENT_MA


# ---------------------------------------------------------------------------
# Vision replay
# ---------------------------------------------------------------------------


class TestVisionReplay:
    def test_recording_round_trips(self, tmp_path):
        from neurogrip.vision.backends.replay import (
            ReplaySettings,
            ReplayVisionBackend,
            VisionRecorder,
        )
        from neurogrip.vision.types import BoundingBox, Detection, VisionResult

        path = tmp_path / "rec.jsonl"
        with VisionRecorder(path, backend="test") as recorder:
            for index in range(5):
                recorder.write(
                    VisionResult(
                        timestamp=index * 0.05,
                        frame_index=index,
                        backend="test",
                        detections=(
                            Detection(
                                label="bottle",
                                confidence=0.9,
                                bbox=BoundingBox(0.3, 0.3, 0.6, 0.7),
                                track_id=1,
                                age=index,
                            ),
                        ),
                    )
                )

        backend = ReplayVisionBackend(ReplaySettings(path=str(path), loop=False))
        backend.initialize()

        class Frame:
            def __init__(self, i):
                self.timestamp = i * 0.1
                self.index = i

        labels = []
        count = 0
        while not backend.exhausted:
            result = backend.process(Frame(count))
            count += 1
            labels.extend(d.label for d in result.detections)
        assert count == 5
        assert labels == ["bottle"] * 5

    def test_replay_restamps_to_the_current_clock(self, tmp_path):
        from neurogrip.vision.backends.replay import (
            ReplaySettings,
            ReplayVisionBackend,
            VisionRecorder,
        )
        from neurogrip.vision.types import VisionResult

        path = tmp_path / "rec.jsonl"
        with VisionRecorder(path) as recorder:
            recorder.write(VisionResult(timestamp=99999.0, backend="old"))

        backend = ReplayVisionBackend(ReplaySettings(path=str(path)))
        backend.initialize()

        class Frame:
            timestamp = 12.0
            index = 3

        # A recorded timestamp would be stale by any measure the stack applies.
        assert backend.process(Frame()).timestamp == 12.0

    def test_a_truncated_recording_keeps_its_good_frames(self, tmp_path):
        from neurogrip.vision.backends.replay import load_recording

        path = tmp_path / "rec.jsonl"
        path.write_text(
            json.dumps({"format_version": 1})
            + "\n"
            + json.dumps({"timestamp": 0.1, "detections": [], "grasps": []})
            + "\n"
            + '{"timestamp": 0.2, "detec'
        )
        assert len(load_recording(path)) == 1

    def test_a_future_format_is_refused(self, tmp_path):
        from neurogrip.core.errors import VisionError
        from neurogrip.vision.backends.replay import load_recording

        path = tmp_path / "rec.jsonl"
        path.write_text(json.dumps({"format_version": 99}) + "\n")
        with pytest.raises(VisionError, match="format v99"):
            load_recording(path)

    def test_the_bundled_reference_recording_loads(self):
        from neurogrip.vision.backends.replay import load_recording

        results = load_recording("data/vision/reference-bottle.jsonl")
        assert len(results) == 120
        assert any(d.label == "bottle" for r in results for d in r.detections)


# ---------------------------------------------------------------------------
# AnyGrasp adapter
# ---------------------------------------------------------------------------


class TestAnyGraspAdapter:
    def _backend(self, grasps):
        from neurogrip.vision.backends.anygrasp import AnyGraspBackend, AnyGraspSettings

        class Model:
            def predict(self, points, colors=None):
                return list(grasps)

        return AnyGraspBackend(AnyGraspSettings(), model=Model())

    def test_the_registry_offers_more_than_one_model(self):
        from neurogrip.ai.grasp import available_planners
        from neurogrip.vision.backend import available_backends

        assert {"hggd", "anygrasp", "heuristic"} <= set(available_planners())
        assert {"hggd_mcu", "anygrasp", "replay"} <= set(available_backends())

    def test_reports_clearly_when_the_runtime_is_missing(self):
        from neurogrip.core.errors import ModelLoadError
        from neurogrip.vision.backend import create_backend

        with pytest.raises(ModelLoadError, match="depth sensor"):
            create_backend("anygrasp", _config({}))

    def test_converts_6dof_poses_and_drops_out_of_range_ones(self):
        from neurogrip.vision.backends.anygrasp import SixDofGrasp

        backend = self._backend(
            [
                SixDofGrasp((0.0, 0.0, 0.30), (0, 0, 1), 0.06, 0.9, "cup"),
                SixDofGrasp((0.0, 0.0, 3.00), (0, 0, 1), 0.06, 0.95),
            ]
        )

        class Frame:
            timestamp = 1.0
            index = 0

            def __init__(self) -> None:
                self.point_cloud = [(0.0, 0.0, 0.3)]

        result = backend.process(Frame())
        assert len(result.grasps) == 1
        assert result.grasps[0].approach_vector == (0, 0, 1)
        assert result.grasps[0].depth_m == pytest.approx(0.30)

    def test_an_rgb_frame_is_not_an_error(self):
        backend = self._backend([])

        class Frame:
            timestamp = 1.0
            index = 0

        result = backend.process(Frame())
        assert result.grasps == ()
        assert "no depth" in result.error

    def _plan_for(self, approach):
        from neurogrip.ai.grasp.anygrasp import AnyGraspPlanner
        from neurogrip.ai.grasp.base import GraspContext
        from neurogrip.ai.objects import AffordanceDatabase
        from neurogrip.control.grips import GripLibrary
        from neurogrip.core.types import IntentKind, ModeId
        from neurogrip.emg.intent import IntentEstimate
        from neurogrip.vision.backends.anygrasp import SixDofGrasp
        from neurogrip.vision.types import VisionResult

        grasps = [SixDofGrasp((0.0, 0.0, 0.30), approach, 0.06, 0.9, "cup")]
        vision = VisionResult(
            timestamp=1.0, backend="anygrasp", grasps=self._backend(grasps)._convert(grasps)
        )
        return AnyGraspPlanner(GripLibrary(), AffordanceDatabase()).plan(
            GraspContext(
                timestamp=1.0,
                intent=IntentEstimate(
                    kind=IntentKind.CLOSE, confidence=0.9, strength=0.7, timestamp=1.0
                ),
                vision=vision,
                current_pose=HandPose.open_hand(),
                mode=ModeId.AI_ASSIST,
            )
        )

    def test_confidence_degrades_with_approach_mismatch(self):
        aligned = self._plan_for((0, 0, 1))
        tilted = self._plan_for((0.5, 0, 0.866))
        assert aligned is not None and tilted is not None
        assert tilted.confidence < aligned.confidence

    def test_declines_a_grasp_the_wrist_cannot_reach(self):
        """No powered wrist: an approach the arm is not making is not executable."""
        assert self._plan_for((1, 0, 0)) is None

    def test_a_planar_candidate_is_not_penalised(self):
        """Backends that report less than AnyGrasp must not be treated as misaligned."""
        from neurogrip.ai.grasp.anygrasp import AnyGraspPlanner
        from neurogrip.ai.objects import AffordanceDatabase
        from neurogrip.control.grips import GripLibrary
        from neurogrip.vision.types import GraspCandidate

        planner = AnyGraspPlanner(GripLibrary(), AffordanceDatabase())
        planar = GraspCandidate(
            center_x=0.5, center_y=0.5, angle=0.0, width=0.2, quality=0.9, approach_vector=None
        )
        assert planner._misalignment_deg(planar) == 0.0


# ---------------------------------------------------------------------------
# Emergency-stop integrity checking
# ---------------------------------------------------------------------------


class TestEstopIntegrity:
    """A stop that has never been checked is an assumption, not a safety system.

    These cover the periodic checker: the cheap rehearsal that proves the
    signalling path, and the gated proof test that proves the hardware one.
    """

    def _rig(self, clock, *, listen=True, **kwargs):
        from neurogrip.core.events import EventBus
        from neurogrip.safety.integrity import EstopSelfCheck

        controller, servo = _hand(clock)
        estop = EmergencyStop(clock)
        if listen:
            estop.add_listener(controller.on_estop_record)
        check = EstopSelfCheck(estop, controller, clock, EventBus(clock), **kwargs)
        return check, estop, controller, servo

    def _run(self, check, controller, clock, seconds, step=0.005):
        for _ in range(int(seconds / step)):
            clock.advance(step)
            check.tick()
            controller.tick()

    # -- the rehearsal ----------------------------------------------------

    def test_a_rehearsal_does_not_touch_the_hand(self, clock: SimulatedClock):
        check, estop, controller, _ = self._rig(clock, proof_enabled=False)
        self._run(check, controller, clock, 1.0)

        assert check.rehearsals >= 1
        assert not controller.state.estop
        assert controller.state.enabled
        assert estop.engage_count == 0, "a rehearsal must never latch the stop"

    def test_a_rehearsal_alone_does_not_claim_full_verification(
        self, clock: SimulatedClock
    ):
        """It proves the record arrives, not that the actuators would stop."""
        from neurogrip.safety.integrity import IntegrityStatus

        check, _, controller, _ = self._rig(clock, proof_enabled=False)
        self._run(check, controller, clock, 1.0)
        assert check.status is IntegrityStatus.CHAIN_OK

    def test_a_missing_listener_is_caught(self, clock: SimulatedClock):
        """The exact defect this whole mechanism exists for."""
        from neurogrip.safety.integrity import IntegrityStatus

        check, _, controller, _ = self._rig(clock, listen=False, proof_enabled=False)
        self._run(check, controller, clock, 1.0)

        assert check.status is IntegrityStatus.FAILED
        assert check.last_failure is not None
        assert "nothing is listening" in check.last_failure.message

    def test_a_listener_that_does_not_reach_the_controller_is_caught(
        self, clock: SimulatedClock
    ):
        """A registration that exists but goes somewhere else is still broken."""
        from neurogrip.safety.integrity import IntegrityStatus

        check, estop, controller, _ = self._rig(clock, listen=False, proof_enabled=False)
        estop.add_listener(lambda record: None)  # plausible, and useless
        self._run(check, controller, clock, 1.0)

        assert check.status is IntegrityStatus.FAILED
        assert "does not reach the motion controller" in check.last_failure.message

    def test_rehearsals_repeat_on_the_configured_interval(self, clock: SimulatedClock):
        check, _, controller, _ = self._rig(
            clock, proof_enabled=False, rehearsal_interval_s=5.0
        )
        self._run(check, controller, clock, 21.0)
        # One immediately, then one per interval.
        assert check.rehearsals == 5

    def test_a_wire_removed_later_is_noticed(self, clock: SimulatedClock):
        """Regressions happen after the first green run, not before it."""
        from neurogrip.safety.integrity import IntegrityStatus

        check, estop, controller, _ = self._rig(
            clock, proof_enabled=False, rehearsal_interval_s=5.0
        )
        self._run(check, controller, clock, 1.0)
        assert check.status is IntegrityStatus.CHAIN_OK

        estop._listeners.clear()  # simulate a refactor dropping the registration
        self._run(check, controller, clock, 6.0)
        assert check.status is IntegrityStatus.FAILED

    # -- the proof test ---------------------------------------------------

    def test_the_proof_test_cuts_drive_and_re_arms(self, clock: SimulatedClock):
        from neurogrip.safety.integrity import IntegrityStatus

        check, _, controller, _ = self._rig(clock, proof_interval_s=60.0)
        self._run(check, controller, clock, 2.0)

        assert check.proofs >= 1
        assert check.failures == 0
        assert check.status is IntegrityStatus.VERIFIED
        # And the hand is usable again afterwards.
        assert controller.state.enabled
        assert not controller.state.estop

    def test_a_stop_that_does_not_de_energise_is_caught(self, clock: SimulatedClock):
        """The failure the proof test exists to find."""
        check, _, controller, servo = self._rig(clock, proof_interval_s=60.0)

        # A firmware regression: the stop is accepted and ignored.
        servo.emergency_stop = lambda: None
        self._run(check, controller, clock, 3.0)

        assert check.last_failure is not None
        assert "not de-energised" in check.last_failure.message

    def test_the_proof_test_waits_for_an_idle_hand(self, clock: SimulatedClock):
        check, _, controller, _ = self._rig(clock, proof_interval_s=0.0)
        controller.move_to(HandPose.closed_hand(), force=0.4, speed=0.2)
        clock.advance(0.005)
        controller.tick()

        # Moving: the check must not interrupt.
        for _ in range(40):
            clock.advance(0.005)
            check.tick()
            controller.tick()
        assert check.proofs == 0
        assert not controller.state.estop

    def test_the_proof_test_never_runs_while_holding(self, clock: SimulatedClock):
        from neurogrip.hal.servo.simulated import ContactModel

        check, _, controller, servo = self._rig(clock, proof_interval_s=0.0)
        servo.set_contact(ContactModel.uniform(0.4))
        controller.move_to(HandPose.closed_hand(), force=0.6, speed=0.8)
        self._run(check, controller, clock, 3.0)

        assert check.proofs == 0, "cutting drive on a held object would drop it"

    def test_the_proof_test_can_be_disabled(self, clock: SimulatedClock):
        check, _, controller, _ = self._rig(clock, proof_enabled=False)
        self._run(check, controller, clock, 5.0)
        assert check.proofs == 0
        assert check.rehearsals >= 1

    def test_a_failure_is_sticky(self, clock: SimulatedClock):
        """A stop that failed once is not trustworthy because it passed later."""
        from neurogrip.safety.integrity import IntegrityStatus

        check, estop, controller, _ = self._rig(
            clock, proof_enabled=False, rehearsal_interval_s=1.0
        )
        listeners = list(estop._listeners)
        estop._listeners.clear()
        self._run(check, controller, clock, 1.0)
        assert check.status is IntegrityStatus.FAILED

        estop._listeners.extend(listeners)
        self._run(check, controller, clock, 5.0)
        assert check.status is IntegrityStatus.FAILED
        assert check.last_result.passed, "later checks still run and still pass"

        check.reset("investigated")
        self._run(check, controller, clock, 2.0)
        assert check.status is IntegrityStatus.CHAIN_OK

    # -- the fault it raises ----------------------------------------------

    def test_a_failure_degrades_to_manual_rather_than_stopping_the_hand(
        self, clock: SimulatedClock
    ):
        """A broken backup must not cost the user their limb."""
        from neurogrip.core.errors import Severity
        from neurogrip.safety.integrity import EstopIntegrityRule
        from neurogrip.safety.rules import SafetyContext

        check, _, controller, _ = self._rig(clock, listen=False, proof_enabled=False)
        self._run(check, controller, clock, 1.0)

        rule = EstopIntegrityRule(check)
        fault = rule.evaluate(SafetyContext(timestamp=clock.monotonic(), hand=controller.state))
        assert fault is not None
        assert fault.severity is Severity.FALLBACK
        assert fault.severity is not Severity.CRITICAL

    def test_no_fault_while_the_stop_is_healthy(self, clock: SimulatedClock):
        from neurogrip.safety.integrity import EstopIntegrityRule
        from neurogrip.safety.rules import SafetyContext

        check, _, controller, _ = self._rig(clock)
        self._run(check, controller, clock, 2.0)
        rule = EstopIntegrityRule(check)
        assert rule.evaluate(
            SafetyContext(timestamp=clock.monotonic(), hand=controller.state)
        ) is None

    # -- announcement ------------------------------------------------------

    def test_a_proof_test_does_not_look_like_a_real_stop(self, clock: SimulatedClock):
        """Routine checks must not bury real incidents."""
        from neurogrip.core.events import EventBus
        from neurogrip.core.topics import Topics
        from neurogrip.safety.integrity import EstopSelfCheck

        bus = EventBus(clock)
        events = []
        bus.subscribe(Topics.ESTOP_ENGAGED, events.append)

        controller, _ = _hand(clock, bus=bus)
        estop = EmergencyStop(clock)
        estop.add_listener(controller.on_estop_record)
        check = EstopSelfCheck(estop, controller, clock, bus, proof_interval_s=60.0)
        self._run(check, controller, clock, 2.0)

        assert events, "the event is still recorded — it is part of what happened"
        assert all(e.payload["diagnostic"] for e in events)

    def test_a_real_stop_is_not_marked_diagnostic(self, clock: SimulatedClock):
        from neurogrip.core.events import EventBus
        from neurogrip.core.topics import Topics

        bus = EventBus(clock)
        events = []
        bus.subscribe(Topics.ESTOP_ENGAGED, events.append)
        servo = SimulatedServoBus(clock)
        controller = HandController(servo, clock, bus)
        controller.start()
        controller.emergency_stop("a real one")

        assert len(events) == 1
        assert not events[0].payload["diagnostic"]

    def test_a_diagnostic_stop_does_not_flush_the_black_box(self, clock, tmp_path):
        """An incident file every few hours would bury the real ones.

        Events are delivered to the recorder on a worker thread, so the handler
        is driven directly here rather than through the bus — the decision under
        test is "does this event count as an incident", and racing a thread to
        observe it would make the test flaky for no added coverage.
        """
        from neurogrip.core.events import Event, EventBus
        from neurogrip.core.topics import Topics
        from neurogrip.telemetry import BlackBoxRecorder

        recorder = BlackBoxRecorder(
            EventBus(clock), clock, directory=str(tmp_path / "bb")
        )
        records = tmp_path / "bb"

        def estop_event(diagnostic: bool) -> Event:
            return Event(
                topic=Topics.ESTOP_ENGAGED,
                payload={"reason": "test", "diagnostic": diagnostic},
                timestamp=clock.monotonic(),
                source="control",
            )

        recorder._on_event(estop_event(diagnostic=True))
        assert not list(records.glob("*.json")), "a routine check is not an incident"

        recorder._on_event(estop_event(diagnostic=False))
        assert list(records.glob("*.json")), "a real stop still writes a record"
