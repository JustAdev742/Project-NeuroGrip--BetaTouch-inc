"""End-to-end tests against the fully assembled system.

Everything above the HAL is production code: the real EMG filters, the real
vision pipeline, the real fusion gates, the real ESP32 driver speaking the real
wire protocol to an in-process firmware emulator, and the real safety rules.

The tests in :class:`TestSharedControlInvariants` are the ones that matter most.
They assert the properties the whole architecture exists to guarantee.
"""

from __future__ import annotations

import pytest

from neurogrip.core.clock import SimulatedClock
from neurogrip.core.topics import Topics
from neurogrip.core.types import HandPose, ModeId
from neurogrip.runtime.application import Application, build_application
from neurogrip.runtime.bootstrap import load_configuration
from neurogrip.simulation import ScenarioRunner, SimulatedWorld, build_scenario

pytestmark = pytest.mark.integration


@pytest.fixture
def application(tmp_path):
    """A fully wired system on simulated hardware and a simulated clock."""
    config = load_configuration(profile="simulation").with_overlay(
        {
            "ui": {"renderer": "null"},
            "telemetry": {"blackbox": False},
            "training": {"stats_path": str(tmp_path / "training.json")},
            "emg": {"calibration_path": str(tmp_path / "calibration.json")},
        },
        source="test",
    )
    clock = SimulatedClock()
    app = build_application(config, clock)
    app.start(allow_motion=True)
    yield app
    app.stop()


def _run(app: Application, seconds: float, step: float = 0.005) -> None:
    """Advance the whole system by ``seconds`` of simulated time."""
    for _ in range(int(seconds / step)):
        app.scheduler.step()
        app.clock.advance(step)


def _world(app: Application) -> SimulatedWorld:
    return SimulatedWorld.from_application(app)


class TestStartup:
    def test_reaches_a_running_state(self, application):
        assert application.state.value in ("ready", "active", "degraded")
        assert application.modes.current is not None

    def test_the_hand_is_homed_and_enabled(self, application):
        _run(application, 0.5)
        state = application.controller.state
        assert state.comms_ok
        assert state.enabled

    def test_every_rate_group_is_scheduled(self, application):
        names = {group.name for group in application.scheduler.groups}
        assert {"control", "emg", "decision", "diagnostics"} <= names

    def test_all_services_report_health(self, application):
        _run(application, 1.0)
        reports = application.health()
        assert reports
        assert all(report.name for report in reports)

    def test_the_power_on_selftest_ran(self, application):
        assert application.startup_report is not None
        assert application.startup_report.results


class TestSharedControlInvariants:
    """The properties the architecture exists to guarantee."""

    def test_a_visible_object_alone_never_moves_the_hand(self, application):
        world = _world(application)
        world.place_object("bottle", width_m=0.068)
        world.relax()
        _run(application, 4.0)

        # The camera definitely saw it…
        assert application.vision.latest.primary is not None
        # …and the hand definitely did not move.
        assert application.controller.state.pose.max_difference(HandPose.open_hand()) < 0.05

    def test_intent_plus_vision_produces_an_assisted_grasp(self, application):
        world = _world(application)
        world.place_object("bottle", width_m=0.068)
        _run(application, 1.5)
        world.contract(0.8)
        _run(application, 4.0)

        decision = application.modes.active.last_decision
        assert decision is not None
        assert decision.plan is not None
        assert application.controller.state.holding

    def test_the_user_can_cancel_mid_grasp(self, application):
        world = _world(application)
        world.place_object("bottle", width_m=0.068)
        _run(application, 1.0)
        world.contract(0.8)
        _run(application, 0.8)

        world.co_contract(0.7)
        _run(application, 0.6)
        assert application.fusion.cancels > 0

        world.relax()
        _run(application, 1.0)
        settled = application.controller.state.pose
        _run(application, 1.0)
        assert application.controller.state.pose.max_difference(settled) < 0.05

    def test_the_hand_still_works_with_no_recognisable_object(self, application):
        world = _world(application)
        world.clear_object()
        _run(application, 1.0)
        world.contract(0.85)
        _run(application, 4.0)
        assert max(application.controller.state.pose) > 0.3

    def test_force_is_limited_for_a_fragile_object(self, application):
        world = _world(application)
        world.place_object("fruit", width_m=0.072, shape="sphere", stiffness=0.4)
        _run(application, 1.5)
        world.contract(0.95)  # the user squeezes hard
        _run(application, 4.0)
        # The object's affordance caps the force regardless of user effort.
        assert application.controller.state.force <= 0.5

    def test_manual_mode_never_uses_the_ai(self, application):
        application.modes.activate(ModeId.MANUAL, force=True)
        world = _world(application)
        world.place_object("bottle", width_m=0.068)
        _run(application, 1.0)
        world.contract(0.8)
        _run(application, 3.0)

        decision = application.modes.active.last_decision
        assert decision is not None
        assert decision.plan is None
        assert not decision.ai_contributed
        # …but the hand still moves, under direct control.
        assert max(application.controller.state.pose) > 0.2


class TestFaultHandling:
    def test_a_detached_electrode_stops_the_hand_acting_on_noise(self, application):
        world = _world(application)
        world.place_object("bottle", width_m=0.068)
        _run(application, 1.0)
        world.detach_electrode(0)
        world.detach_electrode(1)
        world.contract(0.9)  # meaningless: the electrodes are off
        _run(application, 4.0)

        assert application.controller.state.pose.max_difference(HandPose.open_hand()) < 0.15

    def test_losing_the_camera_degrades_assistance_not_the_hand(self, application):
        world = _world(application)
        world.clear_object()
        _run(application, 2.5)

        world.contract(0.8)
        _run(application, 3.0)
        assert max(application.controller.state.pose) > 0.25

    def test_an_emergency_stop_halts_and_latches(self, application):
        world = _world(application)
        world.place_object("bottle", width_m=0.068)
        _run(application, 1.0)
        world.contract(0.8)
        _run(application, 0.6)

        closing = max(application.controller.state.pose)
        application.safety.trigger_estop("test", source="user:test")
        _run(application, 0.3)

        world.contract(1.0)  # the user keeps contracting; it must have no effect
        _run(application, 2.0)

        assert application.controller.state.estop
        # The invariant is that the hand is no longer *driven*: it must not close
        # any further. It does relax open, because an e-stop de-energises the
        # servos and the tendon return springs then extend the fingers — that is
        # what the mechanism does when power is removed, and it is why *cancel*
        # (which holds position) exists as a separate, gentler abort.
        assert max(application.controller.state.pose) <= closing + 0.02

    def test_estop_recovery_requires_an_explicit_acknowledgement(self, application):
        application.safety.trigger_estop("test", source="user:test")
        _run(application, 0.5)
        assert application.safety.state.estop_engaged

        assert application.safety.acknowledge("user:test")
        application.controller.clear_emergency_stop()
        _run(application, 1.0)
        assert not application.controller.state.estop

    def test_a_stalled_control_loop_is_detected(self, application):
        _run(application, 0.5)
        # Simulate the loop stalling: time advances but nothing is scheduled.
        application.clock.advance(1.0)
        application.safety.watchdogs.check_all()
        assert "control" in application.safety.watchdogs.expired


class TestScenarios:
    """The bundled scenarios, run as tests so they cannot silently rot."""

    @pytest.mark.parametrize(
        "name",
        ["grasp-bottle", "no-intent-no-motion", "user-cancel", "vision-lost", "fragile-object"],
    )
    def test_scenario(self, application, name):
        runner = ScenarioRunner(application, application.clock)
        result = runner.run(build_scenario(name))
        assert result.passed, result.report()


class TestRuntimeBehaviour:
    def test_the_control_loop_actually_runs_at_its_configured_rate(self, application):
        _run(application, 2.0)
        stats = {loop.name: loop for loop in application.scheduler.stats()}
        control = stats["control"]
        assert control.iterations > 300
        assert control.actual_hz > control.target_hz * 0.8

    def test_events_are_published_for_the_black_box(self, application):
        world = _world(application)
        world.place_object("bottle", width_m=0.068)
        _run(application, 1.0)
        world.contract(0.8)
        _run(application, 2.0)

        topics = {event.topic for event in application.bus.history(limit=500)}
        assert Topics.HAND_STATE in topics
        assert Topics.EMG_FRAME in topics
        assert Topics.DECISION_MADE in topics

    def test_the_ui_renders_every_frame_without_raising(self, application):
        _run(application, 1.0)
        if application.ui is not None:
            scene = application.ui.tick()
            assert scene.title

    def test_diagnostics_collects_a_snapshot(self, application):
        _run(application, 1.5)
        snapshot = application.diagnostics.snapshot
        assert snapshot is not None
        assert snapshot.health
        assert snapshot.loops

    def test_the_debug_console_reports_live_state(self, application):
        _run(application, 1.0)
        result = application.console.execute("status")
        assert result.ok
        assert "mode" in result.text

    def test_describe_summarises_the_build(self, application):
        described = application.describe()
        assert described["version"]
        assert "servo_bus" in described

    def test_shutdown_leaves_the_actuators_disabled(self, tmp_path):
        config = load_configuration(profile="simulation").with_overlay(
            {"ui": {"renderer": "null"}, "telemetry": {"blackbox": False}}, source="test"
        )
        clock = SimulatedClock()
        app = build_application(config, clock)
        app.start(allow_motion=True)
        _run(app, 0.5)
        app.stop()
        assert not app.controller.running


class TestCalibrationEndToEnd:
    def test_the_wizard_runs_through_the_live_pipeline(self, application, tmp_path):
        from neurogrip.emg.calibration import CalibrationPhase

        world = _world(application)
        service = application.emg
        service.start_calibration()

        drives = {
            CalibrationPhase.REST: (0.0, 0.0),
            CalibrationPhase.FLEXOR_MAX: (0.9, 0.0),
            CalibrationPhase.EXTENSOR_MAX: (0.0, 0.9),
            CalibrationPhase.CO_CONTRACTION: (0.7, 0.7),
        }
        for _ in range(6000):
            phase = service.wizard.phase
            if service.wizard.progress().finished:
                break
            flexor, extensor = drives.get(phase, (0.0, 0.0))
            world.emg.set_flexor(flexor)
            world.emg.set_extensor(extensor)
            application.scheduler.step()
            application.clock.advance(0.005)

        assert service.wizard.phase is CalibrationPhase.COMPLETE
        result = service.wizard.result
        assert result is not None and result.is_valid

    def test_intent_is_suppressed_while_calibrating(self, application):
        """The user is following instructions, not driving the hand."""
        world = _world(application)
        application.emg.start_calibration()
        world.contract(0.9)
        _run(application, 1.5)
        assert application.controller.state.pose.max_difference(HandPose.open_hand()) < 0.05


class TestAuditFixesInTheAssembledSystem:
    """The audit's findings, asserted against the wired system rather than in isolation.

    Every one of these covers a mechanism that existed in a module and was
    connected to nothing. A unit test of the mechanism would have passed
    throughout; only assembling the system shows the wire is missing.
    """

    def test_emergency_stop_reaches_the_actuators_without_the_decision_loop(
        self, application
    ):
        """`EmergencyStop.add_listener` existed and nothing registered.

        Before the fix, a software stop reached the controller only via
        `decision_tick`. Here the decision group never runs, so the only path
        that can work is the listener.
        """
        world = _world(application)
        world.contract(0.9)
        _run(application, 0.5)
        assert application.controller.state.moving

        application.safety.trigger_estop("test", source="user:ui")
        # Deliberately tick only the control group.
        for _ in range(40):
            application.controller.tick()
            application.clock.advance(0.005)

        assert application.controller.state.estop
        assert not application.controller.state.enabled

    def test_servo_calibration_is_pushed_to_the_controller_at_startup(self, application):
        """The calibration path was dead end to end before the audit."""
        from neurogrip.core.types import Finger

        pushed = application.controller._calibration
        assert len(pushed) == len(Finger), "every finger must be calibrated at start"

    def test_calibration_owns_the_hand_while_it_runs(self, application):
        """Two writers would make the measured slack meaningless."""
        from neurogrip.core.types import Finger

        world = _world(application)
        wizard = application.servo_calibration
        assert wizard is not None
        # Let telemetry arrive so the wizard's precondition check sees the drive.
        _run(application, 0.2)
        wizard.start((Finger.INDEX,))

        # A user contraction during calibration must not command the hand.
        world.contract(0.95)
        for _ in range(200):
            application.scheduler.step()
            application.clock.advance(0.005)

        assert wizard.active
        # Every motion command in flight came from the wizard, not from a mode.
        active = application.controller.queue.active
        assert active is None or active.source == "servo-calibration"
        wizard.cancel("test finished")

    def test_an_invalid_configuration_refuses_to_build(self, tmp_path):
        from neurogrip.core.errors import ConfigurationError

        config = load_configuration(profile="simulation").with_overlay(
            {"servo": {"max_force": 4.0}}, source="test"
        )
        with pytest.raises(ConfigurationError, match="max_force"):
            build_application(config, SimulatedClock())

    def test_a_crash_starts_the_next_run_in_manual(self, tmp_path):
        """Recovery must be more conservative, never less."""
        from neurogrip.core.runstate import RunMarker

        state_path = tmp_path / "run-state.json"
        stale = RunMarker(state_path, version="test")
        stale.begin()
        stale.checkpoint(state="active", mode="ai_assist", moving=True)

        config = load_configuration(profile="simulation").with_overlay(
            {
                "ui": {"renderer": "null"},
                "telemetry": {"blackbox": False},
                "modes": {"default": "ai_assist"},
                "system": {"state_path": str(state_path)},
                "training": {"stats_path": str(tmp_path / "training.json")},
                "emg": {"calibration_path": str(tmp_path / "calibration.json")},
            },
            source="test",
        )
        app = build_application(config, SimulatedClock())
        try:
            app.start(allow_motion=True)
            assert app.recovered_from_crash
            assert app.modes.current is ModeId.MANUAL
            assert not app.modes.current.ai_enabled
        finally:
            app.stop()

    def test_a_clean_shutdown_leaves_no_crash_flag(self, tmp_path):
        from neurogrip.core.runstate import RunMarker

        state_path = tmp_path / "run-state.json"
        config = load_configuration(profile="simulation").with_overlay(
            {
                "ui": {"renderer": "null"},
                "telemetry": {"blackbox": False},
                "system": {"state_path": str(state_path)},
                "training": {"stats_path": str(tmp_path / "training.json")},
                "emg": {"calibration_path": str(tmp_path / "calibration.json")},
            },
            source="test",
        )
        app = build_application(config, SimulatedClock())
        app.start(allow_motion=True)
        _run(app, 1.0)
        app.stop()

        assert not RunMarker(state_path).begin().crashed

    def test_user_preferences_are_reloaded_on_the_next_start(self, tmp_path):
        """The write path that `var/user.toml` never had."""
        from neurogrip.core.profiles import ProfileStore

        store = ProfileStore(tmp_path / "profiles")
        profile = store.active()
        profile.set("ui.theme", "high_contrast")
        profile.set("ui.accessibility.font_scale", 1.5)
        store.save(profile)

        config = load_configuration(profile="simulation").with_overlay(
            store.overlay(), source="profile"
        )
        assert config.get_str("ui.theme") == "high_contrast"
        assert config.get_float("ui.accessibility.font_scale") == pytest.approx(1.5)

    def test_replayed_perception_drives_the_real_pipeline(self, tmp_path):
        """A recording must exercise fusion exactly as a live backend would."""
        config = load_configuration(profile="simulation").with_overlay(
            {
                "ui": {"renderer": "null"},
                "telemetry": {"blackbox": False},
                "vision": {
                    "backend": "replay",
                    "replay": {"path": "data/vision/reference-bottle.jsonl", "loop": True},
                },
                "training": {"stats_path": str(tmp_path / "training.json")},
                "emg": {"calibration_path": str(tmp_path / "calibration.json")},
            },
            source="test",
        )
        app = build_application(config, SimulatedClock())
        try:
            app.start(allow_motion=True)
            _run(app, 2.0)
            assert app.vision is not None
            latest = app.vision.latest
            assert latest is not None
            assert latest.backend == "replay"
            assert any(d.label == "bottle" for d in latest.detections)
        finally:
            app.stop()

    def test_the_emergency_stop_is_verified_while_the_system_runs(self, application):
        """The stop path is checked continuously, not only when someone asks."""
        from neurogrip.safety.integrity import IntegrityStatus

        check = application.estop_check
        assert check is not None

        _run(application, 2.0)
        assert check.rehearsals >= 1
        assert check.failures == 0
        assert check.status is not IntegrityStatus.FAILED
        # And the routine verification left the hand exactly as it found it.
        assert application.controller.state.enabled
        assert not application.controller.state.estop

    def test_losing_the_estop_wiring_disables_ai_assistance(self, application):
        """A broken backup costs assistance, not the user's hand."""
        check = application.estop_check
        assert check is not None
        _run(application, 1.0)

        # Simulate a refactor dropping the listener registration.
        application.safety.estop._listeners.clear()
        _run(application, 40.0)

        assert check.status.value == "failed"
        assert not application.safety.state.ai_allowed
        # Direct control survives: the hand is still the user's to move.
        assert application.safety.state.motion_allowed

    def test_the_trigger_sources_are_audited_in_the_running_system(self, application):
        """A stop nothing can trigger fails as quietly as one that does nothing."""
        check = application.estop_check
        assert check is not None and check.triggers is not None
        _run(application, 2.0)

        assert check.triggers.static_problems() == ()
        assert check.failures == 0
        # The probe's expiry must not surface as a fault to the user.
        codes = {f.code for f in application.safety.state.faults}
        assert not any("estop-probe" in code for code in codes)

    def test_disabling_a_critical_rule_disables_ai_assistance(self, application):
        """A rule that can no longer stop the hand is a broken trigger source."""
        check = application.estop_check
        _run(application, 1.0)

        for rule in application.safety.rules:
            if rule.name == "communication":
                rule.set_enabled(False)
        _run(application, 2.0)

        assert check.status.value == "failed"
        assert not application.safety.state.ai_allowed
        assert application.safety.state.motion_allowed
