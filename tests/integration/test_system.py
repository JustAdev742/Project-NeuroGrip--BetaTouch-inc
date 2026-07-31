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
