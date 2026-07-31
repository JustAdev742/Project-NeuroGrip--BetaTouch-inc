"""The composition root.

Everything is wired here, in one place, from configuration. Read this file to
understand how the system fits together — no other module constructs its
collaborators, so there is nowhere else for the wiring to hide.

Startup sequence (``docs/safety.md`` explains why this order):

1. build every object; nothing is opened yet;
2. start the non-actuating services (EMG, vision, safety, diagnostics);
3. run the power-on self-test **before the actuators are energised**;
4. home the hand;
5. enable drive and enter the default mode;
6. start the scheduler.

Shutdown is the exact reverse, and always ends with the actuators de-energised.
"""

from __future__ import annotations

import signal
from dataclasses import dataclass

from .. import __version__
from ..ai.grasp import build_default_planner
from ..ai.models import ModelRegistry
from ..ai.objects import AffordanceDatabase
from ..control.controller import HandController
from ..control.grips import GripLibrary
from ..control.kinematics import HandKinematics
from ..control.motion import MotionLimits
from ..core.clock import Clock, RealClock
from ..core.config import Config
from ..core.container import ServiceRegistry
from ..core.errors import Severity
from ..core.events import EventBus
from ..core.lifecycle import HealthReport, HealthStatus
from ..core.logging import get_logger
from ..core.state import SystemState, build_system_state_machine
from ..core.topics import Topics
from ..core.types import ModeId
from ..diagnostics.console import DebugConsole, build_console
from ..diagnostics.metrics import MetricsRegistry
from ..diagnostics.service import DiagnosticsService, build_standard_selftests
from ..emg.calibration import CalibrationWizard, EmgCalibration
from ..emg.gestures import ThresholdSettings, create_classifier
from ..emg.intent import IntentEngine, IntentSettings
from ..emg.pipeline import EmgPipeline, PipelineSettings
from ..emg.recorder import AutoRecalibrator
from ..fusion.fusion import DecisionFusion
from ..fusion.policy import policy_for_mode
from ..hal.factory import HardwareBundle, HardwareFactory
from ..hal.servo.base import ServoLimits
from ..modes.base import ModeContext
from ..modes.manager import ModeManager
from ..modes.profiles import build_modes
from ..safety.estop import EmergencyStop, EstopSource
from ..safety.monitor import SafetyMonitor
from ..safety.rules import SafetyContext
from ..safety.watchdog import WatchdogGroup
from ..telemetry import BlackBoxRecorder, TelemetryWriter
from ..training.achievements import AchievementTracker
from ..training.session import TrainingSession
from ..training.stats import TrainingStats
from ..ui.app import UiService
from ..ui.renderers import create_renderer
from ..ui.theme import Theme
from ..vision.backend import create_backend
from ..vision.depth import MonocularDepthEstimator
from ..vision.pipeline import VisionPipeline
from ..vision.tracking import ObjectTracker
from .scheduler import Scheduler
from .services import EmgService, VisionService

__all__ = ["Application", "build_application"]

log = get_logger(__name__)


@dataclass(slots=True)
class Application:
    """The assembled system."""

    config: Config
    clock: Clock
    bus: EventBus
    metrics: MetricsRegistry
    hardware: HardwareBundle
    controller: HandController
    emg: EmgService
    vision: VisionService | None
    fusion: DecisionFusion
    modes: ModeManager
    safety: SafetyMonitor
    diagnostics: DiagnosticsService
    scheduler: Scheduler
    ui: UiService | None
    training: TrainingSession
    services: ServiceRegistry
    console: DebugConsole | None = None
    blackbox: BlackBoxRecorder | None = None
    telemetry: TelemetryWriter | None = None
    state: SystemState = SystemState.BOOT
    _machine: object = None
    #: Populated at startup with the power-on self-test result.
    startup_report: object = None

    # -- lifecycle ------------------------------------------------------------

    def start(self, *, allow_motion: bool = True) -> bool:
        """Bring the system up. Returns ``False`` if startup was refused."""
        self.bus.publish(Topics.SYSTEM_STARTING, {"version": __version__}, source="application")
        self._machine = build_system_state_machine(self.clock)
        self._machine.fire("boot_complete")
        self.state = SystemState.SELFTEST

        self.services.start_all()

        report = self.diagnostics.selftest.run(allow_motion=False)
        self.startup_report = report
        self.bus.publish(Topics.SELFTEST_RESULT, report, source="application")
        if not report.ok:
            # A failed power-on test must not silently produce a working-looking
            # device. Enter DEGRADED, keep manual control available, and say why.
            log.error("power-on self-test failed", failures=[f.name for f in report.failures])
            self._machine.fire("selftest_degraded")
            self.state = SystemState.DEGRADED
            self.modes.fall_back_to_manual("power-on self-test failed")
            for failure in report.failures:
                self.bus.publish(
                    Topics.UI_NOTIFICATION,
                    {"level": "error", "text": f"{failure.name}: {failure.message}", "sticky": True},
                    source="application",
                )
        else:
            self._machine.fire("selftest_passed")
            self.state = SystemState.HOMING

        if allow_motion and self.safety.state.motion_allowed:
            try:
                self.controller.home()
                self.controller.enable()
            except Exception as exc:
                log.error("homing failed", error=str(exc))
                self.safety.trigger_estop(f"homing failed: {exc}", source=EstopSource.SELFTEST)
                return False

        if self.state is not SystemState.DEGRADED:
            self._machine.fire("homed")
            self.state = SystemState.READY

        self.safety.watchdogs.reset_all()
        self.bus.publish(Topics.SYSTEM_READY, {"state": self.state.value}, source="application")
        log.info("system ready", state=self.state.value, version=__version__)
        return True

    def stop(self) -> None:
        """Safe shutdown, in reverse dependency order."""
        self.bus.publish(Topics.SYSTEM_STOPPING, {}, source="application")
        log.info("shutting down")
        self.scheduler.stop()
        try:
            self.controller.disable()
        except Exception as exc:
            log.warning("could not disable drive", error=str(exc))
        self.services.stop_all()
        if self.blackbox is not None:
            self.blackbox.flush(reason="shutdown")
            self.blackbox.stop()
        if self.telemetry is not None:
            self.telemetry.stop()
        self.state = SystemState.SHUTDOWN
        self.bus.publish(Topics.SYSTEM_STOPPED, {}, source="application")
        log.info("shutdown complete")

    def run(self) -> None:
        """Run until interrupted. Installs signal handlers for a clean exit."""
        self._install_signal_handlers()
        try:
            self.scheduler.run()
        finally:
            self.stop()

    def _install_signal_handlers(self) -> None:
        def handler(signum, _frame):  # pragma: no cover - signal path
            log.warning("signal received; stopping", signal=signum)
            self.scheduler.stop()

        try:
            signal.signal(signal.SIGINT, handler)
            signal.signal(signal.SIGTERM, handler)
        except (ValueError, OSError):  # pragma: no cover - not on the main thread
            pass

    # -- the decision cycle ---------------------------------------------------

    def decision_tick(self) -> None:
        """Safety → mode → fusion. The 100 Hz group.

        Ordering is load-bearing: safety is evaluated *before* the mode runs, so
        the mode's decision is made against a current verdict rather than a stale
        one.
        """
        now = self.clock.monotonic()
        hand = self.controller.state
        emg_frame = self.emg.frame
        intent = self.emg.intent
        vision_result = self.vision.latest if self.vision is not None else None
        system = self.diagnostics.snapshot

        safety_state = self.safety.evaluate(
            SafetyContext(
                timestamp=now,
                hand=hand,
                intent=intent,
                vision=vision_result,
                battery=system.battery if system else None,
                cpu_temperature_c=system.system.cpu_temperature_c if system else 0.0,
                moving=hand.moving,
                expired_watchdogs=self.safety.watchdogs.expired,
            )
        )

        if safety_state.estop_engaged and not hand.estop:
            self.controller.emergency_stop(safety_state.primary_reason)

        decision = self.modes.update(
            ModeContext(
                timestamp=now,
                hand=hand,
                intent=intent,
                emg=emg_frame,
                vision=vision_result,
                safety=safety_state,
            )
        )
        if decision is not None:
            self.bus.publish(Topics.DECISION_MADE, decision, source="application")

        # The hands-free double-pulse gesture cycles modes.
        if intent is not None and intent.kind.value == "toggle":
            self.modes.cycle(safety=safety_state)

        self.safety.watchdogs.kick("decision")

    def control_tick(self) -> None:
        """The 200 Hz group: one actuator cycle."""
        self.controller.tick()
        self.safety.watchdogs.kick("control")

    # -- reporting ------------------------------------------------------------

    def health(self) -> list[HealthReport]:
        return [*self.services.health(), self.scheduler.health()]

    @property
    def overall_health(self) -> HealthStatus:
        return max((r.status for r in self.health()), default=HealthStatus.OK)

    def describe(self) -> dict[str, object]:
        """Summary for ``neurogrip info`` and the System Information screen."""
        return {
            "version": __version__,
            "state": self.state.value,
            "mode": self.modes.current.value if self.modes.current else "none",
            "config_sources": list(self.config.sources),
            **self.hardware.describe(),
            "vision_backend": (
                str(self.vision.pipeline.latest.backend) if self.vision else "none"
            ),
            "notes": list(self.hardware.notes),
        }


def build_application(
    config: Config, clock: Clock | None = None, *, bus: EventBus | None = None
) -> Application:
    """Construct the whole system from configuration."""
    clock = clock or RealClock()
    bus = bus or EventBus(clock)
    metrics = MetricsRegistry(clock)
    services = ServiceRegistry()

    bus.error_reporter = lambda name, exc: log.error(
        "event handler failed", handler=name, error=str(exc)
    )

    # -- hardware -------------------------------------------------------------
    hardware = HardwareFactory(config, clock).build()
    for note in hardware.notes:
        log.info("hardware", detail=note)

    # -- safety ---------------------------------------------------------------
    estop = EmergencyStop(clock)
    watchdogs = WatchdogGroup(clock)
    watchdog_cfg = config.section("safety.watchdogs")
    watchdogs.add(
        "control",
        watchdog_cfg.get_float("control_s", 0.1),
        severity=Severity.CRITICAL,
        detail="The control loop stalled. This is a software fault.",
    )
    watchdogs.add(
        "emg",
        watchdog_cfg.get_float("emg_s", 0.3),
        severity=Severity.FALLBACK,
        detail="No EMG data. Check the electrode leads.",
    )
    watchdogs.add(
        "decision",
        watchdog_cfg.get_float("decision_s", 0.2),
        severity=Severity.FALLBACK,
        detail="The decision loop stalled.",
    )
    watchdogs.add(
        "vision",
        watchdog_cfg.get_float("vision_s", 2.0),
        severity=Severity.DEGRADED,
        detail="No camera frames. Assistance is limited.",
    )
    watchdogs.add(
        "ui", watchdog_cfg.get_float("ui_s", 3.0), severity=Severity.MINOR,
        detail="The interface stopped updating.",
    )
    safety = SafetyMonitor(clock, bus, estop, watchdogs)

    # -- control --------------------------------------------------------------
    grips = GripLibrary.from_config(config)
    kinematics = HandKinematics()
    servo_section = config.section("servo")
    servo_limits = ServoLimits(
        max_velocity=servo_section.get_float("max_velocity", 2.0),
        max_acceleration=servo_section.get_float("max_acceleration", 8.0),
        max_current_ma=servo_section.get_int("max_current_ma", 900),
        max_temperature_c=servo_section.get_float("max_temperature_c", 65.0),
        max_force=servo_section.get_float("max_force", 0.85),
    )
    control_section = config.section("control")
    controller = HandController(
        hardware.servo_bus,
        clock,
        bus,
        grips=grips,
        kinematics=kinematics,
        servo_limits=servo_limits,
        motion_limits=MotionLimits(
            max_velocity=control_section.get_float("max_velocity", 1.8),
            max_acceleration=control_section.get_float("max_acceleration", 7.0),
            max_jerk=control_section.get_float("max_jerk", 40.0),
            position_tolerance=control_section.get_float("position_tolerance", 0.015),
        ),
    )

    # -- EMG ------------------------------------------------------------------
    emg_section = config.section("emg")
    calibration = EmgCalibration()
    calibration_path = emg_section.get_str("calibration_path", "var/calibration.json")
    try:
        calibration = EmgCalibration.load(calibration_path)
        log.info("loaded EMG calibration", path=calibration_path, channels=len(calibration.channels))
    except Exception:
        log.info("no stored calibration; using conservative defaults")

    pipeline = EmgPipeline(
        hardware.emg_source.channels,
        calibration,
        PipelineSettings(
            sample_rate_hz=hardware.emg_source.sample_rate_hz,
            mains_hz=emg_section.get_float("mains_hz", 50.0),
            band_low_hz=emg_section.get_float("band_low_hz", 20.0),
            band_high_hz=emg_section.get_float("band_high_hz", 400.0),
            envelope_attack_s=emg_section.get_float("envelope_attack_s", 0.03),
            envelope_release_s=emg_section.get_float("envelope_release_s", 0.15),
        ),
    )
    classifier = create_classifier(
        emg_section.get_str("classifier", "threshold"),
        settings=ThresholdSettings(
            onset=emg_section.get_float("onset_threshold", 0.22),
            offset=emg_section.get_float("offset_threshold", 0.12),
            co_contraction=emg_section.get_float("co_contraction_threshold", 0.35),
        ),
        model_path=emg_section.get_str("model_path", "") or None,
    )
    intent_engine = IntentEngine(
        classifier,
        clock,
        IntentSettings(
            dwell_s=emg_section.get_float("dwell_s", 0.12),
            cancel_dwell_s=emg_section.get_float("cancel_dwell_s", 0.04),
            release_s=emg_section.get_float("release_s", 0.15),
        ),
    )
    emg_service = EmgService(
        hardware.emg_source,
        pipeline,
        intent_engine,
        clock,
        bus,
        watchdogs,
        wizard=CalibrationWizard(hardware.emg_source.channels, clock),
        recalibrator=AutoRecalibrator(calibration, clock)
        if emg_section.get_bool("auto_recalibrate", True)
        else None,
        calibration_path=calibration_path,
    )

    # -- vision ---------------------------------------------------------------
    vision_service: VisionService | None = None
    vision_pipeline: VisionPipeline | None = None
    if config.get_bool("vision.enabled", True) and hardware.camera is not None:
        backend = create_backend(config.get_str("vision.backend", "hggd_mcu"), config)
        vision_pipeline = VisionPipeline(
            hardware.camera,
            backend,
            clock,
            tracker=ObjectTracker(
                iou_threshold=config.get_float("vision.tracker_iou", 0.3),
                max_missed=config.get_int("vision.tracker_max_missed", 5),
            ),
            depth_estimator=MonocularDepthEstimator(
                horizontal_fov_deg=config.get_float("camera.fov_deg", 62.0),
                image_width=hardware.camera.settings.width,
                image_height=hardware.camera.settings.height,
            ),
            max_result_age=config.get_float("vision.max_result_age_s", 0.5),
        )
        vision_service = VisionService(vision_pipeline, bus, watchdogs)
    else:
        log.info("vision disabled or no camera present; running EMG-only")

    # -- AI -------------------------------------------------------------------
    affordances = AffordanceDatabase.from_config(config)
    planner = build_default_planner(config, grips, affordances, kinematics)
    models = ModelRegistry(config.get_str("ai.model_root", "models"))
    fusion = DecisionFusion(planner, clock, policy=policy_for_mode(ModeId.AI_ASSIST, config))

    # -- training -------------------------------------------------------------
    training = TrainingSession(
        clock,
        bus,
        TrainingStats(config.get_str("training.stats_path", "var/training.json")),
        AchievementTracker(),
    )

    # -- modes ----------------------------------------------------------------
    modes = build_modes(controller, fusion, clock, bus, training_session=training)
    mode_manager = ModeManager(
        modes,
        clock,
        bus,
        intent_engine,
        default_mode=ModeId(config.get_str("modes.default", "ai_assist")),
        min_dwell_s=config.get_float("modes.min_dwell_s", 1.0),
    )

    # -- diagnostics ----------------------------------------------------------
    diagnostics = DiagnosticsService(
        clock, bus, metrics, hardware.system, hardware.power, hardware.connectivity
    )
    build_standard_selftests(
        diagnostics.selftest,
        controller=controller,
        vision=vision_pipeline,
        emg_source=hardware.emg_source,
        system=hardware.system,
        power=hardware.power,
        model_registry=models,
    )

    # -- UI -------------------------------------------------------------------
    ui_service: UiService | None = None
    if config.get_bool("ui.enabled", True):
        renderer = create_renderer(
            config.get_str("ui.renderer", "text"),
            width=config.get_int("ui.width", 800),
            height=config.get_int("ui.height", 480),
            fullscreen=config.get_bool("ui.fullscreen", False),
        )
        ui_service = UiService(
            renderer,
            clock,
            bus,
            Theme.from_config(config),
            controller=controller,
            modes=mode_manager,
            safety=safety,
            vision=vision_pipeline,
            diagnostics=diagnostics,
            training=training,
            calibration=emg_service.wizard,
            software_version=__version__,
        )

    # -- telemetry ------------------------------------------------------------
    blackbox = None
    if config.get_bool("telemetry.blackbox", True):
        blackbox = BlackBoxRecorder(
            bus, clock, directory=config.get_str("telemetry.blackbox_dir", "var/blackbox")
        )
        blackbox.start()

    telemetry = None
    if config.get_bool("telemetry.enabled", False):
        telemetry = TelemetryWriter(
            bus, clock, path=config.get_str("telemetry.path", "var/telemetry/session.jsonl")
        )
        telemetry.start()

    # -- service registry (start order = this order; stop is the reverse) -----
    services.add(safety)
    services.add(emg_service)
    if vision_service is not None:
        services.add(vision_service)
    services.add(controller)
    services.add(diagnostics)
    services.add(mode_manager)
    if ui_service is not None:
        services.add(ui_service)

    for service in services:
        diagnostics.add_service(service)

    # -- scheduler ------------------------------------------------------------
    scheduler = Scheduler(clock)
    scheduler.on_stats = diagnostics.report_loop

    application = Application(
        config=config,
        clock=clock,
        bus=bus,
        metrics=metrics,
        hardware=hardware,
        controller=controller,
        emg=emg_service,
        vision=vision_service,
        fusion=fusion,
        modes=mode_manager,
        safety=safety,
        diagnostics=diagnostics,
        scheduler=scheduler,
        ui=ui_service,
        training=training,
        services=services,
        blackbox=blackbox,
        telemetry=telemetry,
    )

    rates = config.section("runtime")
    scheduler.add("control", rates.get_float("control_hz", 200.0), application.control_tick, priority=100)
    scheduler.add("emg", rates.get_float("emg_hz", 200.0), emg_service.tick, priority=90)
    scheduler.add("decision", rates.get_float("decision_hz", 100.0), application.decision_tick, priority=80)
    if vision_service is not None:
        scheduler.add("vision", rates.get_float("vision_hz", 20.0), vision_service.tick, priority=40)
    if ui_service is not None:
        scheduler.add(
            "ui",
            rates.get_float("ui_hz", 15.0),
            lambda: (ui_service.tick(), watchdogs.kick("ui")),
            priority=20,
        )
    scheduler.add("diagnostics", rates.get_float("diagnostics_hz", 2.0), diagnostics.tick, priority=10)

    application.console = build_console(application)
    if ui_service is not None:
        ui_service.set_device_info(
            tuple(
                (key, str(value))
                for key, value in {
                    **hardware.describe(),
                    "software": __version__,
                    "config": ", ".join(config.sources) or "defaults",
                    "planner": planner.name,
                    "models": ", ".join(models.names) or "none declared",
                }.items()
            )
        )

    log.info(
        "application built",
        version=__version__,
        vision=bool(vision_service),
        ui=bool(ui_service),
        planner=planner.name,
    )
    return application
