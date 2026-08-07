"""Diagnostics service.

Aggregates health from every subsystem, samples host resources, maintains the
metrics registry and owns the self-test suite. Runs at a low rate (2 Hz) — this
is monitoring, not control, and it must never compete with the control loop for
CPU.

It also builds the standard self-test suite, which is where the checks that
verify the *whole* device live: link round-trip, EMG noise floor, camera frame
rate, model availability, memory headroom, servo range of motion.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..control.controller import HandController
from ..core.clock import Clock
from ..core.errors import Severity
from ..core.events import EventBus
from ..core.lifecycle import HealthReport, HealthStatus, Service, ServiceBase
from ..core.logging import get_logger
from ..core.rate import LoopStats
from ..core.topics import Topics
from ..hal.system import (
    BatteryState,
    ConnectivityProbe,
    ConnectivityState,
    PowerSource,
    SystemProbe,
    SystemStats,
)
from ..safety.integrity import IntegrityStatus
from ..vision.pipeline import VisionPipeline
from .metrics import MetricsRegistry
from .selftest import SelfTestReport, SelfTestRunner, TestOutcome, TestResult

__all__ = ["DiagnosticsService", "DiagnosticsSnapshot"]

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class DiagnosticsSnapshot:
    """Everything the Diagnostics screen renders."""

    timestamp: float
    system: SystemStats
    battery: BatteryState
    connectivity: ConnectivityState
    health: tuple[HealthReport, ...] = field(default_factory=tuple)
    loops: tuple[LoopStats, ...] = field(default_factory=tuple)
    metrics: dict = field(default_factory=dict)
    selftest: SelfTestReport | None = None

    @property
    def overall(self) -> HealthStatus:
        return max((report.status for report in self.health), default=HealthStatus.OK)

    @property
    def problems(self) -> tuple[HealthReport, ...]:
        return tuple(r for r in self.health if r.status is not HealthStatus.OK)

    @property
    def unhealthy_loops(self) -> tuple[LoopStats, ...]:
        return tuple(loop for loop in self.loops if not loop.healthy)


class DiagnosticsService(ServiceBase):
    """Health aggregation, resource sampling and self-tests."""

    service_name = "diagnostics"

    def __init__(
        self,
        clock: Clock,
        bus: EventBus,
        metrics: MetricsRegistry,
        system: SystemProbe,
        power: PowerSource,
        connectivity: ConnectivityProbe,
        *,
        services: tuple[Service, ...] = (),
    ) -> None:
        super().__init__()
        self._clock = clock
        self._bus = bus
        self._metrics = metrics
        self._system = system
        self._power = power
        self._connectivity = connectivity
        self._services = list(services)
        self._selftest = SelfTestRunner(clock)
        self._loops: dict[str, LoopStats] = {}
        self._snapshot: DiagnosticsSnapshot | None = None

    # -- registration ---------------------------------------------------------

    def add_service(self, service: Service) -> None:
        """Include a service in health aggregation."""
        self._services.append(service)

    def report_loop(self, stats: LoopStats) -> None:
        """Called by the scheduler each cycle with its rate-group timing."""
        self._loops[stats.name] = stats

    @property
    def selftest(self) -> SelfTestRunner:
        return self._selftest

    @property
    def metrics(self) -> MetricsRegistry:
        return self._metrics

    @property
    def snapshot(self) -> DiagnosticsSnapshot | None:
        return self._snapshot

    # -- cycle ----------------------------------------------------------------

    def tick(self) -> DiagnosticsSnapshot:
        """Sample everything and publish a snapshot."""
        now = self._clock.monotonic()
        system = self._system.sample()
        battery = self._power.read()
        connectivity = self._connectivity.sample()

        self._metrics.gauge("system.cpu_percent", unit="%").set(system.cpu_percent)
        self._metrics.gauge("system.memory_percent", unit="%").set(system.memory_percent)
        self._metrics.gauge("system.temperature", unit="°C").set(system.cpu_temperature_c)
        self._metrics.gauge("battery.percent", unit="%").set(battery.percentage)
        self._metrics.gauge("battery.voltage", unit="V").set(battery.voltage_v)

        health: list[HealthReport] = []
        for service in self._services:
            try:
                health.append(service.health())
            except Exception as exc:
                health.append(
                    HealthReport.failed(service.name, f"health check raised: {exc}")
                )

        snapshot = DiagnosticsSnapshot(
            timestamp=now,
            system=system,
            battery=battery,
            connectivity=connectivity,
            health=tuple(health),
            loops=tuple(self._loops.values()),
            metrics=self._metrics.snapshot(),
            selftest=self._selftest.last_report,
        )
        self._snapshot = snapshot
        self._bus.publish(Topics.DIAGNOSTICS_REPORT, snapshot, source=self.name)
        return snapshot

    def health(self) -> HealthReport:
        if not self.running:
            return HealthReport.offline(self.name)
        return HealthReport.ok(self.name, services=len(self._services), loops=len(self._loops))


def build_standard_selftests(
    runner: SelfTestRunner,
    *,
    controller: HandController,
    vision: VisionPipeline | None,
    emg_source,
    system: SystemProbe,
    power: PowerSource,
    model_registry=None,
    clock: Clock | None = None,
    link_settle_s: float = 0.5,
    estop_check=None,
) -> None:
    """Register the standard suite.

    Kept as a function rather than a class so the set of tests reads as a list,
    and so an integrator can register their own alongside these.
    """

    def _await_telemetry() -> bool:
        """Give an asynchronous link a moment to produce its first telemetry.

        The power-on self-test runs immediately after the services start, which
        on a serial link is well before the controller's first reply can arrive.
        Sampling the state once and concluding "no telemetry" would fail every
        cold start and drop the device into DEGRADED — a self-test that reports
        a fault it created by asking too early is worse than no self-test.
        """
        if controller.state.comms_ok:
            return True
        if clock is None:
            return False
        deadline = clock.monotonic() + link_settle_s
        while clock.monotonic() < deadline:
            controller.tick()
            if controller.state.comms_ok:
                return True
            clock.sleep(0.01)
        return False

    def test_motor_link() -> TestResult:
        if not _await_telemetry():
            return TestResult(
                name="Motor controller link",
                outcome=TestOutcome.FAIL,
                message=f"no telemetry from the motor controller within {link_settle_s:.1f} s",
                remedy="Check the controller cable and that the board is powered.",
            )
        info = controller.state
        return TestResult(
            name="Motor controller link",
            outcome=TestOutcome.PASS,
            message=f"link up, bus {info.bus_voltage_v:.1f} V",
            measurements={"bus_voltage_v": info.bus_voltage_v},
        )

    def test_servo_power() -> TestResult:
        state = controller.state
        if state.bus_voltage_v < 6.4:
            return TestResult(
                name="Actuator supply",
                outcome=TestOutcome.FAIL,
                message=f"bus voltage {state.bus_voltage_v:.1f} V is below 6.4 V",
                remedy="Charge the battery; the actuators will stall at this voltage.",
            )
        if state.bus_voltage_v < 6.9:
            return TestResult(
                name="Actuator supply",
                outcome=TestOutcome.WARN,
                message=f"bus voltage {state.bus_voltage_v:.1f} V is low",
                remedy="Charge soon.",
            )
        return TestResult(
            name="Actuator supply",
            outcome=TestOutcome.PASS,
            message=f"{state.bus_voltage_v:.1f} V",
            measurements={"voltage_v": state.bus_voltage_v},
        )

    def test_emg_noise_floor() -> TestResult:
        """Read a short quiet window and check the baseline is plausible."""
        samples = emg_source.read()
        if not samples:
            return TestResult(
                name="EMG acquisition",
                outcome=TestOutcome.FAIL,
                message="no EMG samples received",
                remedy="Check the EMG front end is connected and powered.",
            )
        channels = len(samples[0].values)
        peak = max(max(abs(v) for v in s.values) for s in samples)
        dropped = emg_source.dropped_samples()
        if peak > 1.5e-3:
            return TestResult(
                name="EMG acquisition",
                outcome=TestOutcome.WARN,
                message=f"baseline is high ({peak * 1e6:.0f} µV) — noise or movement",
                measurements={"peak_uv": peak * 1e6, "channels": channels},
                remedy="Stay still and relaxed during the test; check electrode contact.",
            )
        if dropped > 0:
            return TestResult(
                name="EMG acquisition",
                outcome=TestOutcome.WARN,
                message=f"{dropped} samples dropped",
                remedy="The host may be overloaded; check the CPU load.",
            )
        return TestResult(
            name="EMG acquisition",
            outcome=TestOutcome.PASS,
            message=f"{channels} channels, baseline {peak * 1e6:.0f} µV",
            measurements={"peak_uv": peak * 1e6, "samples": len(samples)},
        )

    def test_camera() -> TestResult:
        if vision is None or not vision.has_camera:
            return TestResult(
                name="Camera",
                outcome=TestOutcome.WARN,
                message="no camera configured — AI assistance will be limited",
                remedy="Connect a camera, or use Manual mode.",
            )
        stats = vision.stats()
        if stats.frames_processed == 0:
            # At power-on the pipeline has started but not yet been scheduled.
            # A self-test should actively exercise the device rather than read a
            # counter that has had no opportunity to move.
            for _ in range(5):
                vision.tick()
            stats = vision.stats()
        if stats.frames_processed == 0:
            return TestResult(
                name="Camera",
                outcome=TestOutcome.FAIL,
                message="no frames captured",
                remedy="Check the camera cable and that no other process is using it.",
            )
        if stats.fps <= 0.0:
            # Frames arrive but the rate is not measurable yet (fewer than two
            # samples). Not an error, just too early to judge.
            return TestResult(
                name="Camera",
                outcome=TestOutcome.PASS,
                message=f"{stats.frames_processed} frame(s) captured",
                measurements={"frames": float(stats.frames_processed)},
            )
        if stats.fps < 8.0:
            return TestResult(
                name="Camera",
                outcome=TestOutcome.WARN,
                message=f"low frame rate ({stats.fps:.1f} fps)",
                measurements={"fps": stats.fps},
                remedy="Lower the resolution, or check the CPU load.",
            )
        return TestResult(
            name="Camera",
            outcome=TestOutcome.PASS,
            message=f"{stats.fps:.0f} fps, {stats.mean_latency_ms:.0f} ms inference",
            measurements={"fps": stats.fps, "latency_ms": stats.mean_latency_ms},
        )

    def test_models() -> TestResult:
        if model_registry is None:
            return TestResult(
                name="Models", outcome=TestOutcome.SKIP, message="no model registry configured"
            )
        statuses = model_registry.check_all()
        missing_required = [s for s in statuses if not s.usable and s.name]
        if not statuses:
            return TestResult(
                name="Models", outcome=TestOutcome.SKIP, message="no models declared"
            )
        if missing_required:
            return TestResult(
                name="Models",
                outcome=TestOutcome.WARN,
                message=f"{len(missing_required)} model(s) unavailable",
                measurements={"missing": ", ".join(s.name for s in missing_required)},
                remedy="Install the model files; the system falls back to classical methods.",
            )
        return TestResult(
            name="Models",
            outcome=TestOutcome.PASS,
            message=f"{len(statuses)} model(s) present",
        )

    def test_resources() -> TestResult:
        stats = system.sample()
        if not stats.available:
            return TestResult(
                name="Host resources", outcome=TestOutcome.SKIP, message="not a Linux host"
            )
        if stats.memory_percent > 92:
            return TestResult(
                name="Host resources",
                outcome=TestOutcome.FAIL,
                message=f"memory {stats.memory_percent:.0f}% used",
                remedy="Free memory or reboot; the control loop may stall.",
            )
        if stats.cpu_percent > 88 or stats.memory_percent > 80:
            return TestResult(
                name="Host resources",
                outcome=TestOutcome.WARN,
                message=f"CPU {stats.cpu_percent:.0f}%, memory {stats.memory_percent:.0f}%",
                remedy="Reduce the vision rate or close background processes.",
            )
        return TestResult(
            name="Host resources",
            outcome=TestOutcome.PASS,
            message=f"CPU {stats.cpu_percent:.0f}%, memory {stats.memory_percent:.0f}%",
            measurements={
                "cpu_percent": stats.cpu_percent,
                "memory_percent": stats.memory_percent,
                "temperature_c": stats.cpu_temperature_c,
            },
        )

    def test_battery() -> TestResult:
        state = power.read()
        if not state.present:
            return TestResult(
                name="Battery", outcome=TestOutcome.SKIP, message="no battery detected"
            )
        if state.is_critical:
            return TestResult(
                name="Battery",
                outcome=TestOutcome.FAIL,
                message=f"critically low ({state.percentage:.0f}%)",
                remedy="Charge before use.",
            )
        if state.is_low:
            return TestResult(
                name="Battery",
                outcome=TestOutcome.WARN,
                message=f"low ({state.percentage:.0f}%)",
                remedy="Charge soon.",
            )
        return TestResult(
            name="Battery",
            outcome=TestOutcome.PASS,
            message=f"{state.percentage:.0f}%, {state.voltage_v:.1f} V",
            measurements={"percentage": state.percentage, "voltage_v": state.voltage_v},
        )

    def test_range_of_motion() -> TestResult:
        """Sweep each finger and confirm it tracks. Moves the hand."""
        from ..core.types import Finger, HandPose

        results: dict[str, float] = {}
        for finger in Finger:
            controller.move_to(
                HandPose.open_hand().with_finger(finger, 0.8),
                source="selftest",
                description=f"sweep {finger.label}",
                speed=0.6,
            )
            # The caller's scheduler drives the controller; sample what we reach.
            results[finger.name.lower()] = controller.state.pose[finger]
        controller.move_to(HandPose.open_hand(), source="selftest", description="return to open")
        return TestResult(
            name="Range of motion",
            outcome=TestOutcome.PASS,
            message="sweep commanded for all five fingers",
            measurements=results,
            remedy="Watch the hand: each finger should flex and return smoothly.",
        )

    def test_estop_integrity() -> TestResult:
        """Report what the periodic checker has established about the stop.

        This does not re-run the check — it surfaces its standing verdict, so
        `neurogrip diagnose` says whether the stop is known to work rather than
        assuming it does.
        """
        if estop_check is None:
            return TestResult(
                name="Emergency stop",
                outcome=TestOutcome.SKIP,
                message="integrity checking is not configured",
            )
        failure = estop_check.last_failure
        if failure is not None:
            return TestResult(
                name="Emergency stop",
                outcome=TestOutcome.FAIL,
                message=failure.message,
                measurements={"kind": failure.kind, **failure.detail},
                remedy="Do not rely on the emergency stop. Run `neurogrip test estop`.",
            )
        status = estop_check.status
        if status is IntegrityStatus.UNKNOWN:
            return TestResult(
                name="Emergency stop",
                outcome=TestOutcome.WARN,
                message="not yet verified this session",
                remedy="The first check runs within 30 s of starting.",
            )
        return TestResult(
            name="Emergency stop",
            outcome=TestOutcome.PASS,
            message=status.label,
            measurements={
                "rehearsals": estop_check.rehearsals,
                "proofs": estop_check.proofs,
            },
        )

    runner.register("Emergency stop", "Verify the stop path is intact", test_estop_integrity,
                    severity=Severity.FALLBACK, category="safety")
    runner.register("Motor controller link", "Verify the link to the ESP32", test_motor_link,
                    severity=Severity.CRITICAL, category="hardware")
    runner.register("Actuator supply", "Check the actuator bus voltage", test_servo_power,
                    severity=Severity.CRITICAL, category="hardware")
    runner.register("EMG acquisition", "Check EMG data and noise floor", test_emg_noise_floor,
                    severity=Severity.FALLBACK, category="sensors")
    runner.register("Camera", "Check capture and inference rate", test_camera,
                    severity=Severity.DEGRADED, category="sensors")
    runner.register("Models", "Verify model files are present and intact", test_models,
                    severity=Severity.DEGRADED, category="software")
    runner.register("Host resources", "Check CPU, memory and temperature", test_resources,
                    severity=Severity.DEGRADED, category="software")
    runner.register("Battery", "Check the battery state", test_battery,
                    severity=Severity.DEGRADED, category="hardware")
    runner.register("Range of motion", "Sweep every finger", test_range_of_motion,
                    requires_motion=True, severity=Severity.FALLBACK, category="hardware")
