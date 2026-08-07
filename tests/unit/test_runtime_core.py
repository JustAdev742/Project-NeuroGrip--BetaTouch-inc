"""Tests for request/response, the service manager and hardware availability.

The availability tests matter most: they pin the property that the runtime can
never quietly substitute a simulated device for a real one, which is the whole
reason that module exists.
"""

from __future__ import annotations

import threading

import pytest

from neurogrip.core.clock import SimulatedClock
from neurogrip.core.config import Config
from neurogrip.core.errors import ServiceError
from neurogrip.core.events import EventBus
from neurogrip.core.lifecycle import HealthReport, ServiceBase
from neurogrip.core.requests import (
    NoResponder,
    RequestBroker,
    RequestError,
    RequestTimeout,
)
from neurogrip.core.topics import Topics
from neurogrip.hal.availability import (
    Availability,
    DeviceStatus,
    HardwareInventory,
    HardwareScanner,
    ProductionRequirements,
    Requirement,
)
from neurogrip.runtime.manager import RestartPolicy, ServiceManager, ServiceState

# --------------------------------------------------------------------------- #
# request / response
# --------------------------------------------------------------------------- #


@pytest.fixture()
def bus() -> EventBus:
    return EventBus(SimulatedClock())


def test_request_returns_the_responder_result(bus: EventBus) -> None:
    broker = RequestBroker(bus, SimulatedClock())
    broker.respond("servo.limits", lambda payload: {"max_current_ma": 900})
    assert broker.request("servo.limits") == {"max_current_ma": 900}


def test_request_passes_the_payload_through(bus: EventBus) -> None:
    broker = RequestBroker(bus, SimulatedClock())
    broker.respond("math.double", lambda payload: payload * 2)
    assert broker.request("math.double", 21) == 42


def test_missing_responder_is_distinct_from_a_timeout(bus: EventBus) -> None:
    """Wiring bugs and slow responders need different diagnoses."""
    broker = RequestBroker(bus, SimulatedClock())
    with pytest.raises(NoResponder):
        broker.request("nobody.home")


def test_responder_exception_surfaces_at_the_caller(bus: EventBus) -> None:
    broker = RequestBroker(bus, SimulatedClock())

    def explode(_payload: object) -> None:
        raise ValueError("bad calibration")

    broker.respond("emg.calibration", explode)
    with pytest.raises(RequestError, match="bad calibration"):
        broker.request("emg.calibration")


def test_request_times_out_when_the_reply_never_arrives(bus: EventBus) -> None:
    broker = RequestBroker(bus, SimulatedClock())
    # A responder that accepts the request but never replies.
    bus.subscribe("slow.topic", lambda ev: None)
    broker._responders["slow.topic"] = bus.subscribe("slow.topic", lambda ev: None)
    with pytest.raises(RequestTimeout):
        broker.request("slow.topic", timeout=0.05)
    assert broker.timeouts == 1


def test_zero_timeout_is_rejected(bus: EventBus) -> None:
    """A request without a deadline is a latent hang."""
    broker = RequestBroker(bus, SimulatedClock())
    broker.respond("x", lambda _p: 1)
    with pytest.raises(RequestError, match="timeout must be positive"):
        broker.request("x", timeout=0)


def test_cross_thread_reply_is_correlated(bus: EventBus) -> None:
    """The Event path, not the inline path."""
    broker = RequestBroker(bus, SimulatedClock())
    started = threading.Event()

    def responder(_payload: object) -> str:
        started.set()
        return "from-thread"

    broker.respond("threaded", responder)
    result: list[str] = []
    worker = threading.Thread(target=lambda: result.append(broker.request("threaded")))
    worker.start()
    worker.join(timeout=2.0)
    assert started.is_set()
    assert result == ["from-thread"]


def test_try_request_degrades_instead_of_raising(bus: EventBus) -> None:
    broker = RequestBroker(bus, SimulatedClock())
    assert broker.try_request("absent", default="fallback") == "fallback"


def test_duplicate_responder_is_rejected(bus: EventBus) -> None:
    broker = RequestBroker(bus, SimulatedClock())
    broker.respond("dup", lambda _p: 1)
    with pytest.raises(RequestError):
        broker.respond("dup", lambda _p: 2)


# --------------------------------------------------------------------------- #
# service manager
# --------------------------------------------------------------------------- #


class _Fake(ServiceBase):
    def __init__(self, name: str, *, fail_start: bool = False) -> None:
        super().__init__()
        self.service_name = name
        self.fail_start = fail_start
        self.starts = 0
        self.stops = 0
        self.report = HealthReport.ok(name)

    def on_start(self) -> None:
        self.starts += 1
        if self.fail_start:
            raise RuntimeError("boom")

    def on_stop(self) -> None:
        self.stops += 1

    def health(self) -> HealthReport:
        return self.report


def test_services_start_in_dependency_order() -> None:
    manager = ServiceManager(clock=SimulatedClock())
    manager.register(_Fake("control"), depends_on=("servo",))
    manager.register(_Fake("servo"), depends_on=("transport",))
    manager.register(_Fake("transport"))
    assert manager.resolve_order() == ("transport", "servo", "control")


def test_shutdown_is_exact_reverse_of_startup() -> None:
    """The servo bus must be disabled before its transport closes."""
    order: list[str] = []

    class Recorder(_Fake):
        def on_start(self) -> None:
            order.append(f"start:{self.name}")

        def on_stop(self) -> None:
            order.append(f"stop:{self.name}")

    manager = ServiceManager(clock=SimulatedClock())
    manager.register(Recorder("control"), depends_on=("servo",))
    manager.register(Recorder("servo"))
    manager.start_all()
    manager.stop_all()
    assert order == ["start:servo", "start:control", "stop:control", "stop:servo"]


def test_circular_dependency_is_rejected() -> None:
    manager = ServiceManager(clock=SimulatedClock())
    manager.register(_Fake("a"), depends_on=("b",))
    manager.register(_Fake("b"), depends_on=("a",))
    with pytest.raises(ServiceError, match="circular"):
        manager.resolve_order()


def test_unknown_dependency_is_rejected() -> None:
    manager = ServiceManager(clock=SimulatedClock())
    manager.register(_Fake("a"), depends_on=("ghost",))
    with pytest.raises(ServiceError, match="unknown service"):
        manager.resolve_order()


def test_optional_service_failure_does_not_abort_startup() -> None:
    """A missing camera must not stop the hand from working."""
    manager = ServiceManager(clock=SimulatedClock())
    manager.register(_Fake("camera", fail_start=True), required=False)
    good = _Fake("servo")
    manager.register(good)
    assert manager.start_all() is True
    assert good.running
    assert manager.get("camera").state is ServiceState.FAILED


def test_required_service_failure_reports_not_ready() -> None:
    manager = ServiceManager(clock=SimulatedClock())
    manager.register(_Fake("servo", fail_start=True), required=True)
    assert manager.start_all() is False


def test_dependent_of_a_failed_service_is_skipped_not_started() -> None:
    manager = ServiceManager(clock=SimulatedClock())
    manager.register(_Fake("transport", fail_start=True), required=False)
    dependent = _Fake("servo")
    manager.register(dependent, depends_on=("transport",), required=False)
    manager.start_all()
    assert manager.get("servo").state is ServiceState.SKIPPED
    assert dependent.starts == 0


def test_supervise_restarts_only_failed_services() -> None:
    """DEGRADED is still usable; bouncing it would make things worse."""
    clock = SimulatedClock()
    manager = ServiceManager(clock=clock)
    degraded = _Fake("vision")
    degraded.report = HealthReport.degraded("vision", "low light")
    failed = _Fake("emg")
    manager.register(degraded, required=False)
    manager.register(failed, required=False)
    manager.start_all()
    failed.report = HealthReport.failed("emg", "electrodes detached")

    clock.advance(1.0)
    restarted = manager.supervise()
    assert restarted == ("emg",)
    assert degraded.starts == 1        # untouched
    assert failed.starts == 2          # restarted once


def test_restart_budget_is_bounded_then_gives_up() -> None:
    """An unbounded crash loop on the motor controller is a safety problem."""
    clock = SimulatedClock()
    manager = ServiceManager(clock=clock)
    svc = _Fake("motor")
    manager.register(svc, policy=RestartPolicy(max_restarts=2, window_s=60.0))
    manager.start_all()

    assert manager.restart("motor") is True
    assert manager.restart("motor") is True
    assert manager.restart("motor") is False           # budget exhausted
    assert manager.get("motor").state is ServiceState.FAILED


def test_restart_policy_never_suppresses_automatic_revival() -> None:
    manager = ServiceManager(clock=SimulatedClock())
    manager.register(_Fake("motor"), policy=RestartPolicy.never())
    manager.start_all()
    assert manager.restart("motor") is False


def test_forced_restart_bypasses_the_budget() -> None:
    """An operator at the diagnostics screen has decided."""
    manager = ServiceManager(clock=SimulatedClock())
    manager.register(_Fake("motor"), policy=RestartPolicy(max_restarts=0))
    manager.start_all()
    assert manager.restart("motor", force=True) is True


def test_manager_publishes_service_errors(bus: EventBus) -> None:
    seen: list[object] = []
    bus.subscribe(Topics.SERVICE_ERROR, lambda ev: seen.append(ev.payload))
    manager = ServiceManager(bus=bus, clock=SimulatedClock())
    manager.register(_Fake("emg", fail_start=True), required=False)
    manager.start_all()
    assert seen and seen[0]["service"] == "emg"


def test_stop_all_never_raises_even_when_a_service_misbehaves() -> None:
    class BadStop(_Fake):
        def on_stop(self) -> None:
            raise RuntimeError("stop exploded")

    manager = ServiceManager(clock=SimulatedClock())
    manager.register(BadStop("bad"))
    good = _Fake("good")
    manager.register(good)
    manager.start_all()
    manager.stop_all()                    # must not raise
    assert good.stops == 1


# --------------------------------------------------------------------------- #
# hardware availability
# --------------------------------------------------------------------------- #


def _inventory(*devices: DeviceStatus) -> HardwareInventory:
    return HardwareInventory(devices=devices)


def test_availability_distinguishes_disabled_from_missing() -> None:
    """Disabled is an operator choice, not a fault."""
    assert Availability.DISABLED.is_fault is False
    assert Availability.MISSING.is_fault is True
    assert Availability.ERROR.is_fault is True
    assert Availability.DETECTED.is_usable is True


def test_device_status_is_never_simulated() -> None:
    status = DeviceStatus("emg", "real_ads1115", Availability.DETECTED)
    assert status.simulated is False


def test_production_gate_lists_each_missing_item_individually() -> None:
    """An operator needs to know it is camera 1 AND 2, not '2 missing'."""
    inv = _inventory(
        DeviceStatus("emg", "real_serial", Availability.DETECTED),
        DeviceStatus("servo_controller", "real_esp32", Availability.DETECTED),
        DeviceStatus("servo", "real_esp32", Availability.DETECTED, count=5),
    )
    missing = inv.missing_for(ProductionRequirements())
    assert missing == ["Depth camera 1", "Depth camera 2"]


def test_operator_message_matches_the_specified_format() -> None:
    inv = _inventory(
        DeviceStatus("emg", "real_serial", Availability.DETECTED),
        DeviceStatus("servo_controller", "real_esp32", Availability.DETECTED),
        DeviceStatus("servo", "real_esp32", Availability.DETECTED, count=5),
    )
    message = inv.operator_message(ProductionRequirements())
    assert message.startswith("BetaTouch cannot start.")
    assert "- Depth camera 1" in message
    assert "- Depth camera 2" in message
    assert message.rstrip().endswith("Please connect all required hardware.")


def test_partial_servo_count_blocks_production() -> None:
    """Three motors on a good controller is not a working hand."""
    inv = _inventory(
        DeviceStatus("emg", "real_serial", Availability.DETECTED),
        DeviceStatus("servo_controller", "real_esp32", Availability.DETECTED),
        DeviceStatus("servo", "real_esp32", Availability.DETECTED, count=3),
        DeviceStatus("depth_camera", "real_v4l2", Availability.DETECTED),
        DeviceStatus("depth_camera", "real_v4l2", Availability.DETECTED),
    )
    assert inv.is_production_ready(ProductionRequirements()) is False
    assert inv.missing_for(ProductionRequirements()) == ["Servo motor 4", "Servo motor 5"]


def test_disabled_hardware_still_blocks_production() -> None:
    """Turning a required device off must not be a way past the gate."""
    inv = _inventory(
        DeviceStatus("emg", "disabled", Availability.DISABLED),
        DeviceStatus("servo_controller", "real_esp32", Availability.DETECTED),
        DeviceStatus("servo", "real_esp32", Availability.DETECTED, count=5),
        DeviceStatus("depth_camera", "real_v4l2", Availability.DETECTED),
        DeviceStatus("depth_camera", "real_v4l2", Availability.DETECTED),
    )
    assert inv.is_production_ready(ProductionRequirements()) is False
    assert "EMG sensor front-end" in inv.missing_for(ProductionRequirements())


def test_fully_populated_inventory_is_production_ready() -> None:
    inv = _inventory(
        DeviceStatus("emg", "real_ads1115", Availability.DETECTED),
        DeviceStatus("servo_controller", "real_servo_controller", Availability.DETECTED),
        DeviceStatus("servo", "real_servo_controller", Availability.DETECTED, count=5),
        DeviceStatus("depth_camera", "real_v4l2", Availability.DETECTED),
        DeviceStatus("depth_camera", "real_v4l2", Availability.DETECTED),
    )
    assert inv.is_production_ready(ProductionRequirements()) is True
    assert inv.operator_message(ProductionRequirements()) == ""


def test_custom_requirements_are_honoured() -> None:
    inv = _inventory(DeviceStatus("emg", "real_serial", Availability.DETECTED))
    reqs = ProductionRequirements([Requirement("emg", 1, "EMG")])
    assert inv.is_production_ready(reqs) is True


def test_scanner_reports_disabled_backends_as_disabled() -> None:
    config = Config({
        "emg": {"backend": "disabled"},
        "servo": {"backend": "disabled"},
        "camera": {"backend": "disabled"},
    })
    inventory = HardwareScanner(config).scan()
    assert inventory.status("emg") is Availability.DISABLED
    assert inventory.status("servo_controller") is Availability.DISABLED


def test_scanner_reports_absent_devices_as_missing_not_simulated() -> None:
    """The core guarantee: no stand-in is ever substituted."""
    config = Config({
        "emg": {"backend": "real_serial", "port": "/nonexistent/ttyACM9"},
        "servo": {"backend": "real_servo_controller", "port": "/nonexistent/ttyUSB9"},
        "camera": {"backend": "real_depth_camera", "depth_devices": ["/nonexistent/video9"]},
    })
    inventory = HardwareScanner(config).scan()
    assert inventory.status("emg") is Availability.MISSING
    assert inventory.status("servo_controller") is Availability.MISSING
    assert inventory.status("servo") is Availability.MISSING
    assert all(not d.simulated for d in inventory.devices)
    assert inventory.is_production_ready(ProductionRequirements()) is False


def test_unverifiable_motors_count_as_missing() -> None:
    """No controller means the motor count cannot be trusted."""
    config = Config({
        "servo": {"backend": "real_servo_controller", "port": "/nonexistent/ttyUSB9"},
    })
    inventory = HardwareScanner(config).scan()
    assert inventory.count_detected("servo") == 0


def test_unknown_backend_is_an_error_not_a_silent_default() -> None:
    config = Config({"emg": {"backend": "totally_made_up"}})
    inventory = HardwareScanner(config).scan()
    assert inventory.status("emg") is Availability.ERROR
