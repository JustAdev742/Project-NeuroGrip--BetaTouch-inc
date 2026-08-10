"""Safety rules.

Each rule is a small, independent, testable predicate over the system state that
returns a :class:`Fault` or ``None``. Keeping them separate means:

* each can be unit-tested in isolation (``tests/unit/test_safety_rules.py``);
* the set is auditable — you can read the whole safety argument in one file;
* adding a rule cannot accidentally weaken an existing one;
* rules can be individually disabled for bench testing *and* that fact is
  reported, so nobody runs a device with a rule quietly switched off.

Every rule declares a :class:`~neurogrip.core.errors.Severity`, which the monitor
maps to an action:

============  ==============================================================
severity      response
============  ==============================================================
``MINOR``     log it
``DEGRADED``  disable the affected feature (e.g. vision assistance)
``FALLBACK``  disable AI assistance; direct manual control continues
``CRITICAL``  emergency stop
============  ==============================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from ..control.controller import HandState
from ..core.errors import Severity
from ..core.types import clamp
from ..emg.intent import IntentEstimate
from ..emg.quality import SignalQuality
from ..hal.system import BatteryState
from ..vision.types import VisionResult

__all__ = [
    "DEFAULT_RULES",
    "BatteryRule",
    "CommunicationRule",
    "Fault",
    "GripForceRule",
    "OverCurrentRule",
    "SafetyContext",
    "SafetyRule",
    "SensorFailureRule",
    "ServoTimeoutRule",
    "StaleIntentRule",
    "ThermalRule",
    "VisionHealthRule",
]


@dataclass(frozen=True, slots=True)
class SafetyContext:
    """Everything the safety rules evaluate against."""

    timestamp: float
    hand: HandState
    intent: IntentEstimate | None = None
    vision: VisionResult | None = None
    battery: BatteryState | None = None
    #: Highest CPU temperature reported by the host, °C.
    cpu_temperature_c: float = 0.0
    #: True while a motion is being executed.
    moving: bool = False
    #: Seconds since the control loop last completed a cycle.
    control_loop_age: float = 0.0
    #: Names of watchdogs currently expired.
    expired_watchdogs: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class Fault:
    """A detected safety condition."""

    #: Stable identifier, used for deduplication and for the fault log.
    code: str
    severity: Severity
    message: str
    #: Which rule produced it.
    rule: str = ""
    #: Structured values for the log and the diagnostics screen.
    detail: dict[str, float | str] = field(default_factory=dict)
    #: Suggested force ceiling while this fault is active; ``1.0`` = no derate.
    force_ceiling: float = 1.0
    #: What the user should do about it, in plain language.
    remedy: str = ""

    def __str__(self) -> str:  # pragma: no cover - display helper
        return f"[{self.severity.name}] {self.message}"


@runtime_checkable
class SafetyRule(Protocol):
    """A single safety check."""

    @property
    def name(self) -> str: ...

    @property
    def enabled(self) -> bool: ...

    def evaluate(self, context: SafetyContext) -> Fault | None:
        """Return a fault when the rule is violated, otherwise ``None``."""
        ...


class _BaseRule:
    """Shared plumbing: a name and an enable flag."""

    rule_name = "rule"

    def __init__(self, *, enabled: bool = True) -> None:
        self._enabled = enabled

    @property
    def name(self) -> str:
        return self.rule_name

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, value: bool) -> None:
        self._enabled = value


class GripForceRule(_BaseRule):
    """Enforces the maximum grip force.

    The single most important rule in the file. A tendon-driven hand can exert
    enough force to injure the wearer's residual limb through the socket, to
    crush what it is holding, or to destroy itself. The ceiling is absolute and
    is applied regardless of mode, plan or user effort.
    """

    rule_name = "grip_force"

    def __init__(self, max_force_n: float = 45.0, *, enabled: bool = True) -> None:
        super().__init__(enabled=enabled)
        self._max_force_n = max_force_n

    def evaluate(self, context: SafetyContext) -> Fault | None:
        grip = context.hand.grip
        if grip is None or not grip.holding:
            return None
        if grip.estimated_force_n <= self._max_force_n:
            return None
        return Fault(
            code="grip_force_exceeded",
            severity=Severity.CRITICAL,
            message=(
                f"grip force {grip.estimated_force_n:.0f} N exceeds the "
                f"{self._max_force_n:.0f} N limit"
            ),
            rule=self.name,
            detail={"force_n": grip.estimated_force_n, "limit_n": self._max_force_n},
            force_ceiling=0.3,
            remedy="Release and re-grip. If this repeats, recalibrate grip force.",
        )


class OverCurrentRule(_BaseRule):
    """Detects sustained total current above the bus limit."""

    rule_name = "overcurrent"

    def __init__(self, limit_ma: int = 3500, *, enabled: bool = True) -> None:
        super().__init__(enabled=enabled)
        self._limit_ma = limit_ma

    def evaluate(self, context: SafetyContext) -> Fault | None:
        total = context.hand.total_current_ma
        if total <= self._limit_ma:
            return None
        return Fault(
            code="overcurrent",
            severity=Severity.CRITICAL,
            message=f"total motor current {total} mA exceeds {self._limit_ma} mA",
            rule=self.name,
            detail={"current_ma": float(total), "limit_ma": float(self._limit_ma)},
            force_ceiling=0.25,
            remedy="Check for a jammed finger or an obstruction.",
        )


class ThermalRule(_BaseRule):
    """Derates and then stops on motor or SoC over-temperature.

    Two thresholds rather than one: a warning band that reduces force (letting
    the user finish what they are doing) and a hard limit that stops. Going
    straight from "fine" to "stopped" in the middle of holding a hot drink would
    be its own hazard.
    """

    rule_name = "thermal"

    def __init__(
        self,
        motor_warn_c: float = 58.0,
        motor_limit_c: float = 70.0,
        cpu_limit_c: float = 85.0,
        *,
        enabled: bool = True,
    ) -> None:
        super().__init__(enabled=enabled)
        self._motor_warn = motor_warn_c
        self._motor_limit = motor_limit_c
        self._cpu_limit = cpu_limit_c

    def evaluate(self, context: SafetyContext) -> Fault | None:
        motor = context.hand.temperature_c
        if motor >= self._motor_limit:
            return Fault(
                code="motor_overtemperature",
                severity=Severity.CRITICAL,
                message=f"motor temperature {motor:.0f} °C at the {self._motor_limit:.0f} °C limit",
                rule=self.name,
                detail={"temperature_c": motor},
                force_ceiling=0.2,
                remedy="Let the hand cool for a few minutes before resuming.",
            )
        if motor >= self._motor_warn:
            # Linear derate across the warning band.
            span = max(1e-6, self._motor_limit - self._motor_warn)
            derate = clamp(1.0 - (motor - self._motor_warn) / span, 0.35, 1.0)
            return Fault(
                code="motor_hot",
                severity=Severity.DEGRADED,
                message=f"motors are hot ({motor:.0f} °C); force reduced",
                rule=self.name,
                detail={"temperature_c": motor, "derate": derate},
                force_ceiling=derate,
                remedy="Reduce sustained gripping to let the motors cool.",
            )
        if context.cpu_temperature_c >= self._cpu_limit:
            return Fault(
                code="cpu_overtemperature",
                severity=Severity.FALLBACK,
                message=f"processor at {context.cpu_temperature_c:.0f} °C; disabling AI assistance",
                rule=self.name,
                detail={"cpu_c": context.cpu_temperature_c},
                remedy="The device is throttling. Manual control continues.",
            )
        return None


class CommunicationRule(_BaseRule):
    """Detects loss of the motor-controller link."""

    rule_name = "communication"

    def evaluate(self, context: SafetyContext) -> Fault | None:
        if context.hand.comms_ok:
            return None
        return Fault(
            code="motor_link_lost",
            severity=Severity.CRITICAL,
            message="no communication with the motor controller",
            rule=self.name,
            detail={"faults": ", ".join(context.hand.faults)},
            force_ceiling=0.0,
            remedy="Check the controller cable and power, then acknowledge to retry.",
        )


class ServoTimeoutRule(_BaseRule):
    """Detects a servo that is not tracking its commanded position.

    A finger commanded to move that is not moving, and is not drawing the
    current that would indicate it has met an object, has failed — a stripped
    horn, a snapped tendon or a dead channel.
    """

    rule_name = "servo_timeout"

    #: Current above which a finger is judged to be loaded rather than failed.
    #: Just above the unloaded holding current — a finger pressing on something
    #: compliant draws only slightly more than one hanging free.
    LOADED_CURRENT_MA = 110

    def __init__(
        self, *, error_threshold: float = 0.25, dwell_s: float = 0.6, enabled: bool = True
    ) -> None:
        super().__init__(enabled=enabled)
        self._threshold = error_threshold
        #: The condition must persist this long. A finger takes time to reach a
        #: new target, and reporting a fault during normal travel would make the
        #: rule fire on every large movement.
        self._dwell = dwell_s
        self._since: dict[int, float] = {}

    def evaluate(self, context: SafetyContext) -> Fault | None:
        hand = context.hand
        if not context.moving or not hand.enabled:
            self._since.clear()
            return None
        grip = hand.grip
        contacts = set(grip.contacts) if grip else set()

        for index, (commanded, measured) in enumerate(zip(hand.commanded, hand.pose)):
            from ..core.types import Finger

            finger = Finger(index)
            current = hand.currents[index] if index < len(hand.currents) else 0
            lagging = abs(commanded - measured) >= self._threshold
            if finger in contacts or not lagging or current > self.LOADED_CURRENT_MA:
                # Blocked by an object, on target, or drawing current: not a fault.
                self._since.pop(index, None)
                continue

            started = self._since.setdefault(index, context.timestamp)
            if context.timestamp - started < self._dwell:
                continue

            return Fault(
                code="servo_not_tracking",
                severity=Severity.FALLBACK,
                message=(
                    f"{finger.label} is not tracking its command "
                    f"({measured:.2f} vs {commanded:.2f}) and is drawing {current} mA"
                ),
                rule=self.name,
                detail={"finger": finger.label, "error": abs(commanded - measured)},
                remedy="Check the tendon and servo for that finger.",
            )
        return None


class SensorFailureRule(_BaseRule):
    """Detects unusable EMG — the input the whole system depends on."""

    rule_name = "sensor_failure"

    def evaluate(self, context: SafetyContext) -> Fault | None:
        intent = context.intent
        if intent is None:
            # "No intent yet" is the normal state for the first few hundred
            # milliseconds after start-up, and raising a fault for it would put
            # every boot into fallback. The EMG watchdog is the authority on
            # whether data has actually stopped arriving; defer to it.
            if "emg" not in context.expired_watchdogs:
                return None
            return Fault(
                code="emg_absent",
                severity=Severity.FALLBACK,
                message="no EMG data; the hand cannot read user intent",
                rule=self.name,
                remedy="Check the electrode connections.",
            )
        if intent.quality <= SignalQuality.UNUSABLE:
            return Fault(
                code="emg_unusable",
                severity=Severity.FALLBACK,
                message="EMG signal quality is unusable",
                rule=self.name,
                detail={"quality": intent.quality.label},
                force_ceiling=0.4,
                remedy="Re-seat the electrodes; the skin may be dry or the lead loose.",
            )
        if intent.quality < SignalQuality.FAIR:
            return Fault(
                code="emg_poor",
                severity=Severity.DEGRADED,
                message=f"EMG quality is {intent.quality.label}; AI assistance limited",
                rule=self.name,
                detail={"quality": intent.quality.label},
                remedy="Re-seat the electrodes or run the calibration wizard.",
            )
        return None


class StaleIntentRule(_BaseRule):
    """Refuses motion when intent data has stopped arriving.

    Complements the EMG watchdog: the watchdog notices that *data* stopped, this
    notices that a *decision* is being made on data that is too old — for
    example if the pipeline is running but stalled on one frame.
    """

    rule_name = "stale_intent"

    def __init__(self, max_age_s: float = 0.5, *, enabled: bool = True) -> None:
        super().__init__(enabled=enabled)
        self._max_age = max_age_s

    def evaluate(self, context: SafetyContext) -> Fault | None:
        intent = context.intent
        if intent is None or not context.moving:
            return None
        age = intent.age(context.timestamp)
        if age <= self._max_age:
            return None
        return Fault(
            code="intent_stale",
            severity=Severity.FALLBACK,
            message=f"acting on {age * 1000:.0f} ms old intent; stopping motion",
            rule=self.name,
            detail={"age_ms": age * 1000},
            force_ceiling=0.0,
            remedy="Transient. If it persists, check EMG acquisition.",
        )


class BatteryRule(_BaseRule):
    """Degrades then stops as the battery runs down.

    A hand that dies mid-grip while carrying something is a hazard, so
    assistance is shed early (at 15 %) to extend the window in which the user
    can finish and put things down safely.
    """

    rule_name = "battery"

    def evaluate(self, context: SafetyContext) -> Fault | None:
        battery = context.battery
        if battery is None or not battery.present or battery.charging:
            return None
        if battery.is_critical:
            return Fault(
                code="battery_critical",
                severity=Severity.CRITICAL,
                message=f"battery critically low ({battery.percentage:.0f} %)",
                rule=self.name,
                detail={"percentage": battery.percentage, "voltage": battery.voltage_v},
                force_ceiling=0.0,
                remedy="Charge the hand now. Motion is disabled to protect the cells.",
            )
        if battery.is_low:
            return Fault(
                code="battery_low",
                severity=Severity.DEGRADED,
                message=f"battery low ({battery.percentage:.0f} %); AI assistance disabled",
                rule=self.name,
                detail={"percentage": battery.percentage},
                force_ceiling=0.7,
                remedy="Charge soon. Manual control is unaffected.",
            )
        return None


class VisionHealthRule(_BaseRule):
    """Reports vision unavailability. Degraded only — never blocks motion.

    Vision failing must never stop the hand. This rule exists so the condition
    is *visible* on the diagnostics screen and in the log, and so the fusion
    layer's AI gate is closed explicitly rather than by accident.
    """

    rule_name = "vision_health"

    def __init__(self, max_age_s: float = 2.0, *, enabled: bool = True) -> None:
        super().__init__(enabled=enabled)
        self._max_age = max_age_s

    def evaluate(self, context: SafetyContext) -> Fault | None:
        vision = context.vision
        if vision is None:
            return Fault(
                code="vision_unavailable",
                severity=Severity.DEGRADED,
                message="vision is unavailable; grasp selection falls back to defaults",
                rule=self.name,
                remedy="Check the camera connection. The hand works normally without it.",
            )
        if not vision.ok:
            return Fault(
                code="vision_error",
                severity=Severity.DEGRADED,
                message=f"vision error: {vision.error}",
                rule=self.name,
                remedy="Check the model files and the camera.",
            )
        if not vision.is_fresh(context.timestamp, self._max_age):
            return Fault(
                code="vision_stale",
                severity=Severity.DEGRADED,
                message=f"vision is {vision.age(context.timestamp):.1f} s stale",
                rule=self.name,
                remedy="The camera may have stopped. Manual control is unaffected.",
            )
        return None


#: The rule set installed by default. Order is irrelevant — the monitor takes
#: the worst severity — but it is written worst-first for readability.
DEFAULT_RULES: tuple[type[_BaseRule], ...] = (
    GripForceRule,
    OverCurrentRule,
    ThermalRule,
    CommunicationRule,
    BatteryRule,
    SensorFailureRule,
    StaleIntentRule,
    ServoTimeoutRule,
    VisionHealthRule,
)
