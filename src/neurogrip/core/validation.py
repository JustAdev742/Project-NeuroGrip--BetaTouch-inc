"""Configuration validation.

Layered configuration has one dangerous property: every lookup has a default, so
a misspelled key is indistinguishable from an absent one. ``max_forse = 0.4``
does not fail — it is ignored, and the hand runs at the 0.85 default. On a device
that squeezes things, silently ignoring a force limit is the worst possible
failure mode for a typo.

Three classes of check, in increasing order of how much they are worth:

* **Type and range.** ``servo.max_force = 2.0`` is not a valid force.
* **Unknown keys.** Caught by comparing against the set of paths the code
  actually reads. Reported as warnings, not errors, because deployment profiles
  legitimately carry keys this build does not know about.
* **Cross-field consistency.** The valuable ones. A watchdog shorter than the
  loop period it guards will trip constantly; a host velocity limit above the
  actuator's does nothing except mislead whoever reads it. Neither is detectable
  by looking at one value in isolation, and both produce behaviour that is very
  hard to diagnose from the symptom.

Validation runs at startup and through ``neurogrip config --check``. Errors
refuse the boot; warnings are logged and shown on the diagnostics screen.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum

from .config import Config

__all__ = [
    "SYSTEM_RULES",
    "Rule",
    "ValidationIssue",
    "ValidationReport",
    "ValidationSeverity",
    "validate_config",
]


class ValidationSeverity(str, Enum):
    """How much a problem matters."""

    #: Refuses startup. The value is unsafe or the system cannot run with it.
    ERROR = "error"
    #: Logged and surfaced, but the system starts.
    WARNING = "warning"


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """One problem found in the configuration."""

    path: str
    message: str
    severity: ValidationSeverity = ValidationSeverity.ERROR
    #: What the user should do about it.
    remedy: str = ""

    def __str__(self) -> str:
        suffix = f" — {self.remedy}" if self.remedy else ""
        return f"[{self.severity.value}] {self.path}: {self.message}{suffix}"


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Result of validating a whole configuration."""

    issues: tuple[ValidationIssue, ...] = field(default_factory=tuple)

    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        return tuple(i for i in self.issues if i.severity is ValidationSeverity.ERROR)

    @property
    def warnings(self) -> tuple[ValidationIssue, ...]:
        return tuple(i for i in self.issues if i.severity is ValidationSeverity.WARNING)

    @property
    def ok(self) -> bool:
        """True when nothing blocks startup. Warnings do not."""
        return not self.errors

    def describe(self) -> tuple[str, ...]:
        return tuple(str(issue) for issue in self.issues)

    def raise_if_invalid(self) -> None:
        """Raise :class:`~neurogrip.core.errors.ConfigurationError` on any error."""
        if self.ok:
            return
        from .errors import ConfigurationError

        detail = "; ".join(str(issue) for issue in self.errors)
        raise ConfigurationError(f"invalid configuration: {detail}")


@dataclass(frozen=True, slots=True)
class Rule:
    """A declarative check on one configuration path."""

    path: str
    kind: type = float
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple[str, ...] = ()
    #: When set, the value must be present — no default is acceptable.
    required: bool = False
    severity: ValidationSeverity = ValidationSeverity.ERROR
    remedy: str = ""

    def check(self, config: Config) -> ValidationIssue | None:
        if self.path not in config:
            if self.required:
                return ValidationIssue(
                    self.path, "required but not set", self.severity, self.remedy
                )
            return None

        value = config.get(self.path)

        if self.kind is bool:
            # Checked before the numeric branch: bool is a subclass of int, so
            # `isinstance(True, int)` would otherwise let `enabled = true` pass
            # a rule meant for a number.
            if not isinstance(value, bool):
                return ValidationIssue(
                    self.path, f"must be true or false, got {value!r}", self.severity, self.remedy
                )
            return None

        if self.kind is str:
            if not isinstance(value, str):
                return ValidationIssue(
                    self.path, f"must be text, got {value!r}", self.severity, self.remedy
                )
            if self.choices and value not in self.choices:
                return ValidationIssue(
                    self.path,
                    f"{value!r} is not one of {', '.join(self.choices)}",
                    self.severity,
                    self.remedy,
                )
            return None

        if isinstance(value, bool) or not isinstance(value, int | float):
            return ValidationIssue(
                self.path, f"must be a number, got {value!r}", self.severity, self.remedy
            )
        if self.minimum is not None and value < self.minimum:
            return ValidationIssue(
                self.path,
                f"{value} is below the minimum of {self.minimum}",
                self.severity,
                self.remedy,
            )
        if self.maximum is not None and value > self.maximum:
            return ValidationIssue(
                self.path,
                f"{value} is above the maximum of {self.maximum}",
                self.severity,
                self.remedy,
            )
        return None


#: Per-value rules. Bounds are "physically implausible" limits, not preferences:
#: anything outside them is a mistake rather than an unusual choice.
SYSTEM_RULES: tuple[Rule, ...] = (
    # -- actuator safety ------------------------------------------------------
    Rule("servo.max_force", float, 0.05, 1.0,
         remedy="grip force is normalised; 1.0 is the actuator's mechanical maximum"),
    Rule("servo.max_velocity", float, 0.1, 10.0),
    Rule("servo.max_acceleration", float, 0.1, 200.0),
    Rule("servo.max_current_ma", int, 50, 5000),
    Rule("servo.max_temperature_c", float, 30.0, 120.0),
    Rule("servo.watchdog_ms", int, 50, 2000,
         remedy="the firmware safes the drive after this long without a command"),
    Rule("servo.driver", str, choices=("microbit", "esp32", "emulator", "simulated")),
    Rule("servo.baud", int, 9600, 4_000_000),
    # -- host control ---------------------------------------------------------
    Rule("control.max_velocity", float, 0.1, 10.0),
    Rule("control.max_acceleration", float, 0.1, 200.0),
    Rule("control.position_tolerance", float, 0.001, 0.2),
    # -- loop rates -----------------------------------------------------------
    Rule("runtime.control_hz", float, 20.0, 1000.0,
         remedy="below ~50 Hz the hand visibly steps between positions"),
    Rule("runtime.emg_hz", float, 20.0, 2000.0),
    Rule("runtime.decision_hz", float, 5.0, 500.0),
    Rule("runtime.vision_hz", float, 1.0, 120.0),
    Rule("runtime.ui_hz", float, 1.0, 60.0),
    # -- EMG ------------------------------------------------------------------
    Rule("emg.sample_rate_hz", float, 200.0, 8000.0,
         remedy="EMG energy extends to ~400 Hz; sample at least 1 kHz"),
    Rule("emg.band_low_hz", float, 1.0, 100.0),
    Rule("emg.band_high_hz", float, 100.0, 1000.0),
    Rule("emg.mains_hz", float, 45.0, 65.0, choices=(),
         remedy="mains is 50 Hz or 60 Hz depending on region"),
    Rule("emg.onset_threshold", float, 0.01, 1.0),
    Rule("emg.offset_threshold", float, 0.0, 1.0),
    Rule("emg.dwell_s", float, 0.0, 2.0,
         remedy="long dwell makes the hand feel unresponsive"),
    # -- fusion ---------------------------------------------------------------
    Rule("fusion.min_intent_confidence", float, 0.0, 1.0),
    Rule("fusion.min_vision_confidence", float, 0.0, 1.0),
    Rule("fusion.max_intent_age_s", float, 0.05, 5.0,
         remedy="how stale an intent may be and still authorise motion"),
    # -- safety ---------------------------------------------------------------
    Rule("safety.estop_check.rehearsal_interval_s", float, 1.0, 3600.0,
         remedy="how long a broken e-stop signalling path could go unnoticed"),
    Rule("safety.estop_check.proof_interval_s", float, 60.0, 604800.0),
    Rule("safety.estop_check.proof_enabled", bool),
    Rule("safety.estop_check.trigger_probe_interval_s", float, 5.0, 86400.0),
    Rule("safety.watchdogs.control_s", float, 0.005, 5.0),
    Rule("safety.watchdogs.emg_s", float, 0.02, 10.0),
    Rule("safety.watchdogs.decision_s", float, 0.01, 10.0),
    Rule("safety.watchdogs.vision_s", float, 0.1, 30.0),
    # -- vision ---------------------------------------------------------------
    Rule("vision.max_result_age_s", float, 0.05, 10.0),
    Rule("camera.fov_deg", float, 10.0, 180.0),
    # -- UI -------------------------------------------------------------------
    Rule(
        "vision.backend",
        str,
        choices=("hggd_mcu", "onnx_detector", "anygrasp", "replay", "mock", "null"),
    ),
    Rule("ui.theme", str, choices=("dark", "light", "high_contrast", "auto")),
    Rule("ui.accessibility.font_scale", float, 0.5, 3.0),
    Rule("modes.default", str, choices=("manual", "ai_assist", "sports", "training")),
)


def _cross_checks(config: Config) -> list[ValidationIssue]:
    """Consistency checks between values that are individually valid.

    These are the ones worth having. Each corresponds to a failure that is
    obvious in hindsight and baffling at the time.
    """
    issues: list[ValidationIssue] = []

    control_hz = config.get_float("runtime.control_hz", 200.0)
    control_period = 1.0 / control_hz if control_hz > 0 else 0.0

    # A watchdog shorter than the loop it guards trips on every healthy cycle.
    for name, rate_key, default_rate in (
        ("control", "runtime.control_hz", 200.0),
        ("emg", "runtime.emg_hz", 200.0),
        ("decision", "runtime.decision_hz", 100.0),
        ("vision", "runtime.vision_hz", 20.0),
    ):
        rate = config.get_float(rate_key, default_rate)
        timeout = config.get_float(f"safety.watchdogs.{name}_s", 0.0)
        if rate <= 0 or timeout <= 0:
            continue
        period = 1.0 / rate
        if timeout < period * 2.0:
            issues.append(
                ValidationIssue(
                    f"safety.watchdogs.{name}_s",
                    f"{timeout:.3f} s is less than two {name} periods ({period * 2:.3f} s)",
                    ValidationSeverity.ERROR,
                    remedy=f"raise it above {period * 2:.3f} s or raise {rate_key}",
                )
            )

    # The firmware watchdog must outlast a control period, or a healthy host
    # still looks like a dead one to the controller.
    watchdog_ms = config.get_float("servo.watchdog_ms", 300.0)
    if control_period > 0 and watchdog_ms / 1000.0 < control_period * 3.0:
        issues.append(
            ValidationIssue(
                "servo.watchdog_ms",
                f"{watchdog_ms:.0f} ms is less than three control periods "
                f"({control_period * 3000:.0f} ms)",
                ValidationSeverity.ERROR,
                remedy="the firmware would safe the drive during normal operation",
            )
        )

    # Host limits above the actuator's are not limits at all.
    host_v = config.get_float("control.max_velocity", 1.8)
    servo_v = config.get_float("servo.max_velocity", 2.0)
    if host_v > servo_v:
        issues.append(
            ValidationIssue(
                "control.max_velocity",
                f"{host_v} exceeds servo.max_velocity ({servo_v})",
                ValidationSeverity.WARNING,
                remedy="the actuator limit wins, so this value is misleading",
            )
        )

    host_a = config.get_float("control.max_acceleration", 7.0)
    servo_a = config.get_float("servo.max_acceleration", 8.0)
    if host_a > servo_a:
        issues.append(
            ValidationIssue(
                "control.max_acceleration",
                f"{host_a} exceeds servo.max_acceleration ({servo_a})",
                ValidationSeverity.WARNING,
                remedy="the actuator limit wins, so this value is misleading",
            )
        )

    # Hysteresis the wrong way round makes activation chatter at the threshold.
    onset = config.get_float("emg.onset_threshold", 0.22)
    offset = config.get_float("emg.offset_threshold", 0.12)
    if offset >= onset:
        issues.append(
            ValidationIssue(
                "emg.offset_threshold",
                f"{offset} is not below emg.onset_threshold ({onset})",
                ValidationSeverity.ERROR,
                remedy="the release threshold must be lower than the activation threshold",
            )
        )

    # Nyquist. A band edge above half the sample rate is not a filter.
    sample_rate = config.get_float("emg.sample_rate_hz", 1000.0)
    band_high = config.get_float("emg.band_high_hz", 400.0)
    if band_high >= sample_rate / 2.0:
        issues.append(
            ValidationIssue(
                "emg.band_high_hz",
                f"{band_high} Hz is at or above Nyquist for "
                f"{sample_rate:.0f} Hz sampling ({sample_rate / 2:.0f} Hz)",
                ValidationSeverity.ERROR,
                remedy="lower the band edge or raise emg.sample_rate_hz",
            )
        )
    band_low = config.get_float("emg.band_low_hz", 20.0)
    if band_low >= band_high:
        issues.append(
            ValidationIssue(
                "emg.band_low_hz",
                f"{band_low} Hz is not below emg.band_high_hz ({band_high} Hz)",
                ValidationSeverity.ERROR,
            )
        )

    # An intent may not authorise motion for longer than the EMG watchdog would
    # take to notice the signal has stopped arriving.
    max_age = config.get_float("fusion.max_intent_age_s", 0.35)
    emg_watchdog = config.get_float("safety.watchdogs.emg_s", 0.3)
    if max_age > emg_watchdog * 2.0:
        issues.append(
            ValidationIssue(
                "fusion.max_intent_age_s",
                f"{max_age} s is more than twice the EMG watchdog ({emg_watchdog} s)",
                ValidationSeverity.WARNING,
                remedy="a stale intent could outlive the detection of a dead electrode",
            )
        )

    # Asking vision for more frames than the camera produces just adds latency.
    vision_hz = config.get_float("runtime.vision_hz", 20.0)
    camera_fps = config.get_float("camera.fps", 0.0)
    if camera_fps > 0 and vision_hz > camera_fps:
        issues.append(
            ValidationIssue(
                "runtime.vision_hz",
                f"{vision_hz} Hz exceeds the camera frame rate ({camera_fps} fps)",
                ValidationSeverity.WARNING,
                remedy="the pipeline will re-process frames it has already seen",
            )
        )

    return issues


def _unknown_keys(config: Config, known: Sequence[str]) -> list[ValidationIssue]:
    """Report configured paths that no rule mentions and no code reads.

    Warning-level by design. The known-path list is maintained by hand and will
    lag the code; treating a stale list as an error would make adding a config
    key a two-step change with a broken state in between.
    """
    known_set = set(known)
    issues: list[ValidationIssue] = []

    def walk(mapping: dict, prefix: str) -> None:
        for key, value in mapping.items():
            path = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                walk(value, path)
                continue
            if path in known_set:
                continue
            # A section whose whole subtree is user data (grip presets, object
            # affordances, per-finger endpoints) is keyed by name, so individual
            # paths cannot be enumerated. The exact-match arm covers arrays of
            # tables, which arrive here as a single list-valued leaf.
            if any(
                path == root or path.startswith(f"{root}.") for root in _FREEFORM_SECTIONS
            ):
                continue
            issues.append(
                ValidationIssue(
                    path,
                    "not a setting this build reads",
                    ValidationSeverity.WARNING,
                    remedy="check the spelling, or remove it",
                )
            )

    walk(config.as_dict(), "")
    return issues


#: Sections whose keys are user-defined names rather than fixed settings.
_FREEFORM_SECTIONS = (
    "grasps",
    "affordances",
    "servo.fingers",
    "emg.channels",
    "vision.mock",
    "vision.hggd_mcu",
    "vision.onnx_detector",
    "vision.anygrasp",
    "vision.replay",
    "camera.scene",
    "simulation",
    "training.exercises",
    "profiles",
)


def validate_config(
    config: Config,
    *,
    rules: Sequence[Rule] = SYSTEM_RULES,
    check_unknown: bool = True,
    extra_checks: Sequence[Callable[[Config], list[ValidationIssue]]] = (),
) -> ValidationReport:
    """Validate ``config`` and return every problem found.

    Collects all issues rather than failing on the first, so one run tells the
    user everything that needs fixing.
    """
    issues: list[ValidationIssue] = []
    for rule in rules:
        issue = rule.check(config)
        if issue is not None:
            issues.append(issue)

    issues.extend(_cross_checks(config))
    for check in extra_checks:
        issues.extend(check(config))

    if check_unknown:
        known = [rule.path for rule in rules] + list(_ADDITIONAL_KNOWN_PATHS)
        issues.extend(_unknown_keys(config, known))

    return ValidationReport(issues=tuple(issues))


#: Paths that are read by the code but carry no constraint worth a rule. Listed
#: so the unknown-key check does not flag them.
_ADDITIONAL_KNOWN_PATHS: tuple[str, ...] = (
    "system.name", "system.device_id",
    "hardware.simulate",
    "servo.port", "servo.rtscts", "servo.dtr_reset", "servo.required",
    "servo.state_timeout_s", "servo.emulator_latency_s", "servo.calibration_path",
    "servo.hand_id", "servo.reconnect", "servo.reconnect_backoff_s",
    "servo.reconnect_max_backoff_s", "servo.driver_board",
    "control.max_jerk",
    "emg.driver", "emg.port", "emg.baud", "emg.required", "emg.classifier",
    "emg.model_path", "emg.calibration_path", "emg.auto_recalibrate",
    "emg.co_contraction_threshold", "emg.cancel_dwell_s", "emg.release_s",
    "emg.envelope_attack_s", "emg.envelope_release_s", "emg.adc_reference_v",
    "emg.adc_bits", "emg.amplifier_gain", "emg.seed", "emg.recording",
    "emg.replay_loop", "emg.replay_speed",
    "camera.driver", "camera.device", "camera.width", "camera.height",
    "camera.fps", "camera.required", "camera.scene", "camera.enabled",
    "camera.pixel_format", "camera.autofocus",
    "vision.enabled", "vision.backend", "vision.tracker_iou",
    "vision.tracker_max_missed", "vision.recording",
    "ai.model_root", "ai.planner", "ai.planners", "ai.min_plan_confidence",
    "fusion.min_combined_confidence", "fusion.force_ceiling", "fusion.speed_ceiling",
    "fusion.max_vision_age_s", "fusion.require_vision",
    "modes.min_dwell_s",
    "runtime.diagnostics_hz",
    "safety.max_grip_force", "safety.battery_warning_pct",
    "safety.battery_critical_pct", "safety.max_cpu_temperature_c",
    "safety.watchdogs.ui_s",
    "power.node",
    "training.stats_path",
    "ui.enabled", "ui.renderer", "ui.width", "ui.height", "ui.fullscreen",
    "ui.accessibility.reduce_motion", "ui.accessibility.high_contrast",
    "ui.accessibility.show_numeric_values", "ui.accessibility.haptics",
    "ui.accessibility.message_duration_s", "ui.accessibility.audio_cues",
    "ui.profile_path",
    "logging.level", "logging.console", "logging.file", "logging.max_bytes",
    "logging.backups", "logging.buffer", "logging.quiet",
    "telemetry.enabled", "telemetry.path", "telemetry.blackbox",
    "telemetry.blackbox_dir",
    "system.state_path",
)
