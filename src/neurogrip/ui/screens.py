"""Screen definitions.

Each screen is a pure function from a :class:`ViewModel` to a
:class:`~neurogrip.ui.widgets.Scene`. No screen holds state, touches hardware or
commands the hand; interaction happens by publishing an action, which
:class:`~neurogrip.ui.app.UiService` routes.

Screens provided:

======================  ====================================================
route                   contents
======================  ====================================================
``dashboard``           the primary view: mode, hand, EMG, AI, camera, status
``settings``            mode, vision backend, force/speed, accessibility
``diagnostics``         health, loop timing, resources, link quality, self-test
``calibration``         the guided EMG calibration wizard
``training``            exercise selection, live exercise, results, statistics
``logs``                the on-device log viewer
``firmware``            firmware and software version, update status
``system``              hardware inventory, build info, licences
``accessibility``       font scale, contrast, motion, audio cues
======================  ====================================================
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..control.controller import HandState
from ..core.types import Finger, ModeId, clamp
from ..diagnostics.service import DiagnosticsSnapshot
from ..emg.calibration import CalibrationPhase, CalibrationProgress
from ..emg.pipeline import EmgFrame
from ..fusion.fusion import Decision
from ..safety.monitor import SafetyState
from ..training.exercises import EXERCISES, ExerciseState
from ..training.session import SessionSummary
from ..training.stats import TrainingStats
from ..vision.types import VisionResult
from .theme import Theme
from .widgets import (
    Badge,
    Bar,
    Button,
    Divider,
    Gauge,
    HandGraphic,
    Label,
    ListView,
    ProgressRing,
    Row,
    Scene,
    Sparkline,
    Toggle,
    panel,
    row,
)

__all__ = ["ROUTES", "SCREENS", "ViewModel", "build_scene"]


@dataclass(slots=True)
class ViewModel:
    """Everything the screens read. Assembled once per UI frame."""

    theme: Theme
    route: str = "dashboard"
    now: float = 0.0
    wall_time: float = 0.0

    mode: ModeId | None = None
    mode_title: str = ""
    mode_subtitle: str = ""
    mode_notes: tuple[str, ...] = field(default_factory=tuple)
    ai_enabled: bool = False

    hand: HandState | None = None
    emg: EmgFrame | None = None
    vision: VisionResult | None = None
    decision: Decision | None = None
    safety: SafetyState | None = None
    diagnostics: DiagnosticsSnapshot | None = None
    calibration: CalibrationProgress | None = None

    training_state: ExerciseState | None = None
    training_summary: SessionSummary | None = None
    training_stats: TrainingStats | None = None
    training_exercise: str = ""

    #: Rolling EMG history for the sparklines, per channel.
    emg_history: tuple[tuple[float, ...], ...] = field(default_factory=tuple)
    #: Rolling AI-confidence history.
    confidence_history: tuple[float, ...] = field(default_factory=tuple)

    logs: tuple[tuple[str, str, str], ...] = field(default_factory=tuple)
    toast: str = ""
    firmware_version: str = ""
    software_version: str = ""
    device_info: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    battery_percent: float = 100.0
    battery_charging: bool = False
    wifi_connected: bool = False
    wifi_ssid: str = ""
    bluetooth_connected: bool = False
    camera_connected: bool = False


ROUTES = (
    "dashboard",
    "settings",
    "diagnostics",
    "calibration",
    "training",
    "logs",
    "firmware",
    "system",
    "accessibility",
)


# ---------------------------------------------------------------------------
# Shared chrome
# ---------------------------------------------------------------------------


def _status_bar(vm: ViewModel) -> tuple[tuple[str, str, str], ...]:
    """Clock, battery, connectivity — the persistent top strip."""
    clock = time.strftime("%H:%M", time.localtime(vm.wall_time or time.time()))
    battery_icon = "⚡" if vm.battery_charging else "▮"
    battery_colour = (
        "danger" if vm.battery_percent <= 15 else "warning" if vm.battery_percent <= 30 else "ok"
    )
    return (
        ("🕐", clock, "text"),
        (battery_icon, f"{vm.battery_percent:.0f}%", battery_colour),
        ("📶", vm.wifi_ssid or "—", "ok" if vm.wifi_connected else "muted"),
        ("ᛒ", "on" if vm.bluetooth_connected else "off", "ok" if vm.bluetooth_connected else "muted"),
        ("📷", "on" if vm.camera_connected else "off", "ok" if vm.camera_connected else "muted"),
    )


def _banner(vm: ViewModel) -> Badge | None:
    """The sticky top banner. Safety first, then the AI-disabled notice."""
    safety = vm.safety
    if safety is not None and safety.estop_engaged:
        return Badge(
            key="banner",
            text="EMERGENCY STOP — tap Acknowledge to reset",
            colour="danger",
            icon="⛔",
        )
    if safety is not None and not safety.motion_allowed:
        return Badge(
            key="banner", text=f"MOTION BLOCKED — {safety.primary_reason}", colour="danger", icon="⚠"
        )
    if safety is not None and not safety.ai_allowed and vm.ai_enabled:
        return Badge(
            key="banner",
            text=f"AI ASSISTANCE UNAVAILABLE — {safety.primary_reason}",
            colour="warning",
            icon="⚠",
        )
    if not vm.ai_enabled and vm.mode is not None:
        # Required by the specification: manual mode must state clearly that the
        # AI is not involved.
        return Badge(key="banner", text="AI DISABLED — direct control", colour="neutral", icon="○")
    return None


def _nav(vm: ViewModel) -> tuple[Button, ...]:
    items = (
        ("dashboard", "Home", "⌂"),
        ("training", "Train", "🎯"),
        ("diagnostics", "Health", "❤"),
        ("settings", "Settings", "⚙"),
    )
    return tuple(
        Button(
            key=f"nav.{route}",
            text=label,
            icon=icon,
            action="navigate",
            args=(route,),
            variant="primary" if vm.route == route else "ghost",
        )
        for route, label, icon in items
    )


def _scene(vm: ViewModel, title: str, *children, extra_nav: tuple[Button, ...] = ()) -> Scene:
    return Scene(
        title=title,
        route=vm.route,
        children=tuple(c for c in children if c is not None),
        nav=_nav(vm) + extra_nav,
        banner=_banner(vm),
        toast=vm.toast,
        status=_status_bar(vm),
    )


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------


def dashboard(vm: ViewModel) -> Scene:
    """The primary screen."""
    hand = vm.hand
    emg = vm.emg
    decision = vm.decision

    mode_panel = panel(
        "Mode",
        Label(
            key="mode.title",
            text=vm.mode_title or (vm.mode.label if vm.mode else "—"),
            role="display",
            colour="ai" if vm.ai_enabled else "neutral",
            bold=True,
        ),
        Label(key="mode.subtitle", text=vm.mode_subtitle, role="caption", colour="muted"),
        row(
            *(
                Button(
                    key=f"mode.{m.value}",
                    text=m.label,
                    action="set_mode",
                    args=(m.value,),
                    variant="primary" if m is vm.mode else "secondary",
                )
                for m in (ModeId.MANUAL, ModeId.AI_ASSIST, ModeId.SPORTS, ModeId.TRAINING)
            ),
            gap=1,
        ),
        colour="ai" if vm.ai_enabled else "neutral",
    )

    if hand is not None:
        hand_visual = HandGraphic(
            key="hand.graphic",
            pose=hand.pose,
            target=hand.goal,
            contacts=tuple(int(f) for f in (hand.grip.contacts if hand.grip else ())),
            ai_driven=bool(decision and decision.ai_contributed),
            label=hand.activity,
        )
    else:
        hand_visual = Label(key="hand.graphic", text="waiting for hand telemetry", colour="muted")

    hand_panel = panel(
        "Hand",
        hand_visual,
        row(
            *(
                Bar(
                    key=f"finger.{finger.name.lower()}",
                    value=hand.pose[finger] if hand else 0.0,
                    label=finger.label[:3],
                    colour="ai" if decision and decision.ai_contributed else "user",
                    show_value=vm.theme.accessibility.show_numeric_values,
                )
                for finger in Finger
            )
        ),
        Label(
            key="hand.status",
            text=(
                f"{'Holding' if hand and hand.holding else 'Open' if hand and hand.is_open else 'Ready'}"
                f" · {hand.total_current_ma if hand else 0} mA"
                f" · {hand.temperature_c if hand else 0:.0f} °C"
            ),
            role="caption",
            colour="muted",
        ),
    )

    emg_panel = panel(
        "Muscle activity",
        *(
            Bar(
                key=f"emg.{channel.role}",
                value=channel.activation,
                label=channel.name,
                colour="user",
                threshold=0.8,
                show_value=vm.theme.accessibility.show_numeric_values,
            )
            for channel in (emg.channels if emg else ())
        ),
        *(
            Sparkline(
                key=f"emg.trace.{index}",
                values=series,
                colour="user",
                label=emg.channels[index].name if emg and index < len(emg.channels) else "",
            )
            for index, series in enumerate(vm.emg_history)
        ),
        Row(
            key="emg.quality",
            children=(
                Badge(
                    key="emg.quality.badge",
                    text=f"Signal: {emg.quality.label}" if emg else "Signal: —",
                    colour=(
                        "ok"
                        if emg and emg.quality.allows_ai
                        else "warning"
                        if emg
                        else "muted"
                    ),
                    icon="◉",
                ),
                Badge(
                    key="emg.intent.badge",
                    text=f"Intent: {decision.action.value}" if decision else "Intent: —",
                    colour="ai" if decision and decision.ai_contributed else "user",
                    icon="→",
                ),
            ),
        ),
    )

    ai_panel = _ai_panel(vm)

    return _scene(
        vm,
        "Dashboard",
        mode_panel,
        hand_panel,
        emg_panel,
        ai_panel,
        _safety_panel(vm),
        extra_nav=(
            Button(
                key="action.estop",
                text="STOP",
                icon="⛔",
                action="estop",
                variant="danger",
                colour="danger",
            ),
        ),
    )


def _ai_panel(vm: ViewModel):
    """AI confidence, current plan and camera preview state."""
    decision = vm.decision
    vision = vm.vision

    if not vm.ai_enabled:
        return panel(
            "AI assistance",
            Label(
                key="ai.disabled",
                text="Disabled in this mode",
                colour="neutral",
                icon="○",
            ),
            *(Label(text=note, role="caption", colour="muted") for note in vm.mode_notes),
            colour="neutral",
        )

    detections = vision.detections if vision else ()
    primary = vision.primary if vision else None

    return panel(
        "AI assistance",
        Gauge(
            key="ai.confidence",
            value=decision.confidence if decision else 0.0,
            label="Confidence",
            colour="ai",
            unit="%",
            caption=decision.explain() if decision else "waiting for intent",
        ),
        Sparkline(
            key="ai.confidence.trace",
            values=vm.confidence_history,
            colour="ai",
            minimum=0.0,
            maximum=1.0,
            reference=0.55,
            label="confidence",
        ),
        Label(
            key="ai.object",
            text=(
                f"Sees: {primary.label} ({primary.confidence * 100:.0f}%)"
                if primary
                else "Sees: nothing recognised"
            ),
            icon="👁",
            colour="ai" if primary else "muted",
        ),
        Label(
            key="ai.grasp",
            text=(
                f"Plan: {decision.plan.grasp.label}" if decision and decision.plan else "Plan: —"
            ),
            icon="✋",
            colour="ai" if decision and decision.plan else "muted",
        ),
        ListView(
            key="ai.reasons",
            rows=tuple(
                ("·", reason, "muted", "") for reason in (decision.reasons[:4] if decision else ())
            ),
            empty_text="No decision yet",
            max_visible=4,
        ),
        ListView(
            key="ai.detections",
            rows=tuple(
                (d.label, f"{d.confidence * 100:.0f}%  track {d.track_id}", "ai", "▣")
                for d in detections[:4]
            ),
            empty_text="No objects detected",
            max_visible=4,
        ),
        colour="ai",
    )


def _safety_panel(vm: ViewModel):
    safety = vm.safety
    if safety is None:
        return None
    if safety.is_nominal:
        return panel(
            "Safety",
            Badge(key="safety.ok", text="All systems nominal", colour="ok", icon="✓"),
        )
    return panel(
        "Safety",
        *(
            Badge(
                key=f"safety.fault.{fault.code}",
                text=fault.message,
                colour="danger" if fault.severity >= 40 else "warning",
                icon="⚠",
            )
            for fault in safety.faults[:4]
        ),
        *(Label(text=remedy, role="caption", colour="muted") for remedy in safety.remedies[:2]),
        Button(
            key="safety.ack",
            text="Acknowledge",
            action="acknowledge",
            variant="secondary",
            confirm=True,
        ),
        emphasised=True,
        colour="danger",
    )


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


def settings(vm: ViewModel) -> Scene:
    return _scene(
        vm,
        "Settings",
        panel(
            "Mode",
            *(
                Button(
                    key=f"settings.mode.{m.value}",
                    text=f"{m.label}",
                    action="set_mode",
                    args=(m.value,),
                    variant="primary" if m is vm.mode else "secondary",
                )
                for m in ModeId
            ),
        ),
        panel(
            "Appearance",
            Button(key="settings.theme.dark", text="Dark", action="set_theme", args=("dark",), variant="secondary"),
            Button(key="settings.theme.light", text="Light", action="set_theme", args=("light",), variant="secondary"),
            Button(
                key="settings.theme.contrast",
                text="High contrast",
                action="set_theme",
                args=("high_contrast",),
                variant="secondary",
            ),
        ),
        panel(
            "Calibration",
            Label(text="Recalibrate EMG for the current electrode placement.", role="caption", colour="muted"),
            Button(key="settings.calibrate", text="Run calibration wizard", action="navigate", args=("calibration",)),
        ),
        panel(
            "More",
            Button(key="settings.accessibility", text="Accessibility", action="navigate", args=("accessibility",), variant="secondary"),
            Button(key="settings.logs", text="Logs", action="navigate", args=("logs",), variant="secondary"),
            Button(key="settings.firmware", text="Firmware & updates", action="navigate", args=("firmware",), variant="secondary"),
            Button(key="settings.system", text="System information", action="navigate", args=("system",), variant="secondary"),
        ),
    )


def accessibility(vm: ViewModel) -> Scene:
    access = vm.theme.accessibility
    return _scene(
        vm,
        "Accessibility",
        panel(
            "Text size",
            Label(text=f"Current: {access.font_scale:.2f}×", role="title"),
            row(
                Button(key="a11y.font.down", text="Smaller", action="font_scale", args=("-",), variant="secondary"),
                Button(key="a11y.font.reset", text="Reset", action="font_scale", args=("1.0",), variant="ghost"),
                Button(key="a11y.font.up", text="Larger", action="font_scale", args=("+",), variant="secondary"),
            ),
        ),
        panel(
            "Display",
            Toggle(
                key="a11y.contrast",
                text="High contrast",
                value=access.high_contrast,
                action="toggle_contrast",
                caption="Maximum contrast for bright sunlight and low vision.",
            ),
            Toggle(
                key="a11y.motion",
                text="Reduce motion",
                value=access.reduce_motion,
                action="toggle_motion",
                caption="Disables animations and transitions.",
            ),
            Toggle(
                key="a11y.values",
                text="Show numeric values",
                value=access.show_numeric_values,
                action="toggle_values",
                caption="Prints exact numbers next to every bar and gauge.",
            ),
        ),
        panel(
            "Feedback",
            Toggle(
                key="a11y.haptics",
                text="Haptic feedback",
                value=access.haptics,
                action="toggle_haptics",
                caption="Vibrate on touch, where supported.",
            ),
            Toggle(
                key="a11y.audio",
                text="Audio cues",
                value=access.audio_cues,
                action="toggle_audio",
                caption="Announce mode changes, grips and alerts.",
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def diagnostics(vm: ViewModel) -> Scene:
    snapshot = vm.diagnostics
    if snapshot is None:
        return _scene(vm, "Diagnostics", Label(text="Collecting data…", colour="muted"))

    health_rows = tuple(
        (
            report.name,
            report.detail or report.status.label,
            {"OK": "ok", "Degraded": "warning", "Failed": "danger", "Offline": "muted"}.get(
                report.status.label, "muted"
            ),
            {"OK": "✓", "Degraded": "!", "Failed": "✗", "Offline": "–"}.get(report.status.label, "·"),
        )
        for report in snapshot.health
    )

    loop_rows = tuple(
        (
            loop.name,
            f"{loop.actual_hz:.0f}/{loop.target_hz:.0f} Hz  "
            f"jitter {loop.jitter_ms:.2f} ms  overruns {loop.overruns}",
            "ok" if loop.healthy else "warning",
            "✓" if loop.healthy else "!",
        )
        for loop in snapshot.loops
    )

    selftest = snapshot.selftest
    selftest_rows = (
        tuple(
            (
                result.name,
                result.message,
                {"pass": "ok", "warn": "warning", "fail": "danger", "skip": "muted"}[
                    result.outcome.value
                ],
                result.outcome.symbol,
            )
            for result in selftest.results
        )
        if selftest
        else ()
    )

    system = snapshot.system
    return _scene(
        vm,
        "Diagnostics",
        panel(
            "Health",
            ListView(key="diag.health", rows=health_rows, empty_text="No services registered"),
        ),
        panel(
            "Resources",
            Bar(key="diag.cpu", value=system.cpu_percent / 100.0, label="CPU", colour="primary", threshold=0.85),
            Bar(key="diag.memory", value=system.memory_percent / 100.0, label="Memory", colour="primary", threshold=0.85),
            Bar(
                key="diag.temperature",
                value=clamp(system.cpu_temperature_c / 100.0),
                label="SoC temp",
                colour="warning",
                unit="",
                threshold=0.80,
            ),
            Label(
                key="diag.resource.detail",
                text=(
                    f"{system.cpu_temperature_c:.0f} °C · "
                    f"{system.memory_used_mb:.0f}/{system.memory_total_mb:.0f} MB · "
                    f"load {system.load_average[0]:.2f} · "
                    f"up {system.uptime_s / 3600:.1f} h"
                ),
                role="caption",
                colour="muted",
            ),
        ),
        panel("Loop timing", ListView(key="diag.loops", rows=loop_rows, monospace=True)),
        panel(
            "Self-test",
            ListView(key="diag.selftest", rows=selftest_rows, empty_text="Not run yet"),
            Label(
                key="diag.selftest.summary",
                text=selftest.summary() if selftest else "",
                role="caption",
                colour="muted",
            ),
            row(
                Button(key="diag.selftest.run", text="Run self-test", action="selftest"),
                Button(
                    key="diag.selftest.motion",
                    text="Run with motion",
                    action="selftest_motion",
                    variant="secondary",
                    confirm=True,
                ),
            ),
        ),
        panel(
            "Battery",
            Gauge(
                key="diag.battery",
                value=snapshot.battery.percentage / 100.0,
                label="Charge",
                colour="ok" if snapshot.battery.percentage > 30 else "warning",
                unit="%",
                caption=(
                    f"{snapshot.battery.voltage_v:.2f} V · "
                    f"{snapshot.battery.current_ma:.0f} mA"
                    + (" · charging" if snapshot.battery.charging else "")
                ),
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def calibration(vm: ViewModel) -> Scene:
    progress = vm.calibration
    if progress is None or progress.phase is CalibrationPhase.IDLE:
        return _scene(
            vm,
            "Calibration",
            panel(
                "EMG calibration",
                Label(
                    text="Calibration teaches the hand what your muscles look like at rest "
                    "and at full effort. It takes about 30 seconds.",
                    colour="muted",
                ),
                Label(text="Sit comfortably with your arm supported.", role="caption", colour="muted"),
                Button(key="cal.start", text="Start calibration", action="calibrate_start"),
            ),
        )

    if progress.phase is CalibrationPhase.COMPLETE:
        return _scene(
            vm,
            "Calibration",
            panel(
                "Complete",
                Badge(key="cal.done", text="Calibration saved", colour="ok", icon="✓"),
                Label(text=progress.message, colour="muted"),
                Button(key="cal.back", text="Done", action="navigate", args=("dashboard",)),
            ),
        )

    if progress.phase is CalibrationPhase.FAILED:
        return _scene(
            vm,
            "Calibration",
            panel(
                "Calibration failed",
                Badge(key="cal.failed", text="Could not calibrate", colour="danger", icon="✗"),
                Label(text=progress.message, colour="warning"),
                Label(
                    text="Check that the electrodes are firmly attached and the skin is clean.",
                    role="caption",
                    colour="muted",
                ),
                Button(key="cal.retry", text="Try again", action="calibrate_start"),
                emphasised=True,
                colour="danger",
            ),
        )

    return _scene(
        vm,
        "Calibration",
        panel(
            f"Step {progress.step_index + 1} of {progress.step_count}: {progress.title}",
            ProgressRing(
                key="cal.progress",
                value=progress.fraction,
                label=f"{max(0.0, progress.duration - progress.elapsed):.0f}s",
                caption=progress.title,
                colour="primary",
            ),
            Label(key="cal.instruction", text=progress.instruction, role="title"),
            *(
                Bar(key=f"cal.level.{i}", value=clamp(level * 4000), label=f"Ch{i}", colour="user")
                for i, level in enumerate(progress.levels)
            ),
            Button(key="cal.cancel", text="Cancel", action="calibrate_cancel", variant="ghost"),
        ),
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def training(vm: ViewModel) -> Scene:
    state = vm.training_state
    summary = vm.training_summary
    stats = vm.training_stats

    if state is not None:
        return _scene(
            vm,
            "Training",
            panel(
                vm.training_exercise.title() or "Exercise",
                Label(key="train.prompt", text=state.prompt, role="display", bold=True),
                ProgressRing(
                    key="train.progress",
                    value=state.progress,
                    label=f"{state.trial + 1}/{state.trials_total}",
                    caption=state.phase,
                    colour="success",
                ),
                Bar(
                    key="train.level",
                    value=state.actual,
                    target=state.target if state.target > 0 else None,
                    label="Your effort",
                    colour="user",
                ),
                Label(key="train.feedback", text=state.feedback, colour="ok", role="title"),
                Label(
                    key="train.score",
                    text=f"Score {state.score * 100:.0f}%",
                    role="caption",
                    colour="muted",
                ),
                Button(key="train.stop", text="Stop", action="training_stop", variant="ghost"),
            ),
        )

    if summary is not None:
        return _scene(
            vm,
            "Training",
            panel(
                "Session complete",
                Label(key="train.stars", text="★" * summary.stars + "☆" * (3 - summary.stars), role="display"),
                Label(key="train.result", text=f"{summary.mean_score * 100:.0f}%  ·  {summary.trials} trials", role="title"),
                Label(key="train.advice", text=summary.advice, colour="muted"),
                *(
                    [Badge(key="train.promoted", text=f"Promoted to {summary.next_difficulty.label}", colour="ok", icon="⬆")]
                    if summary.promoted
                    else []
                ),
                *(
                    [Badge(key="train.demoted", text=f"Moved back to {summary.next_difficulty.label}", colour="warning", icon="⬇")]
                    if summary.demoted
                    else []
                ),
                *(
                    Badge(key=f"train.achievement.{i}", text=name, colour="ai", icon="🏆")
                    for i, name in enumerate(summary.achievements)
                ),
                Button(key="train.again", text="Train again", action="navigate", args=("training",)),
            ),
        )

    exercise_rows = tuple(
        (
            cls.title,
            f"{cls.description}  ·  "
            + (
                f"{stats.progress(key).difficulty.label}, best "
                f"{stats.progress(key).best_score * 100:.0f}% "
                f"{stats.progress(key).trend.symbol}"
                if stats
                else "not attempted"
            ),
            "ok",
            "🎯",
        )
        for key, cls in EXERCISES.items()
    )

    return _scene(
        vm,
        "Training",
        panel(
            "Choose an exercise",
            ListView(key="train.list", rows=exercise_rows),
            *(
                Button(
                    key=f"train.start.{key}",
                    text=cls.title,
                    action="training_start",
                    args=(key,),
                    variant="secondary",
                )
                for key, cls in EXERCISES.items()
            ),
        ),
        panel(
            "Your progress",
            *(
                [
                    Label(
                        key="train.stats",
                        text=(
                            f"{stats.total_sessions} sessions · {stats.total_trials} trials · "
                            f"{stats.total_time_s / 60:.0f} min · {stats.streak_days}-day streak"
                        ),
                    ),
                    Gauge(
                        key="train.mastery",
                        value=stats.overall_mastery,
                        label="Overall mastery",
                        colour="success",
                        unit="%",
                    ),
                ]
                if stats
                else [Label(text="No sessions yet — pick an exercise above.", colour="muted")]
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Logs, firmware, system information
# ---------------------------------------------------------------------------


def logs(vm: ViewModel) -> Scene:
    return _scene(
        vm,
        "Logs",
        panel(
            "Recent log records",
            ListView(
                key="logs.list",
                rows=tuple((level, message, colour, "") for level, message, colour in vm.logs),
                empty_text="No log records",
                monospace=True,
                max_visible=20,
            ),
            row(
                Button(key="logs.refresh", text="Refresh", action="refresh_logs", variant="secondary"),
                Button(key="logs.export", text="Export", action="export_logs", variant="ghost"),
            ),
        ),
    )


def firmware(vm: ViewModel) -> Scene:
    return _scene(
        vm,
        "Firmware & updates",
        panel(
            "Versions",
            Label(key="fw.software", text=f"Software: {vm.software_version or 'unknown'}"),
            Label(key="fw.firmware", text=f"Motor controller: {vm.firmware_version or 'not detected'}"),
        ),
        panel(
            "Updates",
            Label(
                text="Updates are applied from a USB drive or over the network when "
                "the hand is not in use.",
                role="caption",
                colour="muted",
            ),
            # TODO(updates): the update transport is not implemented. The button
            # is present so the flow and its confirmations can be designed and
            # reviewed before any code can install anything onto a worn device.
            Button(key="fw.check", text="Check for updates", action="check_updates", variant="secondary"),
            Badge(key="fw.status", text="Update service not configured", colour="muted", icon="ⓘ"),
        ),
    )


def system(vm: ViewModel) -> Scene:
    return _scene(
        vm,
        "System information",
        panel(
            "Device",
            ListView(
                key="sys.info",
                rows=tuple((name, value, "muted", "") for name, value in vm.device_info),
                empty_text="No device information",
                monospace=True,
                max_visible=16,
            ),
        ),
        panel(
            "About",
            Label(text="NeuroGrip — AI-assisted prosthetic hand", role="title"),
            Label(text=f"Version {vm.software_version}", role="caption", colour="muted"),
            Divider(),
            Label(
                text="Research prototype. Not a medical device and not certified "
                "for clinical use.",
                role="caption",
                colour="warning",
            ),
        ),
    )


#: Route name → screen builder.
SCREENS = {
    "dashboard": dashboard,
    "settings": settings,
    "diagnostics": diagnostics,
    "calibration": calibration,
    "training": training,
    "logs": logs,
    "firmware": firmware,
    "system": system,
    "accessibility": accessibility,
}


def build_scene(vm: ViewModel) -> Scene:
    """Render the current route. Unknown routes fall back to the dashboard."""
    builder = SCREENS.get(vm.route, dashboard)
    return builder(vm)
