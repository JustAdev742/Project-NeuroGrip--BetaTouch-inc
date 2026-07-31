"""UI service: state assembly, navigation and action routing.

Runs in its own rate group at 15–20 Hz. The UI never reaches into a subsystem to
change something; it publishes an action, and this service translates actions
into calls on the appropriate service. That keeps the interface replaceable and
keeps every state change auditable in one place.

Two properties worth stating explicitly:

* **The UI cannot command motion directly.** Grip buttons submit through
  :class:`~neurogrip.control.controller.HandController` at ``USER_DIRECT``
  priority, so they are subject to the same limits, e-stop and preemption as
  everything else.
* **The UI never blocks the control loop.** It reads immutable snapshots that
  other services published; it holds no locks that control code waits on.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from ..core.clock import Clock
from ..core.events import Event, EventBus
from ..core.lifecycle import HealthReport, ServiceBase
from ..core.logging import get_logger, log_buffer
from ..core.topics import Topics
from ..core.types import GraspType, ModeId, clamp
from .renderers import Renderer, UiEvent
from .screens import ROUTES, ViewModel, build_scene
from .theme import Theme, ThemeMode
from .widgets import Scene

__all__ = ["UiService"]

log = get_logger(__name__)


@dataclass(slots=True)
class _Toast:
    text: str
    until: float
    level: str = "info"


class UiService(ServiceBase):
    """Owns the screen state and routes user actions."""

    service_name = "ui"

    #: Number of EMG samples retained per channel for the dashboard sparklines.
    TRACE_LENGTH = 60

    def __init__(
        self,
        renderer: Renderer,
        clock: Clock,
        bus: EventBus,
        theme: Theme,
        *,
        controller=None,
        modes=None,
        safety=None,
        vision=None,
        diagnostics=None,
        training=None,
        calibration=None,
        servo_calibration=None,
        profiles=None,
        software_version: str = "",
    ) -> None:
        super().__init__()
        self._renderer = renderer
        self._clock = clock
        self._bus = bus
        self._theme = theme
        self._controller = controller
        self._modes = modes
        #: Optional :class:`~neurogrip.core.profiles.ProfileStore`. When present,
        #: preference changes are written through immediately rather than at
        #: shutdown — the device may lose power at any moment, and a setting the
        #: user chose and then lost is worse than one they never had.
        self._profiles = profiles
        self._servo_calibration = servo_calibration
        self._safety = safety
        self._vision = vision
        self._diagnostics = diagnostics
        self._training = training
        self._calibration = calibration
        self._version = software_version

        self._route = "dashboard"
        self._route_stack: list[str] = []
        self._toast: _Toast | None = None
        self._emg_traces: list[deque[float]] = []
        self._confidence_trace: deque[float] = deque(maxlen=self.TRACE_LENGTH)
        self._latest_emg = None
        self._latest_scene: Scene | None = None
        self._device_info: tuple[tuple[str, str], ...] = ()
        self.frames = 0
        self.actions = 0

        self._subscriptions = [
            bus.subscribe(Topics.EMG_FRAME, self._on_emg, name="ui.emg"),
            bus.subscribe(Topics.DECISION_MADE, self._on_decision, name="ui.decision"),
            bus.subscribe(Topics.UI_NOTIFICATION, self._on_notification, name="ui.notify"),
            bus.subscribe(Topics.MODE_CHANGED, self._on_mode_changed, name="ui.mode"),
            bus.subscribe(Topics.TRAINING_ACHIEVEMENT, self._on_achievement, name="ui.achievement"),
        ]

    # -- lifecycle ------------------------------------------------------------

    def on_start(self) -> None:
        self._renderer.start()
        log.info("UI started", renderer=type(self._renderer).__name__, route=self._route)

    def on_stop(self) -> None:
        for subscription in self._subscriptions:
            subscription.cancel()
        self._renderer.stop()

    def set_device_info(self, info: tuple[tuple[str, str], ...]) -> None:
        """Populate the System Information screen."""
        self._device_info = info

    # -- event handlers -------------------------------------------------------

    def _on_emg(self, event: Event) -> None:
        frame = event.payload
        self._latest_emg = frame
        if len(self._emg_traces) != len(frame.channels):
            self._emg_traces = [deque(maxlen=self.TRACE_LENGTH) for _ in frame.channels]
        for index, channel in enumerate(frame.channels):
            self._emg_traces[index].append(channel.activation)

    def _on_decision(self, event: Event) -> None:
        decision = event.payload
        if decision is not None:
            self._confidence_trace.append(getattr(decision, "confidence", 0.0))

    def _on_notification(self, event: Event) -> None:
        payload = event.payload or {}
        self._show_toast(str(payload.get("text", "")), str(payload.get("level", "info")))

    def _on_mode_changed(self, event: Event) -> None:
        change = event.payload
        mode = getattr(change, "current", None)
        if mode is not None:
            self._show_toast(f"{mode.label} mode", "info")

    def _on_achievement(self, event: Event) -> None:
        payload = event.payload or {}
        self._show_toast(f"🏆 {payload.get('title', 'Achievement unlocked')}", "success")

    def _show_toast(self, text: str, level: str = "info") -> None:
        if not text:
            return
        self._toast = _Toast(
            text=text,
            until=self._clock.monotonic() + self._theme.accessibility.message_duration_s,
            level=level,
        )

    # -- navigation -----------------------------------------------------------

    @property
    def route(self) -> str:
        return self._route

    def navigate(self, route: str) -> None:
        if route not in ROUTES:
            log.warning("unknown UI route", route=route)
            return
        if route != self._route:
            self._route_stack.append(self._route)
            del self._route_stack[:-8]
            self._route = route
            self._bus.publish(Topics.UI_NAVIGATE, {"route": route}, source=self.name)

    def back(self) -> None:
        if self._route_stack:
            self._route = self._route_stack.pop()

    @property
    def theme(self) -> Theme:
        return self._theme

    # -- frame ----------------------------------------------------------------

    def tick(self) -> Scene:
        """Handle input, build the view model, render one frame."""
        for event in self._renderer.poll_events():
            self.handle_action(event)

        scene = build_scene(self._build_view_model())
        self._renderer.render(scene, self._theme)
        self._latest_scene = scene
        self.frames += 1
        return scene

    @property
    def scene(self) -> Scene | None:
        """The most recently rendered scene; used by tests."""
        return self._latest_scene

    def _build_view_model(self) -> ViewModel:
        now = self._clock.monotonic()
        if self._toast is not None and now > self._toast.until:
            self._toast = None

        mode_id = self._modes.current if self._modes else None
        active_mode = self._modes.active if self._modes else None
        profile = active_mode.profile if active_mode else None
        decision = active_mode.last_decision if active_mode else None

        snapshot = self._diagnostics.snapshot if self._diagnostics else None
        battery = snapshot.battery if snapshot else None
        connectivity = snapshot.connectivity if snapshot else None

        return ViewModel(
            theme=self._theme,
            route=self._route,
            now=now,
            wall_time=self._clock.wall(),
            mode=mode_id,
            mode_title=profile.title if profile else "",
            mode_subtitle=profile.subtitle if profile else "",
            mode_notes=profile.notes if profile else (),
            ai_enabled=bool(profile and profile.policy.ai_enabled),
            hand=self._controller.state if self._controller else None,
            emg=self._latest_emg,
            vision=self._vision.latest if self._vision else None,
            decision=decision,
            safety=self._safety.state if self._safety else None,
            diagnostics=snapshot,
            calibration=self._calibration.progress() if self._calibration else None,
            training_state=self._training.state if self._training else None,
            training_summary=self._training.summary if self._training else None,
            training_stats=self._training.stats if self._training else None,
            training_exercise=(
                self._training.exercise.key
                if self._training and self._training.exercise
                else ""
            ),
            emg_history=tuple(tuple(trace) for trace in self._emg_traces),
            confidence_history=tuple(self._confidence_trace),
            logs=self._log_rows(),
            toast=self._toast.text if self._toast else "",
            firmware_version=self._firmware_version(),
            software_version=self._version,
            device_info=self._device_info,
            battery_percent=battery.percentage if battery else 0.0,
            battery_charging=battery.charging if battery else False,
            wifi_connected=connectivity.wifi_connected if connectivity else False,
            wifi_ssid=connectivity.wifi_ssid if connectivity else "",
            bluetooth_connected=connectivity.bluetooth_connected if connectivity else False,
            camera_connected=bool(self._vision and self._vision.has_camera),
        )

    def _firmware_version(self) -> str:
        if self._controller is None:
            return ""
        try:
            return self._controller._servo.info().firmware_version
        except Exception:
            return ""

    def _log_rows(self) -> tuple[tuple[str, str, str], ...]:
        colours = {
            "DEBUG": "muted",
            "INFO": "text",
            "WARNING": "warning",
            "ERROR": "danger",
            "CRITICAL": "danger",
        }
        return tuple(
            (record.level[:4], f"{record.logger.split('.')[-1]}: {record.message}",
             colours.get(record.level, "text"))
            for record in log_buffer.records(limit=25)
        )

    # -- actions --------------------------------------------------------------

    def handle_action(self, event: UiEvent) -> None:
        """Route a UI action to the owning service."""
        self.actions += 1
        action, args = event.action, event.args
        self._bus.publish(
            Topics.UI_ACTION, {"action": action, "args": list(args)}, source=self.name
        )

        if action == "navigate" and args:
            self.navigate(args[0])
        elif action == "back":
            self.back()
        elif action == "set_mode" and args and self._modes is not None:
            self._set_mode(args[0])
        elif action == "set_theme" and args:
            self._theme = self._theme.with_mode(ThemeMode(args[0]))
            self._remember("ui.theme", args[0])
        elif action == "estop" and self._safety is not None:
            self._safety.trigger_estop("stop button pressed", source="user:ui")
            self._show_toast("Emergency stop engaged", "danger")
        elif action == "acknowledge" and self._safety is not None:
            cleared = self._safety.acknowledge("user:ui")
            self._show_toast(
                "Faults acknowledged" if cleared else "Cannot acknowledge: fault still active",
                "success" if cleared else "warning",
            )
        elif action == "selftest" and self._diagnostics is not None:
            report = self._diagnostics.selftest.run(allow_motion=False)
            self._show_toast(report.summary(), "success" if report.ok else "warning")
        elif action == "selftest_motion" and self._diagnostics is not None:
            report = self._diagnostics.selftest.run(allow_motion=True)
            self._show_toast(report.summary(), "success" if report.ok else "warning")
        elif action == "calibrate_start" and self._calibration is not None:
            self._calibration.start()
            self.navigate("calibration")
        elif action == "calibrate_cancel" and self._calibration is not None:
            self._calibration.cancel()
            self._show_toast("Calibration cancelled", "warning")
        elif action == "training_start" and args and self._training is not None:
            if self._training.start(args[0]):
                self.navigate("training")
        elif action == "training_stop" and self._training is not None:
            self._training.stop("user stopped")
        elif action == "grip" and args and self._controller is not None:
            self._apply_grip(args[0])
        elif action.startswith("toggle_"):
            self._toggle_accessibility(action)
        elif action == "font_scale" and args:
            self._adjust_font(args[0])
        elif action == "refresh_logs":
            self._show_toast("Logs refreshed", "info")
        else:
            log.debug("unhandled UI action", action=action, args=list(args))

    def _set_mode(self, value: str) -> None:
        try:
            mode = ModeId(value)
        except ValueError:
            log.warning("unknown mode requested from UI", value=value)
            return
        safety_state = self._safety.state if self._safety else None
        if not self._modes.activate(mode, reason="user selected", safety=safety_state):
            self._show_toast(f"Cannot switch to {mode.label} right now", "warning")
            return
        # Remembered only on an explicit user choice. An automatic fallback to
        # Manual must not overwrite the mode the user actually prefers.
        self._remember("modes.default", mode.value)

    def _apply_grip(self, value: str) -> None:
        try:
            grasp = GraspType(value)
        except ValueError:
            return
        result = self._controller.apply_grip(grasp, source="ui")
        if not result.accepted:
            self._show_toast(result.reason or "Grip rejected", "warning")

    def _toggle_accessibility(self, action: str) -> None:
        access = self._theme.accessibility
        mapping = {
            "toggle_contrast": "high_contrast",
            "toggle_motion": "reduce_motion",
            "toggle_values": "show_numeric_values",
            "toggle_haptics": "haptics",
            "toggle_audio": "audio_cues",
        }
        field_name = mapping.get(action)
        if field_name is None:
            return
        from dataclasses import replace

        updated = replace(access, **{field_name: not getattr(access, field_name)})
        self._theme = self._theme.with_accessibility(updated)
        self._remember(f"ui.accessibility.{field_name}", getattr(updated, field_name))
        self._show_toast(
            f"{field_name.replace('_', ' ').capitalize()}: "
            f"{'on' if getattr(updated, field_name) else 'off'}",
            "info",
        )

    def _adjust_font(self, argument: str) -> None:
        access = self._theme.accessibility
        if argument == "+":
            scale = clamp(access.font_scale + 0.1, 0.8, 2.0)
        elif argument == "-":
            scale = clamp(access.font_scale - 0.1, 0.8, 2.0)
        else:
            try:
                scale = clamp(float(argument), 0.8, 2.0)
            except ValueError:
                return
        from dataclasses import replace

        self._theme = self._theme.with_accessibility(replace(access, font_scale=scale))
        self._remember("ui.accessibility.font_scale", round(scale, 2))
        self._show_toast(f"Text size {scale:.1f}×", "info")

    def _remember(self, path: str, value: object) -> None:
        """Persist one preference to the active user profile.

        Failure is reported to the user but never raises: a read-only or full
        filesystem must not take the interface down, and the setting is still
        applied for this session.
        """
        if self._profiles is None:
            return
        try:
            profile = self._profiles.active()
            if profile is None:
                return
            profile.set(path, value)
            self._profiles.save(profile)
        except Exception as exc:
            log.warning("could not save preference", setting=path, error=str(exc))
            self._show_toast("Setting applied but not saved", "warning")

    # -- reporting ------------------------------------------------------------

    def health(self) -> HealthReport:
        if not self.running:
            return HealthReport.offline(self.name)
        return HealthReport.ok(
            self.name, frames=self.frames, actions=self.actions, route=self._route
        )
