"""Scene renderers.

A renderer turns a :class:`~neurogrip.ui.widgets.Scene` into something a human
can see. Three ship:

* :class:`TextRenderer` — ANSI text. This is the *reference* renderer: it works
  over SSH, in CI, and on a device with no display attached, and it is what the
  UI tests assert against. Being able to see the interface without hardware is
  worth a great deal during development.
* :class:`TkRenderer` — the touchscreen implementation, using Tkinter (present in
  any standard CPython with Tk). Guarded import.
* :class:`NullRenderer` — for headless runs.

Adding a renderer (LVGL on an MCU, a web front end) means implementing
:meth:`Renderer.render` and :meth:`Renderer.poll_events`. No screen changes.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ..core.types import Finger, clamp
from .theme import Theme
from .widgets import (
    Badge,
    Bar,
    Button,
    Column,
    Divider,
    Gauge,
    HandGraphic,
    Label,
    ListView,
    Panel,
    ProgressRing,
    Row,
    Scene,
    Spacer,
    Sparkline,
    Toggle,
    Widget,
)

__all__ = ["NullRenderer", "Renderer", "TextRenderer", "TkRenderer", "UiEvent", "create_renderer"]


@dataclass(frozen=True, slots=True)
class UiEvent:
    """An interaction produced by a renderer."""

    action: str
    args: tuple[str, ...] = ()
    widget_key: str = ""


@runtime_checkable
class Renderer(Protocol):
    """Draws scenes and reports interactions."""

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def render(self, scene: Scene, theme: Theme) -> None: ...

    def poll_events(self) -> list[UiEvent]:
        """Return interactions since the last call. Must not block."""
        ...


class NullRenderer:
    """Renders nothing. Used for headless operation and tests."""

    def __init__(self) -> None:
        self.last_scene: Scene | None = None
        self.frames = 0

    def start(self) -> None:
        """Nothing to initialise."""

    def stop(self) -> None:
        """Nothing to release."""

    def render(self, scene: Scene, theme: Theme) -> None:
        # Retained so tests can assert on what *would* have been displayed.
        self.last_scene = scene
        self.frames += 1

    def poll_events(self) -> list[UiEvent]:
        return []


# ---------------------------------------------------------------------------
# Text renderer
# ---------------------------------------------------------------------------

_ANSI = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "text": "\033[97m",
    "muted": "\033[90m",
    "primary": "\033[94m",
    "ok": "\033[92m",
    "success": "\033[92m",
    "warning": "\033[93m",
    "danger": "\033[91m",
    "neutral": "\033[37m",
    "ai": "\033[95m",
    "user": "\033[96m",
}

_BLOCKS = " ▁▂▃▄▅▆▇█"


class TextRenderer:
    """ANSI text renderer.

    Deliberately good enough to actually use: bars, sparklines, gauges and a
    schematic hand all render legibly in a terminal, so the whole interface can
    be driven and reviewed over SSH.
    """

    def __init__(self, *, width: int | None = None, colour: bool = True, stream=None) -> None:
        import sys

        self._stream = stream or sys.stdout
        self._width = width
        self._colour = colour
        self._pending: list[UiEvent] = []
        self._last_render = ""

    # -- lifecycle ------------------------------------------------------------

    def start(self) -> None:
        """Nothing to initialise; the terminal is already there."""

    def stop(self) -> None:
        self._stream.flush()

    def queue_event(self, action: str, *args: str) -> None:
        """Inject an interaction (used by the CLI and by tests)."""
        self._pending.append(UiEvent(action=action, args=tuple(args)))

    def poll_events(self) -> list[UiEvent]:
        events, self._pending = self._pending, []
        return events

    # -- rendering ------------------------------------------------------------

    @property
    def width(self) -> int:
        if self._width:
            return self._width
        return max(60, min(120, shutil.get_terminal_size((100, 30)).columns))

    def render(self, scene: Scene, theme: Theme) -> None:
        text = self.render_to_string(scene, theme)
        if text == self._last_render:
            return  # nothing changed; do not repaint
        self._last_render = text
        self._stream.write("\033[2J\033[H" if self._colour else "\n\n")
        self._stream.write(text)
        self._stream.write("\n")
        self._stream.flush()

    def render_to_string(self, scene: Scene, theme: Theme) -> str:
        """Render to a string. This is what the UI tests inspect."""
        width = self.width
        lines: list[str] = []

        status = "  ".join(f"{icon} {text}" for icon, text, _ in scene.status)
        lines.append(self._colourise(status.ljust(width), "muted"))
        lines.append("═" * width)
        lines.append(self._colourise(f" {scene.title.upper()} ", "bold"))

        if scene.banner is not None:
            lines.append(
                self._colourise(
                    f" {scene.banner.icon} {scene.banner.text} ".center(width, "─"),
                    scene.banner.colour,
                )
            )
        if scene.toast:
            lines.append(self._colourise(f"  ⓘ {scene.toast}", "primary"))

        for child in scene.children:
            lines.extend(self._render_widget(child, width, 0))

        lines.append("─" * width)
        nav = "  ".join(
            self._colourise(f"[{b.icon} {b.text}]", "primary" if b.variant == "primary" else "muted")
            for b in scene.nav
        )
        lines.append(nav)
        return "\n".join(lines)

    def _render_widget(self, widget: Widget, width: int, depth: int) -> list[str]:
        pad = "  " * depth
        lines: list[str] = []

        if isinstance(widget, Panel):
            title = f"┌─ {widget.title} " if widget.title else "┌"
            lines.append(pad + self._colourise(title.ljust(width - len(pad), "─"), widget.colour))
            if widget.subtitle:
                lines.append(pad + "│ " + self._colourise(widget.subtitle, "muted"))
            for child in widget.children:
                for line in self._render_widget(child, width - 2, depth):
                    lines.append(pad + "│ " + line)
            lines.append(pad + self._colourise("└".ljust(width - len(pad), "─"), widget.colour))

        elif isinstance(widget, (Row, Column)):
            if isinstance(widget, Row):
                parts: list[str] = []
                for child in widget.children:
                    rendered = self._render_widget(child, max(12, width // max(1, len(widget.children))), 0)
                    parts.append(rendered[0] if rendered else "")
                lines.append("  ".join(parts))
            else:
                for child in widget.children:
                    lines.extend(self._render_widget(child, width, depth))

        elif isinstance(widget, Label):
            prefix = f"{widget.icon} " if widget.icon else ""
            style = "bold" if widget.bold or widget.role in ("display", "title") else widget.colour
            lines.append(self._colourise(prefix + widget.text, style))

        elif isinstance(widget, Badge):
            lines.append(self._colourise(f"[{widget.icon} {widget.text}]", widget.colour))

        elif isinstance(widget, Bar):
            lines.append(self._bar(widget, width))

        elif isinstance(widget, Gauge):
            pct = clamp((widget.value - widget.minimum) / max(1e-9, widget.maximum - widget.minimum))
            lines.append(
                self._colourise(
                    f"{widget.label}: {self._meter(pct, 20)} {pct * 100:5.1f}{widget.unit}",
                    widget.colour,
                )
            )
            if widget.caption:
                lines.append(self._colourise(f"  {widget.caption}", "muted"))

        elif isinstance(widget, ProgressRing):
            lines.append(
                self._colourise(
                    f"{widget.label:>6}  {self._meter(widget.value, 24)}  {widget.caption}",
                    widget.colour,
                )
            )

        elif isinstance(widget, Sparkline):
            lines.append(
                self._colourise(
                    f"{widget.label:<10}{self._sparkline(widget.values, width - 12)}", widget.colour
                )
            )

        elif isinstance(widget, HandGraphic):
            lines.extend(self._hand(widget, width))

        elif isinstance(widget, ListView):
            if not widget.rows:
                lines.append(self._colourise(f"  {widget.empty_text}", "muted"))
            for text, detail, colour, icon in widget.rows[: widget.max_visible]:
                prefix = f"{icon} " if icon else ""
                lines.append(self._colourise(f"  {prefix}{text:<22}{detail}", colour))

        elif isinstance(widget, Button):
            variant = "muted" if not widget.enabled else widget.colour
            lines.append(self._colourise(f"  ( {widget.icon} {widget.text} )", variant))

        elif isinstance(widget, Toggle):
            mark = "▣" if widget.value else "☐"
            lines.append(f"  {mark} {widget.text}")
            if widget.caption:
                lines.append(self._colourise(f"      {widget.caption}", "muted"))

        elif isinstance(widget, Divider):
            lines.append(self._colourise(("─ " + widget.label + " ").ljust(width, "─"), "muted"))

        elif isinstance(widget, Spacer):
            lines.extend([""] * widget.size)

        return lines

    # -- primitives -----------------------------------------------------------

    def _bar(self, widget: Bar, width: int) -> str:
        colour = widget.colour
        if widget.threshold is not None and widget.value >= widget.threshold:
            colour = "danger"
        meter = self._meter(widget.value, max(8, min(28, width - 26)), target=widget.target)
        value_text = f"{widget.value * 100:5.1f}{widget.unit}" if widget.show_value else ""
        return self._colourise(f"{widget.label:<10}{meter} {value_text}", colour)

    @staticmethod
    def _meter(value: float, size: int, target: float | None = None) -> str:
        value = clamp(value)
        filled = int(round(value * size))
        cells = ["█"] * filled + ["░"] * (size - filled)
        if target is not None:
            index = min(size - 1, int(round(clamp(target) * size)))
            cells[index] = "┃"
        return "".join(cells)

    @staticmethod
    def _sparkline(values: tuple[float, ...], size: int) -> str:
        if not values:
            return ""
        data = list(values)[-max(1, size) :]
        low, high = min(data), max(data)
        span = high - low
        if span < 1e-9:
            return _BLOCKS[1] * len(data)
        return "".join(_BLOCKS[int(clamp((v - low) / span) * (len(_BLOCKS) - 1))] for v in data)

    def _hand(self, widget: HandGraphic, width: int) -> list[str]:
        """Schematic hand: one column per finger, drawn bottom-up."""
        rows = 6
        columns: list[list[str]] = []
        for finger in Finger:
            closure = widget.pose[finger] if widget.pose else 0.0
            filled = int(round(clamp(closure) * rows))
            contact = int(finger) in widget.contacts
            glyph = "◉" if contact else ("█" if widget.ai_driven else "▊")
            column = [glyph] * filled + ["·"] * (rows - filled)
            columns.append(list(reversed(column)))

        lines: list[str] = []
        for r in range(rows):
            cells = "   ".join(columns[f][r] for f in range(len(Finger)))
            lines.append(f"    {cells}")
        labels = "   ".join(f.label[0] for f in Finger)
        lines.append(self._colourise(f"    {labels}", "muted"))
        if widget.label:
            source = "AI" if widget.ai_driven else "you"
            lines.append(self._colourise(f"    {widget.label} ({source})", "muted"))
        return lines

    def _colourise(self, text: str, style: str) -> str:
        if not self._colour:
            return text
        code = _ANSI.get(style)
        if code is None:
            return text
        return f"{code}{text}{_ANSI['reset']}"


# ---------------------------------------------------------------------------
# Tk renderer
# ---------------------------------------------------------------------------


class TkRenderer:
    """Touchscreen renderer built on Tkinter.

    Rebuilds the widget tree each frame rather than diffing. At 15–20 Hz with a
    few dozen widgets this is comfortably fast enough on the target hardware, and
    it keeps the renderer stateless — which is the property that makes the
    declarative screens worth having in the first place.

    TODO(ui): if a future screen grows to hundreds of widgets, add key-based
    reconciliation. Every widget already carries a stable ``key`` for exactly
    that purpose.
    """

    def __init__(self, *, width: int = 800, height: int = 480, fullscreen: bool = False) -> None:
        self._width = width
        self._height = height
        self._fullscreen = fullscreen
        self._root = None
        self._canvas_frame = None
        self._pending: list[UiEvent] = []
        self._available = False

    def start(self) -> None:
        try:
            import tkinter as tk
        except ImportError as exc:  # pragma: no cover - depends on the platform
            raise RuntimeError(
                "tkinter is not available; use the 'text' renderer instead"
            ) from exc

        self._tk = tk
        self._root = tk.Tk()
        self._root.title("NeuroGrip")
        self._root.geometry(f"{self._width}x{self._height}")
        if self._fullscreen:
            self._root.attributes("-fullscreen", True)
        self._root.configure(bg="#0d1117")
        self._available = True

    def stop(self) -> None:
        if self._root is not None:
            try:
                self._root.destroy()
            except Exception:
                pass
            self._root = None
        self._available = False

    def poll_events(self) -> list[UiEvent]:
        if self._root is not None:
            # Pump Tk without blocking: the UI shares its thread with nothing,
            # but the scheduler still owns the cadence.
            self._root.update_idletasks()
            self._root.update()
        events, self._pending = self._pending, []
        return events

    def render(self, scene: Scene, theme: Theme) -> None:
        if not self._available or self._root is None:
            return
        tk = self._tk
        palette = theme.palette

        if self._canvas_frame is not None:
            self._canvas_frame.destroy()
        frame = tk.Frame(self._root, bg=palette.background)
        frame.pack(fill="both", expand=True)
        self._canvas_frame = frame

        status = "   ".join(f"{icon} {text}" for icon, text, _ in scene.status)
        tk.Label(
            frame, text=status, bg=palette.background, fg=palette.text_muted,
            font=(theme.font_family, theme.font_size("caption")), anchor="w",
        ).pack(fill="x", padx=theme.spacing)

        tk.Label(
            frame, text=scene.title, bg=palette.background, fg=palette.text,
            font=(theme.font_family, theme.font_size("title"), "bold"), anchor="w",
        ).pack(fill="x", padx=theme.spacing)

        if scene.banner is not None:
            tk.Label(
                frame,
                text=f"{scene.banner.icon}  {scene.banner.text}",
                bg=palette.status(scene.banner.colour),
                fg=palette.background,
                font=(theme.font_family, theme.font_size("body"), "bold"),
            ).pack(fill="x", padx=theme.spacing, pady=theme.spacing // 2)

        body = tk.Frame(frame, bg=palette.background)
        body.pack(fill="both", expand=True, padx=theme.spacing)
        for child in scene.children:
            self._render_widget(body, child, theme)

        nav = tk.Frame(frame, bg=palette.surface, height=theme.touch_target)
        nav.pack(fill="x", side="bottom")
        for button in scene.nav:
            self._make_button(nav, button, theme).pack(side="left", expand=True, fill="both")

    def _render_widget(self, parent, widget: Widget, theme: Theme) -> None:
        tk = self._tk
        palette = theme.palette

        if isinstance(widget, Panel):
            box = tk.LabelFrame(
                parent,
                text=widget.title,
                bg=palette.surface,
                fg=palette.status(widget.colour),
                font=(theme.font_family, theme.font_size("caption"), "bold"),
                bd=theme.border_width,
                relief="solid",
            )
            box.pack(fill="x", pady=theme.spacing // 2)
            for child in widget.children:
                self._render_widget(box, child, theme)

        elif isinstance(widget, (Row, Column)):
            container = tk.Frame(parent, bg=palette.surface)
            container.pack(fill="x")
            side = "left" if isinstance(widget, Row) else "top"
            for child in widget.children:
                holder = tk.Frame(container, bg=palette.surface)
                holder.pack(side=side, fill="x", expand=True)
                self._render_widget(holder, child, theme)

        elif isinstance(widget, Label):
            tk.Label(
                parent,
                text=f"{widget.icon} {widget.text}".strip(),
                bg=palette.surface,
                fg=palette.status(widget.colour),
                font=(
                    theme.font_mono if widget.monospace else theme.font_family,
                    theme.font_size(widget.role),
                    "bold" if widget.bold else "normal",
                ),
                anchor="w",
                justify="left",
                wraplength=self._width - 40,
            ).pack(fill="x", padx=theme.spacing // 2)

        elif isinstance(widget, (Bar, Gauge, ProgressRing)):
            value = getattr(widget, "value", 0.0)
            label = getattr(widget, "label", "")
            colour = palette.status(getattr(widget, "colour", "primary"))
            holder = tk.Frame(parent, bg=palette.surface)
            holder.pack(fill="x", padx=theme.spacing // 2)
            tk.Label(
                holder, text=label, bg=palette.surface, fg=palette.text_muted,
                font=(theme.font_family, theme.font_size("caption")), width=12, anchor="w",
            ).pack(side="left")
            canvas = tk.Canvas(
                holder, height=max(10, theme.font_size("body")), bg=palette.surface_alt,
                highlightthickness=0,
            )
            canvas.pack(side="left", fill="x", expand=True)
            canvas.update_idletasks()
            span = max(1, canvas.winfo_width() or 200)
            canvas.create_rectangle(0, 0, span * clamp(value), 40, fill=colour, width=0)
            if theme.accessibility.show_numeric_values:
                tk.Label(
                    holder, text=f"{clamp(value) * 100:5.1f}%", bg=palette.surface, fg=palette.text,
                    font=(theme.font_mono, theme.font_size("caption")),
                ).pack(side="left")

        elif isinstance(widget, Badge):
            tk.Label(
                parent,
                text=f" {widget.icon} {widget.text} ",
                bg=palette.status(widget.colour),
                fg=palette.background,
                font=(theme.font_family, theme.font_size("caption"), "bold"),
            ).pack(anchor="w", padx=theme.spacing // 2, pady=2)

        elif isinstance(widget, ListView):
            for text, detail, colour, icon in widget.rows[: widget.max_visible]:
                tk.Label(
                    parent,
                    text=f"{icon} {text:<20} {detail}",
                    bg=palette.surface,
                    fg=palette.status(colour),
                    font=(theme.font_mono if widget.monospace else theme.font_family,
                          theme.font_size("caption")),
                    anchor="w",
                ).pack(fill="x", padx=theme.spacing)
            if not widget.rows:
                tk.Label(
                    parent, text=widget.empty_text, bg=palette.surface, fg=palette.text_muted,
                    font=(theme.font_family, theme.font_size("caption")), anchor="w",
                ).pack(fill="x", padx=theme.spacing)

        elif isinstance(widget, Button):
            self._make_button(parent, widget, theme).pack(
                side="left", padx=theme.spacing // 2, pady=theme.spacing // 2
            )

        elif isinstance(widget, Toggle):
            variable = tk.BooleanVar(value=widget.value)
            tk.Checkbutton(
                parent,
                text=widget.text,
                variable=variable,
                bg=palette.surface,
                fg=palette.text,
                selectcolor=palette.surface_alt,
                activebackground=palette.surface,
                font=(theme.font_family, theme.font_size("body")),
                anchor="w",
                command=lambda w=widget: self._pending.append(
                    UiEvent(action=w.action, widget_key=w.key)
                ),
            ).pack(fill="x", padx=theme.spacing)

        elif isinstance(widget, HandGraphic):
            canvas = tk.Canvas(
                parent, height=90, bg=palette.surface_alt, highlightthickness=0
            )
            canvas.pack(fill="x", padx=theme.spacing, pady=theme.spacing // 2)
            for index, finger in enumerate(Finger):
                closure = widget.pose[finger] if widget.pose else 0.0
                x = 30 + index * 46
                height = 12 + 60 * (1.0 - clamp(closure))
                colour = (
                    palette.warning
                    if int(finger) in widget.contacts
                    else (palette.ai if widget.ai_driven else palette.user)
                )
                canvas.create_rectangle(x, 80 - height, x + 26, 80, fill=colour, width=0)
                canvas.create_text(
                    x + 13, 86, text=finger.label[0], fill=palette.text_muted,
                    font=(theme.font_family, theme.font_size("caption")),
                )

        elif isinstance(widget, Divider):
            tk.Frame(parent, bg=palette.border, height=1).pack(fill="x", pady=theme.spacing // 2)

    def _make_button(self, parent, button: Button, theme: Theme):
        tk = self._tk
        palette = theme.palette
        fill = {
            "primary": palette.primary,
            "secondary": palette.surface_alt,
            "ghost": palette.surface,
            "danger": palette.danger,
        }.get(button.variant, palette.primary)
        return tk.Button(
            parent,
            text=f"{button.icon} {button.text}".strip(),
            bg=fill,
            fg=palette.background if button.variant in ("primary", "danger") else palette.text,
            activebackground=fill,
            font=(theme.font_family, theme.font_size("body")),
            height=1,
            bd=0,
            state="normal" if button.enabled else "disabled",
            command=lambda b=button: self._pending.append(
                UiEvent(action=b.action, args=b.args, widget_key=b.key)
            ),
        )


def create_renderer(name: str, **kwargs) -> Renderer:
    """Instantiate a renderer by name, falling back to text then null."""
    from ..core.logging import get_logger

    log = get_logger(__name__)
    if name == "null":
        return NullRenderer()
    if name == "text":
        return TextRenderer(**{k: v for k, v in kwargs.items() if k in ("width", "colour")})
    if name == "tk":
        renderer = TkRenderer(
            **{k: v for k, v in kwargs.items() if k in ("width", "height", "fullscreen")}
        )
        try:
            renderer.start()
            return renderer
        except RuntimeError as exc:
            log.warning("Tk renderer unavailable; falling back to text", error=str(exc))
            return TextRenderer()
    log.warning("unknown renderer; using text", requested=name)
    return TextRenderer()
