"""Declarative widget tree.

Screens are pure functions of application state that return a :class:`Scene` — a
tree of immutable widget descriptions. Renderers turn a scene into pixels (Tk) or
characters (text). Nothing here draws anything.

Why a declarative layer instead of a UI toolkit directly:

* **The interface becomes testable.** ``tests/unit/test_ui.py`` asserts that the
  Manual-mode dashboard contains an "AI DISABLED" banner. That is a real
  requirement, and this makes it checkable without a display.
* **The toolkit becomes replaceable.** Tk today, LVGL or a web front end later,
  without rewriting a single screen.
* **Screens stay honest.** A screen that cannot reach into a widget and mutate it
  cannot accumulate hidden state that disagrees with the system.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum

from ..core.types import HandPose

__all__ = [
    "Align",
    "Badge",
    "Bar",
    "Button",
    "Column",
    "Divider",
    "Gauge",
    "HandGraphic",
    "Label",
    "ListView",
    "Panel",
    "ProgressRing",
    "Row",
    "Scene",
    "Spacer",
    "Sparkline",
    "Toggle",
    "Widget",
]


class Align(str, Enum):
    START = "start"
    CENTER = "center"
    END = "end"
    SPREAD = "spread"


@dataclass(frozen=True, slots=True)
class Widget:
    """Base for every widget. Subclasses add fields; none add behaviour."""

    #: Stable identifier, used by tests and by the renderer to preserve focus.
    key: str = ""

    def walk(self) -> Iterator[Widget]:
        """Depth-first traversal of this widget and its children."""
        yield self
        for child in getattr(self, "children", ()) or ():
            yield from child.walk()

    def find(self, key: str) -> Widget | None:
        """Find a descendant by key."""
        for widget in self.walk():
            if widget.key == key:
                return widget
        return None


@dataclass(frozen=True, slots=True)
class Label(Widget):
    """A run of text."""

    text: str = ""
    #: ``display`` | ``title`` | ``body`` | ``caption``
    role: str = "body"
    #: Semantic colour key resolved by the theme.
    colour: str = "text"
    bold: bool = False
    monospace: bool = False
    align: Align = Align.START
    #: Optional leading icon (emoji or a glyph the renderer maps).
    icon: str = ""


@dataclass(frozen=True, slots=True)
class Divider(Widget):
    """A horizontal rule."""

    label: str = ""


@dataclass(frozen=True, slots=True)
class Spacer(Widget):
    """Fixed empty space, in theme spacing units."""

    size: int = 1


@dataclass(frozen=True, slots=True)
class Badge(Widget):
    """A small status pill. Always carries text, never colour alone."""

    text: str = ""
    colour: str = "neutral"
    icon: str = ""


@dataclass(frozen=True, slots=True)
class Bar(Widget):
    """A horizontal level bar."""

    value: float = 0.0
    label: str = ""
    colour: str = "primary"
    #: Optional target marker, drawn as a line across the bar.
    target: float | None = None
    #: Optional warning threshold; the bar turns ``danger`` above it.
    threshold: float | None = None
    show_value: bool = True
    #: Formatting for the numeric readout.
    unit: str = "%"


@dataclass(frozen=True, slots=True)
class Gauge(Widget):
    """A radial gauge for a single scalar."""

    value: float = 0.0
    label: str = ""
    colour: str = "primary"
    minimum: float = 0.0
    maximum: float = 1.0
    unit: str = ""
    caption: str = ""


@dataclass(frozen=True, slots=True)
class ProgressRing(Widget):
    """Circular progress, used for timed exercises and calibration steps."""

    value: float = 0.0
    label: str = ""
    caption: str = ""
    colour: str = "primary"


@dataclass(frozen=True, slots=True)
class Sparkline(Widget):
    """A compact time series."""

    values: tuple[float, ...] = field(default_factory=tuple)
    colour: str = "primary"
    label: str = ""
    minimum: float | None = None
    maximum: float | None = None
    #: Optional horizontal reference line (a threshold, a target).
    reference: float | None = None


@dataclass(frozen=True, slots=True)
class Button(Widget):
    """A touch target."""

    text: str = ""
    #: Action name published on ``Topics.UI_ACTION`` when pressed.
    action: str = ""
    #: Arguments included with the action.
    args: tuple[str, ...] = field(default_factory=tuple)
    colour: str = "primary"
    icon: str = ""
    enabled: bool = True
    #: ``primary`` | ``secondary`` | ``ghost`` | ``danger``
    variant: str = "primary"
    #: Require a confirmation tap. Used for anything that moves the hand.
    confirm: bool = False


@dataclass(frozen=True, slots=True)
class Toggle(Widget):
    """A boolean switch."""

    text: str = ""
    value: bool = False
    action: str = ""
    caption: str = ""
    enabled: bool = True


@dataclass(frozen=True, slots=True)
class ListView(Widget):
    """A scrollable list of rows."""

    #: Each row is ``(text, detail, colour, icon)``.
    rows: tuple[tuple[str, str, str, str], ...] = field(default_factory=tuple)
    empty_text: str = "Nothing to show"
    monospace: bool = False
    max_visible: int = 12


@dataclass(frozen=True, slots=True)
class HandGraphic(Widget):
    """A schematic of the hand, showing per-finger closure.

    The single most important widget on the dashboard: it is the user's direct
    feedback that the hand is doing what they asked. ``contacts`` and
    ``ai_driven`` are rendered distinctly so the user can always tell whether a
    finger stopped because it touched something, and whether the position came
    from them or from the assistance.
    """

    pose: HandPose = field(default_factory=HandPose.open_hand)
    #: Commanded pose, drawn as a ghost behind the measured one.
    target: HandPose | None = None
    #: Fingers currently in contact with an object.
    contacts: tuple[int, ...] = field(default_factory=tuple)
    #: True when the current pose came from the AI rather than direct control.
    ai_driven: bool = False
    label: str = ""


@dataclass(frozen=True, slots=True)
class Row(Widget):
    """Horizontal layout."""

    children: tuple[Widget, ...] = field(default_factory=tuple)
    align: Align = Align.START
    gap: int = 1


@dataclass(frozen=True, slots=True)
class Column(Widget):
    """Vertical layout."""

    children: tuple[Widget, ...] = field(default_factory=tuple)
    gap: int = 1


@dataclass(frozen=True, slots=True)
class Panel(Widget):
    """A titled card grouping related widgets."""

    title: str = ""
    children: tuple[Widget, ...] = field(default_factory=tuple)
    #: Accent colour for the title bar.
    colour: str = "surface"
    #: Draw with an emphasised border; used for alerts.
    emphasised: bool = False
    subtitle: str = ""


@dataclass(frozen=True, slots=True)
class Scene:
    """A complete screen."""

    title: str
    children: tuple[Widget, ...] = field(default_factory=tuple)
    #: Route key of this screen.
    route: str = ""
    #: Buttons rendered in the persistent bottom navigation bar.
    nav: tuple[Button, ...] = field(default_factory=tuple)
    #: Sticky banner across the top (e.g. "AI DISABLED", "E-STOP ENGAGED").
    banner: Badge | None = None
    #: Transient toast message.
    toast: str = ""
    #: Status-bar items: ``(icon, text, colour)``.
    status: tuple[tuple[str, str, str], ...] = field(default_factory=tuple)

    def walk(self) -> Iterator[Widget]:
        for child in self.children:
            yield from child.walk()
        yield from self.nav

    def find(self, key: str) -> Widget | None:
        for widget in self.walk():
            if widget.key == key:
                return widget
        return None

    def text_content(self) -> str:
        """All text in the scene, concatenated. Used by tests and screen readers."""
        parts: list[str] = [self.title]
        if self.banner is not None:
            parts.append(self.banner.text)
        for widget in self.walk():
            for attribute in ("text", "title", "label", "caption", "subtitle"):
                value = getattr(widget, attribute, "")
                if isinstance(value, str) and value:
                    parts.append(value)
            if isinstance(widget, ListView):
                parts.extend(f"{row[0]} {row[1]}" for row in widget.rows)
        return " | ".join(parts)

    def buttons(self) -> tuple[Button, ...]:
        """Every actionable button, including the navigation bar."""
        return tuple(w for w in self.walk() if isinstance(w, Button))


def panel(title: str, *children: Widget, **kwargs) -> Panel:
    """Terse constructor used throughout ``screens.py``."""
    return Panel(title=title, children=tuple(c for c in children if c is not None), **kwargs)


def column(*children: Widget, **kwargs) -> Column:
    return Column(children=tuple(c for c in children if c is not None), **kwargs)


def row(*children: Widget, **kwargs) -> Row:
    return Row(children=tuple(c for c in children if c is not None), **kwargs)
