"""Theming and accessibility.

The touchscreen is small, often viewed at arm's length, sometimes outdoors, and
operated by someone who may be using their *other* hand — the one that works — to
hold the device steady. That shapes every decision here:

* **Large touch targets.** ``Theme.touch_target`` is never below 44 px, the
  smallest reliably hittable target, and scales with the font size.
* **Contrast first.** The dark and light palettes both meet WCAG AA against their
  backgrounds; the high-contrast variant meets AAA.
* **Motion is optional.** ``reduce_motion`` disables every animation. Vestibular
  sensitivity is common, and an animated UI on a device you are wearing is worse
  than one on a desk.
* **Colour is never the only signal.** Status is always conveyed by an icon or
  label as well as a colour, so the interface works for colour-blind users and in
  direct sunlight.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum

__all__ = ["DARK", "HIGH_CONTRAST", "LIGHT", "AccessibilitySettings", "Palette", "Theme", "ThemeMode"]


class ThemeMode(str, Enum):
    """Selected appearance."""

    DARK = "dark"
    LIGHT = "light"
    HIGH_CONTRAST = "high_contrast"
    #: Follow ambient light, if the device has a sensor; falls back to dark.
    AUTO = "auto"

    @property
    def label(self) -> str:
        return self.value.replace("_", " ").title()


@dataclass(frozen=True, slots=True)
class Palette:
    """Named colours, as ``#rrggbb`` strings."""

    background: str
    surface: str
    surface_alt: str
    border: str
    text: str
    text_muted: str
    primary: str
    success: str
    warning: str
    danger: str
    neutral: str
    #: Colour for AI-contributed elements, so assistance is always visually
    #: distinguishable from the user's own direct control.
    ai: str
    #: Colour for elements driven directly by the user's muscles.
    user: str

    def status(self, key: str) -> str:
        """Resolve a semantic status name to a colour."""
        return {
            "ok": self.success,
            "good": self.success,
            "warn": self.warning,
            "warning": self.warning,
            "error": self.danger,
            "fail": self.danger,
            "danger": self.danger,
            "info": self.primary,
            "primary": self.primary,
            "neutral": self.neutral,
            "muted": self.text_muted,
            "ai": self.ai,
            "user": self.user,
        }.get(key, self.text)


DARK = Palette(
    background="#0d1117",
    surface="#161b22",
    surface_alt="#1f2530",
    border="#30363d",
    text="#e6edf3",
    text_muted="#8b949e",
    primary="#4c9aff",
    success="#3fb950",
    warning="#d29922",
    danger="#f85149",
    neutral="#6e7681",
    ai="#a371f7",
    user="#2dd4bf",
)

LIGHT = Palette(
    background="#f6f8fa",
    surface="#ffffff",
    surface_alt="#eef1f5",
    border="#d0d7de",
    text="#1f2328",
    text_muted="#59636e",
    primary="#0969da",
    success="#1a7f37",
    warning="#9a6700",
    danger="#cf222e",
    neutral="#6e7781",
    ai="#8250df",
    user="#0f766e",
)

#: Pure black/white with saturated accents. Meets WCAG AAA and is legible in
#: direct sunlight, which matters for a device used outdoors.
HIGH_CONTRAST = Palette(
    background="#000000",
    surface="#000000",
    surface_alt="#1a1a1a",
    border="#ffffff",
    text="#ffffff",
    text_muted="#d0d0d0",
    primary="#00b0ff",
    success="#00e676",
    warning="#ffd600",
    danger="#ff1744",
    neutral="#9e9e9e",
    ai="#e040fb",
    user="#00e5ff",
)


@dataclass(frozen=True, slots=True)
class AccessibilitySettings:
    """User accessibility preferences."""

    #: Multiplier applied to every font size and touch target.
    font_scale: float = 1.0
    #: Disable all animation and transitions.
    reduce_motion: bool = False
    #: Use the high-contrast palette regardless of theme mode.
    high_contrast: bool = False
    #: Show numeric values alongside every gauge and bar.
    show_numeric_values: bool = True
    #: Haptic feedback on touch, where the hardware supports it.
    haptics: bool = True
    #: Extend how long transient messages stay on screen, in seconds.
    message_duration_s: float = 4.0
    #: Announce state changes for screen readers / audio feedback.
    audio_cues: bool = False

    def scaled(self, size: int) -> int:
        """Apply the font scale to a base size, with a sensible floor."""
        return max(9, int(round(size * self.font_scale)))


@dataclass(frozen=True, slots=True)
class Theme:
    """Resolved appearance: palette, typography and spacing."""

    mode: ThemeMode = ThemeMode.DARK
    palette: Palette = DARK
    accessibility: AccessibilitySettings = field(default_factory=AccessibilitySettings)

    # -- typography -----------------------------------------------------------
    font_family: str = "Inter, DejaVu Sans, sans-serif"
    font_mono: str = "JetBrains Mono, DejaVu Sans Mono, monospace"
    size_display: int = 34
    size_title: int = 22
    size_body: int = 16
    size_caption: int = 13

    # -- metrics --------------------------------------------------------------
    spacing: int = 8
    radius: int = 10
    border_width: int = 1
    #: Minimum touch target in pixels — never scaled below this.
    min_touch_target: int = 44

    @property
    def touch_target(self) -> int:
        return max(
            self.min_touch_target,
            int(round(self.min_touch_target * self.accessibility.font_scale)),
        )

    @property
    def animations_enabled(self) -> bool:
        return not self.accessibility.reduce_motion

    def font_size(self, role: str = "body") -> int:
        base = {
            "display": self.size_display,
            "title": self.size_title,
            "body": self.size_body,
            "caption": self.size_caption,
        }.get(role, self.size_body)
        return self.accessibility.scaled(base)

    def colour(self, key: str) -> str:
        """Resolve a semantic colour name."""
        return self.palette.status(key)

    def with_mode(self, mode: ThemeMode) -> Theme:
        """Switch appearance, honouring the high-contrast accessibility override."""
        if self.accessibility.high_contrast:
            palette = HIGH_CONTRAST
            resolved = ThemeMode.HIGH_CONTRAST
        else:
            palette = {
                ThemeMode.DARK: DARK,
                ThemeMode.LIGHT: LIGHT,
                ThemeMode.HIGH_CONTRAST: HIGH_CONTRAST,
                ThemeMode.AUTO: DARK,
            }[mode]
            resolved = mode
        return replace(self, mode=resolved, palette=palette)

    def with_accessibility(self, settings: AccessibilitySettings) -> Theme:
        """Apply accessibility settings, re-resolving the palette."""
        return replace(self, accessibility=settings).with_mode(self.mode)

    @classmethod
    def from_config(cls, config) -> Theme:
        """Build from the ``[ui]`` and ``[ui.accessibility]`` sections."""
        section = config.section("ui")
        access = section.section("accessibility")
        settings = AccessibilitySettings(
            font_scale=access.get_float("font_scale", 1.0),
            reduce_motion=access.get_bool("reduce_motion", False),
            high_contrast=access.get_bool("high_contrast", False),
            show_numeric_values=access.get_bool("show_numeric_values", True),
            haptics=access.get_bool("haptics", True),
            message_duration_s=access.get_float("message_duration_s", 4.0),
            audio_cues=access.get_bool("audio_cues", False),
        )
        mode = ThemeMode(section.get_str("theme", "dark"))
        return cls(accessibility=settings).with_mode(mode)
