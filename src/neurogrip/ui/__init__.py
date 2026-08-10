"""Touchscreen interface.

Three layers, strictly separated::

    ViewModel   immutable snapshot of application state, assembled once per frame
        ▼
    screens     pure functions: ViewModel → Scene (a tree of widget descriptions)
        ▼
    Renderer    Scene → pixels (Tk) or characters (text) or nothing (null)

Screens hold no state and touch no hardware. Interaction is a published action,
routed by :class:`~neurogrip.ui.app.UiService` to the owning service — so the UI
can never command the actuators except through the same
:class:`~neurogrip.control.controller.HandController` path everything else uses.

The practical benefit: the entire interface renders in a terminal (or into a
string, in tests) with no display attached. ``tests/unit/test_ui.py`` asserts, for
example, that Manual Mode shows the "AI DISABLED" banner — a real requirement,
made checkable.
"""

from __future__ import annotations

from .app import UiService
from .renderers import NullRenderer, Renderer, TextRenderer, TkRenderer, UiEvent, create_renderer
from .screens import ROUTES, SCREENS, ViewModel, build_scene
from .theme import DARK, HIGH_CONTRAST, LIGHT, AccessibilitySettings, Palette, Theme, ThemeMode
from .widgets import (
    Badge,
    Bar,
    Button,
    Gauge,
    HandGraphic,
    Label,
    ListView,
    Panel,
    Scene,
    Sparkline,
    Toggle,
    Widget,
)

__all__ = [
    "DARK",
    "HIGH_CONTRAST",
    "LIGHT",
    "ROUTES",
    "SCREENS",
    "AccessibilitySettings",
    "Badge",
    "Bar",
    "Button",
    "Gauge",
    "HandGraphic",
    "Label",
    "ListView",
    "NullRenderer",
    "Palette",
    "Panel",
    "Renderer",
    "Scene",
    "Sparkline",
    "TextRenderer",
    "Theme",
    "ThemeMode",
    "TkRenderer",
    "Toggle",
    "UiEvent",
    "UiService",
    "ViewModel",
    "Widget",
    "build_scene",
    "create_renderer",
]
