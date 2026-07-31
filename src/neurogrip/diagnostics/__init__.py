"""Diagnostics: health, metrics, self-tests and the debug console.

Everything here is observation. Nothing in this package commands the hand — the
one exception, the range-of-motion self-test, goes through
:class:`~neurogrip.control.controller.HandController` like every other caller and
is skipped unless the user explicitly enables motion tests.
"""

from __future__ import annotations

from .console import Command, ConsoleResult, DebugConsole, build_console
from .metrics import Counter, Gauge, Histogram, MetricsRegistry, RateMeter
from .selftest import SelfTestReport, SelfTestRunner, TestOutcome, TestResult
from .service import DiagnosticsService, DiagnosticsSnapshot, build_standard_selftests

__all__ = [
    "Command",
    "ConsoleResult",
    "Counter",
    "DebugConsole",
    "DiagnosticsService",
    "DiagnosticsSnapshot",
    "Gauge",
    "Histogram",
    "MetricsRegistry",
    "RateMeter",
    "SelfTestReport",
    "SelfTestRunner",
    "TestOutcome",
    "TestResult",
    "build_console",
    "build_standard_selftests",
]
