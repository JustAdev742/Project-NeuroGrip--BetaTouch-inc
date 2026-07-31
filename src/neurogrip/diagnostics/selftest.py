"""Power-on and on-demand self-tests.

Run automatically at startup (before the actuators are enabled) and on demand
from the Diagnostics screen. Structured so a failing test tells the user *what to
do*, not just that something is wrong — a device that reports "ERROR 7" is a
device that goes back to the clinic unnecessarily.

Tests are ordered from least to most physical. Anything that moves the hand is
marked ``requires_motion`` and is skipped unless the caller explicitly opts in,
so a self-test at boot never surprises the wearer by moving their limb.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

from ..core.clock import Clock
from ..core.errors import Severity
from ..core.logging import get_logger

__all__ = ["SelfTest", "SelfTestReport", "SelfTestRunner", "TestOutcome", "TestResult"]

log = get_logger(__name__)


class TestOutcome(str, Enum):
    """Result of one test."""

    #: These are self-test result types, not pytest test cases. Without this,
    #: pytest tries to collect them by name and emits a collection warning.
    __test__ = False

    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"

    @property
    def symbol(self) -> str:
        return {"pass": "✓", "warn": "!", "fail": "✗", "skip": "–"}[self.value]

    @property
    def is_blocking(self) -> bool:
        return self is TestOutcome.FAIL


@dataclass(frozen=True, slots=True)
class TestResult:
    """Outcome of one self-test."""

    #: See :class:`TestOutcome` — not a pytest test case.
    __test__ = False

    name: str
    outcome: TestOutcome
    message: str = ""
    duration_ms: float = 0.0
    #: Measured values, shown in the expanded row on the diagnostics screen.
    measurements: dict[str, float | str] = field(default_factory=dict)
    #: What the user should do about a warning or failure.
    remedy: str = ""

    def __str__(self) -> str:  # pragma: no cover - display helper
        return f"{self.outcome.symbol} {self.name}: {self.message}"


@dataclass(frozen=True, slots=True)
class SelfTest:
    """A registered test."""

    name: str
    description: str
    run: Callable[[], TestResult]
    #: True for tests that command the hand to move.
    requires_motion: bool = False
    #: Failure severity, used to decide whether startup may continue.
    severity: Severity = Severity.FALLBACK
    category: str = "general"


@dataclass(frozen=True, slots=True)
class SelfTestReport:
    """Aggregate result of a self-test run."""

    results: tuple[TestResult, ...]
    duration_s: float
    timestamp: float

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.outcome is TestOutcome.PASS)

    @property
    def warnings(self) -> int:
        return sum(1 for r in self.results if r.outcome is TestOutcome.WARN)

    @property
    def failures(self) -> tuple[TestResult, ...]:
        return tuple(r for r in self.results if r.outcome is TestOutcome.FAIL)

    @property
    def skipped(self) -> int:
        return sum(1 for r in self.results if r.outcome is TestOutcome.SKIP)

    @property
    def ok(self) -> bool:
        """True when nothing failed. Warnings do not block startup."""
        return not self.failures

    def summary(self) -> str:
        return (
            f"{self.passed} passed, {self.warnings} warnings, "
            f"{len(self.failures)} failed, {self.skipped} skipped "
            f"({self.duration_s:.1f} s)"
        )

    def remedies(self) -> tuple[str, ...]:
        return tuple(r.remedy for r in self.results if r.remedy and r.outcome is not TestOutcome.PASS)


class SelfTestRunner:
    """Registers and runs self-tests."""

    def __init__(self, clock: Clock) -> None:
        self._clock = clock
        self._tests: list[SelfTest] = []
        self._last_report: SelfTestReport | None = None

    def register(
        self,
        name: str,
        description: str,
        run: Callable[[], TestResult],
        *,
        requires_motion: bool = False,
        severity: Severity = Severity.FALLBACK,
        category: str = "general",
    ) -> None:
        """Add a test."""
        self._tests.append(
            SelfTest(
                name=name,
                description=description,
                run=run,
                requires_motion=requires_motion,
                severity=severity,
                category=category,
            )
        )

    @property
    def tests(self) -> tuple[SelfTest, ...]:
        return tuple(self._tests)

    @property
    def last_report(self) -> SelfTestReport | None:
        return self._last_report

    def run(
        self, *, allow_motion: bool = False, categories: tuple[str, ...] | None = None
    ) -> SelfTestReport:
        """Run the registered tests.

        Motion tests are skipped unless ``allow_motion`` is set — startup never
        sets it, the Diagnostics screen asks the user first.
        """
        started = self._clock.monotonic()
        results: list[TestResult] = []

        for test in self._tests:
            if categories is not None and test.category not in categories:
                continue
            if test.requires_motion and not allow_motion:
                results.append(
                    TestResult(
                        name=test.name,
                        outcome=TestOutcome.SKIP,
                        message="skipped: this test moves the hand",
                        remedy="Run from Diagnostics ▸ Self-test with motion enabled.",
                    )
                )
                continue

            test_start = time.perf_counter()
            try:
                result = test.run()
            except Exception as exc:
                log.error("self-test raised", test=test.name, error=str(exc))
                result = TestResult(
                    name=test.name,
                    outcome=TestOutcome.FAIL,
                    message=f"test raised: {exc}",
                    remedy="This is a software fault. Please report it with the logs.",
                )
            duration_ms = (time.perf_counter() - test_start) * 1000.0
            results.append(
                TestResult(
                    name=result.name or test.name,
                    outcome=result.outcome,
                    message=result.message,
                    duration_ms=duration_ms,
                    measurements=result.measurements,
                    remedy=result.remedy,
                )
            )
            log.info(
                "self-test",
                test=test.name,
                outcome=result.outcome.value,
                detail=result.message,
            )

        report = SelfTestReport(
            results=tuple(results),
            duration_s=self._clock.monotonic() - started,
            timestamp=self._clock.wall(),
        )
        self._last_report = report
        log.info("self-test complete", summary=report.summary())
        return report
