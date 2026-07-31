"""Metrics registry.

Counters, gauges, histograms and rates, with bounded memory. Deliberately small:
this runs on the same CPU as a 200 Hz control loop, so a metrics system that
allocates per sample would be a poor trade.

Histograms use fixed buckets rather than storing samples, which makes their cost
constant and their memory footprint known — a property that matters far more on
an embedded target than exact percentiles do.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from ..core.clock import Clock
from ..core.ringbuffer import RingBuffer

__all__ = ["Counter", "Gauge", "Histogram", "MetricsRegistry", "RateMeter"]


@dataclass(slots=True)
class Counter:
    """Monotonically increasing count."""

    name: str
    value: int = 0
    description: str = ""

    def increment(self, amount: int = 1) -> None:
        self.value += amount

    def reset(self) -> None:
        self.value = 0


@dataclass(slots=True)
class Gauge:
    """A value that goes up and down, with min/max tracking."""

    name: str
    value: float = 0.0
    minimum: float = float("inf")
    maximum: float = float("-inf")
    description: str = ""
    unit: str = ""

    def set(self, value: float) -> None:
        self.value = value
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)

    def reset(self) -> None:
        self.minimum = float("inf")
        self.maximum = float("-inf")

    @property
    def range(self) -> tuple[float, float]:
        if self.minimum == float("inf"):
            return (0.0, 0.0)
        return (self.minimum, self.maximum)


class Histogram:
    """Fixed-bucket distribution with a mean and a bounded recent window."""

    #: Default buckets in milliseconds, covering the latencies this system cares
    #: about: sub-millisecond control steps up to a slow model inference.
    DEFAULT_BUCKETS = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 250.0, 1000.0)

    def __init__(
        self, name: str, buckets: tuple[float, ...] = DEFAULT_BUCKETS, *, unit: str = "ms"
    ) -> None:
        self.name = name
        self.unit = unit
        self._buckets = buckets
        self._counts = [0] * (len(buckets) + 1)
        self._sum = 0.0
        self._count = 0
        self._recent = RingBuffer(128)

    def observe(self, value: float) -> None:
        self._sum += value
        self._count += 1
        self._recent.append(value)
        for index, edge in enumerate(self._buckets):
            if value <= edge:
                self._counts[index] += 1
                return
        self._counts[-1] += 1

    @property
    def count(self) -> int:
        return self._count

    @property
    def mean(self) -> float:
        return self._sum / self._count if self._count else 0.0

    @property
    def p95(self) -> float:
        """95th percentile over the recent window (exact, small sample)."""
        return self._recent.percentile(0.95)

    @property
    def maximum(self) -> float:
        return self._recent.maximum()

    def distribution(self) -> list[tuple[str, int]]:
        """Bucket labels and counts, for the diagnostics histogram widget."""
        labels = [f"≤{edge:g}" for edge in self._buckets] + [f">{self._buckets[-1]:g}"]
        return list(zip(labels, self._counts))

    def reset(self) -> None:
        self._counts = [0] * (len(self._buckets) + 1)
        self._sum = 0.0
        self._count = 0
        self._recent.clear()


class RateMeter:
    """Events per second over a sliding window."""

    def __init__(self, name: str, clock: Clock, *, window_s: float = 5.0) -> None:
        self.name = name
        self._clock = clock
        self._window = window_s
        self._events: list[float] = []

    def mark(self, count: int = 1) -> None:
        now = self._clock.monotonic()
        self._events.extend([now] * count)
        self._prune(now)

    def _prune(self, now: float) -> None:
        cutoff = now - self._window
        # The list is time-ordered, so a single scan from the front suffices.
        index = len(self._events)
        for position, timestamp in enumerate(self._events):
            if timestamp >= cutoff:
                index = position
                break
        if index:
            del self._events[:index]

    @property
    def rate(self) -> float:
        now = self._clock.monotonic()
        self._prune(now)
        return len(self._events) / self._window


class MetricsRegistry:
    """Central registry. Thread-safe; metric objects themselves are not."""

    def __init__(self, clock: Clock) -> None:
        self._clock = clock
        self._counters: dict[str, Counter] = {}
        self._gauges: dict[str, Gauge] = {}
        self._histograms: dict[str, Histogram] = {}
        self._rates: dict[str, RateMeter] = {}
        self._lock = threading.Lock()

    def counter(self, name: str, description: str = "") -> Counter:
        with self._lock:
            return self._counters.setdefault(name, Counter(name, description=description))

    def gauge(self, name: str, *, unit: str = "", description: str = "") -> Gauge:
        with self._lock:
            return self._gauges.setdefault(
                name, Gauge(name, unit=unit, description=description)
            )

    def histogram(self, name: str, *, unit: str = "ms") -> Histogram:
        with self._lock:
            existing = self._histograms.get(name)
            if existing is None:
                existing = Histogram(name, unit=unit)
                self._histograms[name] = existing
            return existing

    def rate(self, name: str, *, window_s: float = 5.0) -> RateMeter:
        with self._lock:
            existing = self._rates.get(name)
            if existing is None:
                existing = RateMeter(name, self._clock, window_s=window_s)
                self._rates[name] = existing
            return existing

    def snapshot(self) -> dict[str, dict[str, float | int]]:
        """Flat view of every metric, for the diagnostics screen and the log."""
        with self._lock:
            return {
                "counters": {c.name: c.value for c in self._counters.values()},
                "gauges": {g.name: round(g.value, 4) for g in self._gauges.values()},
                "histograms": {
                    h.name: {
                        "count": h.count,
                        "mean": round(h.mean, 3),
                        "p95": round(h.p95, 3),
                        "max": round(h.maximum, 3),
                    }
                    for h in self._histograms.values()
                },
                "rates": {r.name: round(r.rate, 2) for r in self._rates.values()},
            }

    def reset(self) -> None:
        with self._lock:
            for counter in self._counters.values():
                counter.reset()
            for gauge in self._gauges.values():
                gauge.reset()
            for histogram in self._histograms.values():
                histogram.reset()
