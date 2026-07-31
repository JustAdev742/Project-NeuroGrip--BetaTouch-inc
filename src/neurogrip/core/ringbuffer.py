"""Fixed-capacity buffers and running statistics.

Signal processing, metrics and UI sparklines all need "the last N values" with
bounded memory and no per-sample allocation. Written in pure Python so the core
runs without NumPy; the hot EMG path uses :class:`RunningStats`, which is O(1) per
sample and never allocates.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable, Iterator, Sequence

__all__ = ["RingBuffer", "RunningStats", "SlidingWindow", "median", "percentile"]


def median(values: Sequence[float]) -> float:
    """Median of ``values`` (returns ``0.0`` for an empty sequence)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) * 0.5


def percentile(values: Sequence[float], fraction: float) -> float:
    """Linear-interpolated percentile; ``fraction`` in ``[0, 1]``.

    Used for loop-jitter reporting (p95/p99), where the tail is what matters.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    if fraction <= 0:
        return ordered[0]
    if fraction >= 1:
        return ordered[-1]
    position = fraction * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[int(position)]
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


class RingBuffer:
    """Bounded FIFO of floats with cheap statistics.

    Unlike a bare :class:`collections.deque` this maintains a running sum, so
    :meth:`mean` is O(1) — it is called every UI frame for five channels.
    """

    __slots__ = ("_capacity", "_data", "_sum")

    def __init__(self, capacity: int, initial: Iterable[float] = ()) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._capacity = capacity
        self._data: deque[float] = deque(maxlen=capacity)
        self._sum = 0.0
        for value in initial:
            self.append(value)

    def append(self, value: float) -> None:
        value = float(value)
        if len(self._data) == self._capacity:
            self._sum -= self._data[0]
        self._data.append(value)
        self._sum += value

    def extend(self, values: Iterable[float]) -> None:
        for value in values:
            self.append(value)

    def clear(self) -> None:
        self._data.clear()
        self._sum = 0.0

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[float]:
        return iter(self._data)

    def __getitem__(self, index: int) -> float:
        return self._data[index]

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def is_full(self) -> bool:
        return len(self._data) == self._capacity

    @property
    def latest(self) -> float:
        return self._data[-1] if self._data else 0.0

    @property
    def oldest(self) -> float:
        return self._data[0] if self._data else 0.0

    def mean(self) -> float:
        return self._sum / len(self._data) if self._data else 0.0

    def rms(self) -> float:
        if not self._data:
            return 0.0
        return math.sqrt(sum(v * v for v in self._data) / len(self._data))

    def std(self) -> float:
        n = len(self._data)
        if n < 2:
            return 0.0
        mu = self.mean()
        return math.sqrt(sum((v - mu) ** 2 for v in self._data) / (n - 1))

    def minimum(self) -> float:
        return min(self._data) if self._data else 0.0

    def maximum(self) -> float:
        return max(self._data) if self._data else 0.0

    def median(self) -> float:
        return median(list(self._data))

    def percentile(self, fraction: float) -> float:
        return percentile(list(self._data), fraction)

    def to_list(self) -> list[float]:
        return list(self._data)

    def downsample(self, points: int) -> list[float]:
        """Reduce to ``points`` samples for sparkline rendering.

        Uses max-of-bucket rather than mean so that short EMG bursts stay visible
        at a glance — a plot that hides transients is worse than no plot.
        """
        data = list(self._data)
        if points <= 0 or not data:
            return []
        if len(data) <= points:
            return data
        bucket = len(data) / points
        out: list[float] = []
        for i in range(points):
            start = int(i * bucket)
            end = max(start + 1, int((i + 1) * bucket))
            out.append(max(data[start:end], key=abs))
        return out


class RunningStats:
    """O(1) online mean/variance (Welford) with min/max tracking.

    Used where the whole history matters but storing it does not: EMG rest
    baselines, loop timing over an entire session, training consistency scores.
    """

    __slots__ = ("_m2", "_max", "_mean", "_min", "_n")

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._n = 0
        self._mean = 0.0
        self._m2 = 0.0
        self._min = math.inf
        self._max = -math.inf

    def add(self, value: float) -> None:
        value = float(value)
        self._n += 1
        delta = value - self._mean
        self._mean += delta / self._n
        self._m2 += delta * (value - self._mean)
        self._min = min(self._min, value)
        self._max = max(self._max, value)

    @property
    def count(self) -> int:
        return self._n

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def variance(self) -> float:
        return self._m2 / (self._n - 1) if self._n > 1 else 0.0

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)

    @property
    def minimum(self) -> float:
        return self._min if self._n else 0.0

    @property
    def maximum(self) -> float:
        return self._max if self._n else 0.0

    @property
    def coefficient_of_variation(self) -> float:
        """Std/mean — the consistency metric shown in training statistics."""
        return self.std / self._mean if abs(self._mean) > 1e-9 else 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "count": float(self._n),
            "mean": self.mean,
            "std": self.std,
            "min": self.minimum,
            "max": self.maximum,
        }


class SlidingWindow:
    """Time-based window that evicts samples older than ``duration`` seconds.

    Feature extraction is specified in milliseconds (e.g. "200 ms MAV window"),
    not in sample counts, so that a change of sample rate does not silently
    change the filter behaviour.
    """

    __slots__ = ("_duration", "_samples")

    def __init__(self, duration: float) -> None:
        if duration <= 0:
            raise ValueError("duration must be positive")
        self._duration = duration
        self._samples: deque[tuple[float, float]] = deque()

    def add(self, timestamp: float, value: float) -> None:
        self._samples.append((timestamp, float(value)))
        self._evict(timestamp)

    def _evict(self, now: float) -> None:
        cutoff = now - self._duration
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()

    def values(self) -> list[float]:
        return [v for _, v in self._samples]

    def __len__(self) -> int:
        return len(self._samples)

    @property
    def duration(self) -> float:
        return self._duration

    @property
    def span(self) -> float:
        """Actual time covered by the retained samples."""
        if len(self._samples) < 2:
            return 0.0
        return self._samples[-1][0] - self._samples[0][0]

    def mean(self) -> float:
        return sum(v for _, v in self._samples) / len(self._samples) if self._samples else 0.0

    def clear(self) -> None:
        self._samples.clear()
