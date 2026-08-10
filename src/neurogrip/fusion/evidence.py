"""Evidence records.

Every input to a fusion decision is captured as a timestamped, weighted
:class:`Evidence` item. Three reasons this is worth the small cost:

* **Explainability.** The dashboard shows the user exactly what the hand knew.
* **Debugging.** "Why did it pick a pinch?" is answerable from a recording
  without reproducing the situation.
* **Staleness.** Confidence decays with age in one place, consistently, instead
  of each consumer inventing its own rule.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..core.types import clamp

__all__ = ["Evidence", "EvidenceSet"]


@dataclass(frozen=True, slots=True)
class Evidence:
    """One piece of evidence considered by the fusion layer."""

    #: ``emg``, ``vision``, ``depth``, ``proprioception``, …
    source: str
    #: What this evidence asserts (an intent kind, an object class, a distance).
    label: str
    confidence: float
    #: Relative weight in the combined score; ``0.0`` means informational only.
    weight: float = 1.0
    timestamp: float = 0.0
    detail: str = ""

    def age(self, now: float) -> float:
        return max(0.0, now - self.timestamp)

    def decayed_confidence(self, now: float, half_life: float = 0.4) -> float:
        """Confidence reduced by exponential decay with age.

        Half-life rather than a hard cut-off: evidence does not become worthless
        the instant it exceeds a threshold, it becomes gradually less relevant.
        The hard cut-offs live in :class:`~neurogrip.fusion.policy.FusionPolicy`
        and act as an additional floor.
        """
        if half_life <= 0:
            return self.confidence
        return clamp(self.confidence * math.pow(0.5, self.age(now) / half_life))

    def __str__(self) -> str:  # pragma: no cover - display helper
        suffix = f" ({self.detail})" if self.detail else ""
        return f"{self.source}={self.label} @{self.confidence:.2f}{suffix}"


@dataclass(frozen=True, slots=True)
class EvidenceSet:
    """All evidence available for one decision."""

    items: tuple[Evidence, ...] = field(default_factory=tuple)
    evaluated_at: float = 0.0

    def by_source(self, source: str) -> Evidence | None:
        for item in self.items:
            if item.source == source:
                return item
        return None

    def confidence_of(self, source: str, *, decayed: bool = True) -> float:
        item = self.by_source(source)
        if item is None:
            return 0.0
        return item.decayed_confidence(self.evaluated_at) if decayed else item.confidence

    @property
    def weighted_score(self) -> float:
        """Weight-normalised, age-decayed combined score.

        Provided for diagnostics and for planners that want a single number;
        :class:`~neurogrip.fusion.fusion.DecisionFusion` uses the policy's own
        combination rule, which treats missing vision differently.
        """
        total_weight = sum(item.weight for item in self.items)
        if total_weight <= 0:
            return 0.0
        return clamp(
            sum(item.decayed_confidence(self.evaluated_at) * item.weight for item in self.items)
            / total_weight
        )

    @property
    def sources(self) -> tuple[str, ...]:
        return tuple(item.source for item in self.items)

    def describe(self) -> tuple[str, ...]:
        """Lines for the UI's evidence panel, worst-first."""
        return tuple(
            f"{item.source}: {item.label} {item.decayed_confidence(self.evaluated_at) * 100:.0f}%"
            + (
                f" ({item.age(self.evaluated_at) * 1000:.0f} ms old)"
                if item.age(self.evaluated_at) > 0.05
                else ""
            )
            for item in sorted(self.items, key=lambda e: e.confidence)
        )
