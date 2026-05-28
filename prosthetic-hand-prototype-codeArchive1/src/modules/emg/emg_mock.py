from __future__ import annotations

import math
import time
from typing import Optional

from app.models.status import EMGIntent
from .interface import EMGInterface
from .emg_processing import EMGConfig, EMGProcessor


class MockEMG(EMGInterface):
	"""Simulated EMG service with realistic waveform patterns and override support."""

	def __init__(self, config: dict, cycle_seconds: float = 8.0) -> None:
		self.config = config
		self.processor = EMGProcessor(
			EMGConfig(
				smoothing_alpha=config.get("smoothing_alpha", 0.2),
				rest_threshold=config.get("rest_threshold", 0.2),
				open_threshold=config.get("open_threshold", 0.25),
				hold_threshold=config.get("hold_threshold", 0.4),
				close_threshold=config.get("close_threshold", 0.6),
				cancel_threshold=config.get("cancel_threshold", 0.9),
				min_quality=config.get("min_quality", 0.4),
			)
		)
		self.cycle_seconds = cycle_seconds
		self.intent = EMGIntent.REST
		self.quality = 1.0
		self.signal = 0.0

		# Override support: dashboard can inject a specific intent
		self._override_intent: Optional[EMGIntent] = None
		self._override_expires: float = 0.0
		self._override_duration: float = 2.5  # seconds

	def set_override(self, intent: EMGIntent) -> None:
		"""Set a temporary EMG override (expires after a few seconds)."""
		self._override_intent = intent
		self._override_expires = time.monotonic() + self._override_duration

	def read_signal(self) -> float:
		now = time.monotonic()

		# --- Check for active override ---
		if self._override_intent is not None and now < self._override_expires:
			# Synthesize a signal level that produces the overridden intent
			signal = self._intent_to_signal(self._override_intent)
			# Add subtle noise for realism
			noise = math.sin(now * 47.0) * 0.02
			signal = max(0.0, min(1.0, signal + noise))
			self.signal = signal
			self.intent, self.quality = self.processor.update(self.signal)
			# Force the intent in case processor disagrees
			self.intent = self._override_intent
			self.quality = 0.95
			return self.signal

		# Clear expired override
		if self._override_intent is not None and now >= self._override_expires:
			self._override_intent = None

		# --- Natural cycling pattern (more organic than step function) ---
		t = time.time()
		phase = (t % self.cycle_seconds) / self.cycle_seconds

		# Create a smooth, organic signal using overlapping sinusoids
		base = 0.0
		if phase < 0.15:
			# Resting phase (low signal)
			base = 0.12 + 0.03 * math.sin(t * 2.1)
		elif phase < 0.25:
			# Ramp up to close
			ramp = (phase - 0.15) / 0.10
			base = 0.12 + ramp * 0.63
		elif phase < 0.50:
			# Sustained close / hold (high signal)
			base = 0.75 + 0.05 * math.sin(t * 3.7)
		elif phase < 0.60:
			# Transition to hold level
			ramp = (phase - 0.50) / 0.10
			base = 0.75 - ramp * 0.30
		elif phase < 0.75:
			# Hold level
			base = 0.45 + 0.04 * math.sin(t * 2.9)
		elif phase < 0.85:
			# Ramp down to open
			ramp = (phase - 0.75) / 0.10
			base = 0.45 - ramp * 0.30
		else:
			# Back to rest
			base = 0.15 + 0.03 * math.sin(t * 1.8)

		# Add realistic EMG noise (high-frequency jitter)
		noise = (
			math.sin(t * 31.4) * 0.015
			+ math.sin(t * 67.2) * 0.010
			+ math.sin(t * 113.0) * 0.005
		)
		signal = max(0.0, min(1.0, base + noise))

		self.signal = signal
		self.intent, self.quality = self.processor.update(self.signal)
		return self.signal

	def get_intent(self) -> EMGIntent:
		return self.intent

	def signal_quality(self) -> float:
		return self.quality

	@staticmethod
	def _intent_to_signal(intent: EMGIntent) -> float:
		"""Map an intent to a representative signal level."""
		return {
			EMGIntent.REST: 0.12,
			EMGIntent.OPEN: 0.22,
			EMGIntent.HOLD: 0.50,
			EMGIntent.CLOSE: 0.75,
			EMGIntent.CANCEL: 0.95,
		}.get(intent, 0.12)
