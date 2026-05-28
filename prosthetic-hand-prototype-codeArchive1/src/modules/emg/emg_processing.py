from dataclasses import dataclass

from app.models.status import EMGIntent


@dataclass
class EMGConfig:
	smoothing_alpha: float
	rest_threshold: float
	open_threshold: float
	hold_threshold: float
	close_threshold: float
	cancel_threshold: float
	min_quality: float


class EMGProcessor:
	def __init__(self, config: EMGConfig) -> None:
		self.config = config
		self.filtered = 0.0
		self.last_intent = EMGIntent.REST
		self.last_quality = 1.0

	def update(self, signal: float) -> tuple[EMGIntent, float]:
		signal = max(0.0, min(1.0, signal))
		self.filtered = (
			self.config.smoothing_alpha * signal
			+ (1.0 - self.config.smoothing_alpha) * self.filtered
		)
		noise = abs(signal - self.filtered)
		quality = max(0.0, min(1.0, 1.0 - (noise * 2.0)))
		self.last_quality = quality

		if self.filtered >= self.config.cancel_threshold:
			intent = EMGIntent.CANCEL
		elif self.filtered >= self.config.close_threshold:
			intent = EMGIntent.CLOSE
		elif self.filtered <= self.config.open_threshold:
			intent = EMGIntent.OPEN if self.last_intent in (EMGIntent.CLOSE, EMGIntent.HOLD) else EMGIntent.REST
		elif self.filtered >= self.config.hold_threshold and self.last_intent in (EMGIntent.CLOSE, EMGIntent.HOLD):
			intent = EMGIntent.HOLD
		else:
			intent = EMGIntent.REST

		if quality < self.config.min_quality:
			intent = EMGIntent.REST

		self.last_intent = intent
		return intent, quality
