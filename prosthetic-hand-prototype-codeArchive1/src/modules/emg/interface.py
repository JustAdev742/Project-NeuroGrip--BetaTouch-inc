from abc import ABC, abstractmethod

from app.models.status import EMGIntent


class EMGInterface(ABC):
	@abstractmethod
	def read_signal(self) -> float:
		"""Return EMG signal level (0.0-1.0)."""

	@abstractmethod
	def get_intent(self) -> EMGIntent:
		"""Return detected intent."""

	@abstractmethod
	def signal_quality(self) -> float:
		"""Return confidence in intent reading (0.0-1.0)."""
