from abc import ABC, abstractmethod
from typing import Dict

from app.models.grip_types import GripType


class ServoInterface(ABC):
	@abstractmethod
	def move_to_grip(self, grip: GripType, duration_ms: float) -> None:
		"""Execute trajectory to reach target grip."""

	@abstractmethod
	def get_position(self) -> Dict[str, float]:
		"""Return servo positions."""

	@abstractmethod
	def emergency_stop(self) -> None:
		"""Stop all motion immediately."""
