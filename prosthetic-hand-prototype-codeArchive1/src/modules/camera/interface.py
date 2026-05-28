from abc import ABC, abstractmethod

import numpy as np


class CameraInterface(ABC):
	@abstractmethod
	def capture_frame(self) -> np.ndarray:
		"""Return BGR frame (H, W, 3)."""

	@abstractmethod
	def is_connected(self) -> bool:
		"""True if camera is ready."""

	@abstractmethod
	def close(self) -> None:
		"""Release camera resources."""
