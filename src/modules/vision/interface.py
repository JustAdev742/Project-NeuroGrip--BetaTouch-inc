from abc import ABC, abstractmethod
from typing import List

import numpy as np

from app.models.status import Detection


class VisionInterface(ABC):
	@abstractmethod
	def detect(self, frame: np.ndarray) -> List[Detection]:
		"""Detect objects in frame and return a list of detections."""
