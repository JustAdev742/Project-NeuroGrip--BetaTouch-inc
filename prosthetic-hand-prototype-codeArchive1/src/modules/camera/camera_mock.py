import time
from typing import List

import cv2
import numpy as np

from .interface import CameraInterface


class MockCamera(CameraInterface):
	def __init__(
		self,
		width: int = 640,
		height: int = 480,
		labels: List[str] | None = None,
	) -> None:
		self.width = width
		self.height = height
		self.labels = labels or ["pen", "ball", "box", "mug", "empty_palm"]
		self.index = 0

	def capture_frame(self) -> np.ndarray:
		label = self.labels[self.index % len(self.labels)]
		self.index += 1
		frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

		center = (self.width // 2, self.height // 2)
		if label in {"pen", "key", "card"}:
			cv2.rectangle(frame, (center[0] - 150, center[1] - 10), (center[0] + 150, center[1] + 10), (0, 200, 255), -1)
		elif label in {"ball", "marble", "nut"}:
			cv2.circle(frame, center, 80, (0, 120, 255), -1)
		elif label in {"box", "book"}:
			cv2.rectangle(frame, (center[0] - 120, center[1] - 80), (center[0] + 120, center[1] + 80), (0, 160, 80), -1)
		elif label in {"mug", "cup", "handle", "tool"}:
			cv2.rectangle(frame, (center[0] - 80, center[1] - 100), (center[0] + 80, center[1] + 80), (255, 140, 0), -1)
			cv2.circle(frame, (center[0] + 100, center[1] - 10), 35, (255, 140, 0), 8)

		cv2.putText(
			frame,
			f"Mock: {label}",
			(20, self.height - 30),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.8,
			(255, 255, 255),
			2,
			cv2.LINE_AA,
		)
		time.sleep(0.05)
		return frame

	def is_connected(self) -> bool:
		return True

	def close(self) -> None:
		return None
