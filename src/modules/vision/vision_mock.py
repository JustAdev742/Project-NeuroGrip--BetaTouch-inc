import random
import time
from typing import List

import numpy as np

from app.models.status import Detection
from .interface import VisionInterface


class MockVision(VisionInterface):
	"""Simulated vision detector with varied confidence levels and realistic behavior."""

	def __init__(
		self,
		labels: List[str] | None = None,
		seed: int = 42,
		inference_delay_ms: float = 30.0,
	) -> None:
		self.labels = labels or ["pen", "ball", "box", "mug", "empty_palm"]
		self.rng = random.Random(seed)
		self.index = 0
		self.inference_delay = inference_delay_ms / 1000.0
		self._last_switch = time.monotonic()
		self._switch_interval = 4.0  # seconds per object in cycle
		self._current_label = self.labels[0] if self.labels else "empty_palm"

		# Per the paper: pinch objects have highest accuracy, tool/power are lower
		self._confidence_ranges = {
			"pen": (0.82, 0.96),
			"key": (0.80, 0.95),
			"card": (0.78, 0.94),
			"ball": (0.70, 0.90),
			"cup": (0.72, 0.88),
			"mug": (0.68, 0.87),
			"box": (0.65, 0.85),
			"tool": (0.55, 0.78),    # Lowest — often confused with power grasp
			"handle": (0.58, 0.80),
			"marble": (0.75, 0.92),
			"nut": (0.73, 0.90),
			"book": (0.67, 0.86),
			"empty_palm": (0.90, 0.99),
		}

	def detect(self, frame: np.ndarray) -> List[Detection]:
		# Simulate inference latency
		if self.inference_delay > 0:
			time.sleep(self.inference_delay)

		# Switch object on a timer (more natural than every-frame cycling)
		now = time.monotonic()
		if now - self._last_switch >= self._switch_interval:
			self.index += 1
			self._current_label = self.labels[self.index % len(self.labels)]
			self._last_switch = now

		label = self._current_label

		# Occasionally return empty palm or low-confidence detection
		roll = self.rng.random()
		if roll < 0.05:
			# 5% chance of empty palm interruption
			return [Detection(label="empty_palm", confidence=0.95, bbox=(0, 0, 0, 0))]
		if roll < 0.10:
			# 5% chance of unknown/low-confidence
			return [Detection(label=label, confidence=self.rng.uniform(0.25, 0.55), bbox=(80, 80, 260, 260))]

		# Normal detection with label-specific confidence range
		conf_min, conf_max = self._confidence_ranges.get(label, (0.60, 0.85))
		confidence = self.rng.uniform(conf_min, conf_max)

		# Generate a plausible bounding box centered on frame
		h, w = frame.shape[:2] if frame is not None and len(frame.shape) >= 2 else (480, 640)
		cx, cy = w // 2, h // 2
		bw = self.rng.randint(80, 180)
		bh = self.rng.randint(80, 180)
		bbox = (cx - bw // 2, cy - bh // 2, cx + bw // 2, cy + bh // 2)

		return [Detection(label=label, confidence=round(confidence, 3), bbox=bbox)]
