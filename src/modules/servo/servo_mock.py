"""
Servo Mock — Simulated servo driver for development/demo.

Uses the same sine-curve ease-in-out as the real HandController
so dashboard visualizations look identical to hardware behavior.
"""

import math
import time
from typing import Dict, Optional

from app.models.grip_types import GripType
from ..grip.grip_library import GripLibrary
from .interface import ServoInterface


class ServoMock(ServoInterface):
	"""Simulated 5-finger servo driver matching real hardware timing."""

	def __init__(self, servo_config: dict, grip_library: GripLibrary) -> None:
		self.servo_config = servo_config
		self.grip_library = grip_library
		self._positions = grip_library.get_positions(GripType.OPEN)
		self._start_positions: Dict[str, float] = dict(self._positions)
		self._target_positions: Dict[str, float] = dict(self._positions)
		self._motion_start: Optional[float] = None
		self._motion_duration = float(servo_config.get("motion_duration_ms", 500)) / 1000.0

	def move_to_grip(self, grip: GripType, duration_ms: float) -> None:
		self._start_positions = dict(self._positions)
		self._target_positions = self.grip_library.get_positions(grip)
		self._motion_start = time.perf_counter()
		self._motion_duration = max(duration_ms / 1000.0, 0.05)

	def update_motion(self) -> None:
		if self._motion_start is None:
			return
		now = time.perf_counter()
		t = (now - self._motion_start) / self._motion_duration
		if t >= 1.0:
			self._positions = dict(self._target_positions)
			self._motion_start = None
			return

		# Sine ease-in-out (matches real servo driver)
		t_smooth = 0.5 - 0.5 * math.cos(math.pi * t)

		blended: Dict[str, float] = {}
		for name, target in self._target_positions.items():
			start = self._start_positions.get(name, target)
			blended[name] = start + (target - start) * t_smooth
		self._positions = blended

	def get_position(self) -> Dict[str, float]:
		return dict(self._positions)

	def emergency_stop(self) -> None:
		self._motion_start = None

	def cleanup(self) -> None:
		self._motion_start = None
