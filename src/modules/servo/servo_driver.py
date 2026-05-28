"""
Servo Driver — Hardware PWM for 180° 1.5 kg-cm servos on Raspberry Pi.

Supports two GPIO backends (auto-detected):
  1. lgpio   — modern, works on Ubuntu 26.04 / kernel 6.x (preferred)
  2. RPi.GPIO — legacy, works on older Raspbian

Hardware specs:
  - 5x SG90-class servos, 180° range, 1.5 kg-cm stall torque
  - 50 Hz PWM frequency (20 ms period)
  - Pulse range: 0.5 ms (0°) to 2.5 ms (180°)
  - BCM pins: thumb=17, index=27, middle=22, ring=23, pinky=24

Motion uses linear interpolation over configurable duration to avoid
sudden jerks that could strip the 1.5 kg-cm gears.
"""

from __future__ import annotations

import math
import time
from typing import Dict, Optional

from app.models.grip_types import GripType
from app.utils.logging import get_logger
from ..grip.grip_library import GripLibrary
from .interface import ServoInterface

# Attempt to determine which GPIO library is available
_GPIO_BACKEND: Optional[str] = None
try:
	import lgpio  # noqa: F401
	_GPIO_BACKEND = "lgpio"
except ImportError:
	try:
		import RPi.GPIO  # noqa: F401
		_GPIO_BACKEND = "rpigpio"
	except ImportError:
		pass


class HandController(ServoInterface):
	"""
	Drives 5 × 180° / 1.5 kg-cm servos via hardware PWM on Raspberry Pi.

	Supports lgpio (Ubuntu 26.04+, kernel 6.x) and RPi.GPIO (legacy).
	Uses smooth linear interpolation for all movements to protect the
	low-torque gears from shock loads.
	"""

	def __init__(self, servo_config: dict, grip_library: GripLibrary) -> None:
		self.logger = get_logger("servo")
		self.servo_config = servo_config
		self.grip_library = grip_library
		self.driver = servo_config.get("driver", "mock")

		# Hardware specs for 180° 1.5 kg-cm servos
		self.frequency_hz = float(servo_config.get("frequency_hz", 50))
		self.min_pulse_us = float(servo_config.get("min_pulse_ms", 0.5)) * 1000.0  # 500 µs
		self.max_pulse_us = float(servo_config.get("max_pulse_ms", 2.5)) * 1000.0  # 2500 µs
		self.motion_duration = float(servo_config.get("motion_duration_ms", 500)) / 1000.0

		# Servo state
		self._positions = grip_library.get_positions(GripType.OPEN)
		self._start_positions: Dict[str, float] = dict(self._positions)
		self._target_positions: Dict[str, float] = dict(self._positions)
		self._motion_start: Optional[float] = None
		self._motion_duration = self.motion_duration
		self._stopped = False

		# GPIO handles (backend-specific)
		self._chip_handle: Optional[int] = None  # lgpio chip handle
		self._gpio_module = None                  # RPi.GPIO module ref
		self._pwm_handles: Dict[str, object] = {}
		self._pin_map: Dict[str, int] = {}

		# Build pin map from config
		for name, servo in servo_config.get("servos", {}).items():
			self._pin_map[name] = int(servo.get("pin", 0))

		if self.driver in ("gpio", "lgpio", "rpigpio"):
			self._setup_gpio()

	# ------------------------------------------------------------------
	# GPIO Setup (auto-detect backend)
	# ------------------------------------------------------------------

	def _setup_gpio(self) -> None:
		backend = self.driver if self.driver in ("lgpio", "rpigpio") else _GPIO_BACKEND
		if backend is None:
			raise RuntimeError(
				"No GPIO library available. Install lgpio (recommended for Ubuntu 26.04) "
				"or RPi.GPIO: sudo apt install python3-lgpio"
			)

		if backend == "lgpio":
			self._setup_lgpio()
		else:
			self._setup_rpigpio()

	def _setup_lgpio(self) -> None:
		"""Modern lgpio backend — works on Ubuntu 26.04 / kernel 6.18+."""
		import lgpio

		self.logger.info("Initializing lgpio backend for servo PWM")
		self._chip_handle = lgpio.gpiochip_open(0)

		for name, pin in self._pin_map.items():
			lgpio.gpio_claim_output(self._chip_handle, pin)
			# Start at open position
			angle = self._positions.get(name, 0.0)
			pulse_us = self._angle_to_pulse_us(angle)
			lgpio.tx_servo(self._chip_handle, pin, int(pulse_us), self.frequency_hz)

		self.driver = "lgpio"
		self.logger.info(f"lgpio: {len(self._pin_map)} servos initialized on gpiochip0")

	def _setup_rpigpio(self) -> None:
		"""Legacy RPi.GPIO backend — works on older Raspbian."""
		import RPi.GPIO as GPIO

		self.logger.info("Initializing RPi.GPIO backend for servo PWM")
		GPIO.setmode(GPIO.BCM)
		GPIO.setwarnings(False)
		self._gpio_module = GPIO

		for name, pin in self._pin_map.items():
			GPIO.setup(pin, GPIO.OUT)
			pwm = GPIO.PWM(pin, self.frequency_hz)
			angle = self._positions.get(name, 0.0)
			pwm.start(self._angle_to_duty_pct(angle))
			self._pwm_handles[name] = pwm

		self.driver = "rpigpio"
		self.logger.info(f"RPi.GPIO: {len(self._pin_map)} servos initialized (BCM mode)")

	# ------------------------------------------------------------------
	# Angle ↔ PWM conversion (180° servos, 0.5–2.5 ms pulse)
	# ------------------------------------------------------------------

	def _angle_to_pulse_us(self, angle: float) -> float:
		"""Convert angle (0–180°) to pulse width in microseconds."""
		angle = max(0.0, min(180.0, angle))
		return self.min_pulse_us + (self.max_pulse_us - self.min_pulse_us) * (angle / 180.0)

	def _angle_to_duty_pct(self, angle: float) -> float:
		"""Convert angle (0–180°) to duty cycle percentage (for RPi.GPIO)."""
		pulse_us = self._angle_to_pulse_us(angle)
		period_us = 1_000_000.0 / self.frequency_hz
		return (pulse_us / period_us) * 100.0

	# ------------------------------------------------------------------
	# Apply positions to hardware
	# ------------------------------------------------------------------

	def _apply_positions(self, positions: Dict[str, float]) -> None:
		"""Write current angles to the servo hardware."""
		if self._stopped:
			return

		if self.driver == "lgpio" and self._chip_handle is not None:
			import lgpio
			for name, angle in positions.items():
				pin = self._pin_map.get(name)
				if pin is None:
					continue
				pulse_us = self._angle_to_pulse_us(angle)
				try:
					lgpio.tx_servo(self._chip_handle, pin, int(pulse_us), self.frequency_hz)
				except Exception as exc:
					self.logger.warning(f"lgpio servo write failed: pin={pin}, err={exc}")

		elif self.driver == "rpigpio" and self._gpio_module:
			for name, angle in positions.items():
				pwm = self._pwm_handles.get(name)
				if pwm is None:
					continue
				try:
					pwm.ChangeDutyCycle(self._angle_to_duty_pct(angle))
				except Exception as exc:
					self.logger.warning(f"RPi.GPIO duty write failed: {name}, err={exc}")

	# ------------------------------------------------------------------
	# Motion control (smooth interpolation — protects 1.5 kg-cm gears)
	# ------------------------------------------------------------------

	def move_to_grip(self, grip: GripType, duration_ms: float) -> None:
		"""Begin smooth motion toward target grip over the given duration."""
		self._stopped = False
		self._start_positions = dict(self._positions)
		self._target_positions = self.grip_library.get_positions(grip)
		self._motion_start = time.perf_counter()
		# Minimum 50 ms to prevent shock loads on 1.5 kg-cm servos
		self._motion_duration = max(duration_ms / 1000.0, 0.05)

	def update_motion(self) -> None:
		"""Called every control tick to advance servo positions."""
		if self._motion_start is None or self._stopped:
			return

		now = time.perf_counter()
		t = (now - self._motion_start) / self._motion_duration

		if t >= 1.0:
			# Motion complete — snap to target
			self._positions = dict(self._target_positions)
			self._apply_positions(self._positions)
			self._motion_start = None
			return

		# Smooth ease-in-out interpolation (sine curve)
		# Gentler than linear on the 1.5 kg-cm gears
		t_smooth = 0.5 - 0.5 * math.cos(math.pi * t)

		blended: Dict[str, float] = {}
		for name, target in self._target_positions.items():
			start = self._start_positions.get(name, target)
			blended[name] = start + (target - start) * t_smooth
		self._positions = blended
		self._apply_positions(blended)

	def get_position(self) -> Dict[str, float]:
		return dict(self._positions)

	# ------------------------------------------------------------------
	# Safety
	# ------------------------------------------------------------------

	def emergency_stop(self) -> None:
		"""Immediately stop all servo motion and cut PWM signals."""
		self._motion_start = None
		self._stopped = True

		if self.driver == "lgpio" and self._chip_handle is not None:
			import lgpio
			for pin in self._pin_map.values():
				try:
					lgpio.tx_servo(self._chip_handle, pin, 0, 0)
				except Exception:
					pass

		elif self.driver == "rpigpio" and self._gpio_module:
			for pwm in self._pwm_handles.values():
				try:
					pwm.ChangeDutyCycle(0)
				except Exception:
					pass

		self.logger.warning("EMERGENCY STOP — all servos halted")

	def cleanup(self) -> None:
		"""Release GPIO resources. Call on shutdown."""
		self.emergency_stop()

		if self.driver == "lgpio" and self._chip_handle is not None:
			import lgpio
			try:
				lgpio.gpiochip_close(self._chip_handle)
			except Exception:
				pass
			self._chip_handle = None

		elif self.driver == "rpigpio" and self._gpio_module:
			try:
				for pwm in self._pwm_handles.values():
					pwm.stop()
				self._gpio_module.cleanup()
			except Exception:
				pass
			self._gpio_module = None

		self.logger.info("GPIO resources released")
