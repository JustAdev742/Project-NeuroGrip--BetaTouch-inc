"""
Grip Library — Servo position profiles for all grip types and hand modes.

Hardware: 5x 180-degree servos, 1.5 kg-cm torque each.
Servo mapping:  thumb (BCM 17), index (BCM 27), middle (BCM 22),
                ring (BCM 23), pinky (BCM 24).

Each grip is defined as a ratio dict: 0.0 = fully open, 1.0 = fully closed.
The library converts ratios → degrees using each servo's open_deg / close_deg
from config, clamped to min_deg / max_deg (always within 0–180°).
"""

from typing import Dict

from app.models.grip_types import GripType


# ──────────────────────────────────────────────────────────────────────
# Grip ratio definitions
#
# Each entry maps finger name → flex ratio [0.0 = open, 1.0 = closed].
# These are hardware-independent — the GripLibrary converts to degrees.
# ──────────────────────────────────────────────────────────────────────

GRIP_RATIOS: Dict[GripType, Dict[str, float]] = {
	# ── Standard grasps (paper-inspired) ──
	GripType.OPEN: {
		"thumb": 0.0, "index": 0.0, "middle": 0.0, "ring": 0.0, "pinky": 0.0,
	},
	GripType.SAFE: {
		"thumb": 0.35, "index": 0.35, "middle": 0.35, "ring": 0.35, "pinky": 0.35,
	},
	GripType.PINCH: {
		"thumb": 0.9, "index": 0.9, "middle": 0.15, "ring": 0.15, "pinky": 0.15,
	},
	GripType.POWER: {
		"thumb": 1.0, "index": 1.0, "middle": 1.0, "ring": 1.0, "pinky": 1.0,
	},
	GripType.CYLINDRICAL: {
		"thumb": 0.75, "index": 0.80, "middle": 0.80, "ring": 0.80, "pinky": 0.80,
	},
	GripType.TRIPOD: {
		"thumb": 0.85, "index": 0.85, "middle": 0.85, "ring": 0.15, "pinky": 0.15,
	},
	GripType.HOOK: {
		"thumb": 0.0, "index": 1.0, "middle": 1.0, "ring": 1.0, "pinky": 1.0,
	},

	# ── Keyboard / input mode ──
	GripType.TYPING_HOME: {
		# Fingers slightly curved, hovering over keys
		"thumb": 0.20, "index": 0.30, "middle": 0.30, "ring": 0.30, "pinky": 0.30,
	},
	GripType.TYPING_TAP: {
		# Index finger extended down for key press
		"thumb": 0.20, "index": 0.60, "middle": 0.30, "ring": 0.30, "pinky": 0.30,
	},
	GripType.MOUSE_REST: {
		# Hand cupped over mouse body
		"thumb": 0.15, "index": 0.20, "middle": 0.25, "ring": 0.35, "pinky": 0.40,
	},
	GripType.MOUSE_CLICK: {
		# Index finger pressing down
		"thumb": 0.15, "index": 0.50, "middle": 0.25, "ring": 0.35, "pinky": 0.40,
	},
	GripType.GAMING_WASD: {
		# Fingers spread on WASD keys — more extended
		"thumb": 0.10, "index": 0.35, "middle": 0.35, "ring": 0.35, "pinky": 0.20,
	},

	# ── Basketball ──
	GripType.BASKETBALL_PALM: {
		# Wide spread for maximum ball surface contact
		"thumb": 0.50, "index": 0.55, "middle": 0.55, "ring": 0.55, "pinky": 0.50,
	},
	GripType.BASKETBALL_SHOOT: {
		# Shooting release — fingers extend to follow through
		"thumb": 0.15, "index": 0.10, "middle": 0.10, "ring": 0.10, "pinky": 0.10,
	},

	# ── Tennis ──
	GripType.TENNIS_FOREHAND: {
		# Eastern forehand grip — firm wrap around handle
		"thumb": 0.70, "index": 0.85, "middle": 0.90, "ring": 0.90, "pinky": 0.85,
	},
	GripType.TENNIS_BACKHAND: {
		# Backhand — thumb more extended along handle
		"thumb": 0.40, "index": 0.85, "middle": 0.90, "ring": 0.90, "pinky": 0.85,
	},

	# ── Baseball ──
	GripType.BASEBALL_BAT: {
		# Strong bat grip — all fingers wrapped tight
		"thumb": 0.85, "index": 0.95, "middle": 0.95, "ring": 0.95, "pinky": 0.90,
	},

	# ── Cricket ──
	GripType.CRICKET_BAT: {
		# Cricket bat handle — V-grip with thumb/index, others wrap
		"thumb": 0.60, "index": 0.70, "middle": 0.90, "ring": 0.90, "pinky": 0.85,
	},
}


class GripLibrary:
	"""Converts grip ratios to servo angles based on hardware config."""

	def __init__(self, servo_config: dict) -> None:
		self.servo_config = servo_config
		self.servos = servo_config.get("servos", {})

	def _angle(self, servo_name: str, ratio: float) -> float:
		"""Convert a flex ratio [0..1] to a clamped servo angle in degrees."""
		servo = self.servos[servo_name]
		open_deg = float(servo.get("open_deg", 0.0))
		close_deg = float(servo.get("close_deg", 180.0))
		angle = open_deg + (close_deg - open_deg) * ratio

		# Hardware limits: 180° servos, 1.5 kg-cm torque
		min_deg = float(servo.get("min_deg", 0.0))
		max_deg = float(servo.get("max_deg", 180.0))
		return max(min_deg, min(max_deg, angle))

	def _ratios_for_grip(self, grip: GripType) -> Dict[str, float]:
		"""Look up the flex ratios for a given grip type."""
		ratios = GRIP_RATIOS.get(grip)
		if ratios:
			return ratios
		# Fallback: open hand
		return {name: 0.0 for name in self.servos}

	def get_positions(self, grip: GripType) -> Dict[str, float]:
		"""Get final servo angles (degrees) for a grip type."""
		ratios = self._ratios_for_grip(grip)
		return {
			name: self._angle(name, ratios.get(name, 0.0))
			for name in self.servos
		}

	def list_available_grips(self) -> list:
		"""Return all grip types that have defined profiles."""
		return list(GRIP_RATIOS.keys())
