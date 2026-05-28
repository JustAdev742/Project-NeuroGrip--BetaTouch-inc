from app.models.grip_types import GripType, HandMode, MODE_GRIP_MAP
from modules.grip.grip_library import GripLibrary, GRIP_RATIOS


SERVO_CONFIG = {
	"servos": {
		"thumb":  {"pin": 17, "min_deg": 0, "max_deg": 180, "open_deg": 15, "close_deg": 160},
		"index":  {"pin": 27, "min_deg": 0, "max_deg": 180, "open_deg": 10, "close_deg": 170},
		"middle": {"pin": 22, "min_deg": 0, "max_deg": 180, "open_deg": 10, "close_deg": 170},
		"ring":   {"pin": 23, "min_deg": 0, "max_deg": 180, "open_deg": 10, "close_deg": 170},
		"pinky":  {"pin": 24, "min_deg": 0, "max_deg": 180, "open_deg": 10, "close_deg": 170},
	}
}


def test_all_grips_within_servo_limits() -> None:
	"""Every grip profile must produce angles within 0-180 degrees."""
	lib = GripLibrary(SERVO_CONFIG)
	for grip in GRIP_RATIOS:
		positions = lib.get_positions(grip)
		for finger, angle in positions.items():
			assert 0.0 <= angle <= 180.0, f"{grip.value}.{finger} = {angle}° exceeds limits"


def test_open_hand_near_minimum() -> None:
	lib = GripLibrary(SERVO_CONFIG)
	positions = lib.get_positions(GripType.OPEN)
	for finger, angle in positions.items():
		assert angle <= 20.0, f"OPEN.{finger} = {angle}° (expected near 0)"


def test_power_grip_near_maximum() -> None:
	lib = GripLibrary(SERVO_CONFIG)
	positions = lib.get_positions(GripType.POWER)
	for finger, angle in positions.items():
		assert angle >= 140.0, f"POWER.{finger} = {angle}° (expected near max)"


def test_all_modes_have_grip_definitions() -> None:
	"""Every mode must map to grips that have ratio definitions."""
	for mode, grips in MODE_GRIP_MAP.items():
		for grip in grips:
			assert grip in GRIP_RATIOS, f"Mode {mode.value} references {grip.value} but it has no ratio definition"


def test_pinch_thumb_and_index_closed() -> None:
	lib = GripLibrary(SERVO_CONFIG)
	pos = lib.get_positions(GripType.PINCH)
	assert pos["thumb"] > 100, "Pinch thumb should be mostly closed"
	assert pos["index"] > 100, "Pinch index should be mostly closed"
	assert pos["middle"] < 50, "Pinch middle should be mostly open"
