from app.models.grip_types import GripType
from app.models.status import Detection, EMGIntent, GripConfig
from modules.grip.grip_selector import select_grip


def test_select_grip_uses_mapping() -> None:
	config = GripConfig(
		vision_threshold=0.7,
		safe_grip=GripType.SAFE,
		empty_palm_label="empty_palm",
		mapping={"pen": GripType.PINCH},
	)
	detections = [Detection(label="pen", confidence=0.9, bbox=(0, 0, 10, 10))]
	grip, reason = select_grip(EMGIntent.CLOSE, detections, config)
	assert grip == GripType.PINCH
	assert reason == "vision"


def test_select_grip_fallback_on_low_confidence() -> None:
	config = GripConfig(
		vision_threshold=0.7,
		safe_grip=GripType.SAFE,
		empty_palm_label="empty_palm",
		mapping={"box": GripType.POWER},
	)
	detections = [Detection(label="box", confidence=0.4, bbox=(0, 0, 10, 10))]
	grip, reason = select_grip(EMGIntent.CLOSE, detections, config)
	assert grip == GripType.SAFE
	assert reason == "fallback"


def test_select_grip_open_on_emg_open() -> None:
	config = GripConfig(
		vision_threshold=0.7,
		safe_grip=GripType.SAFE,
		empty_palm_label="empty_palm",
		mapping={"pen": GripType.PINCH},
	)
	grip, reason = select_grip(EMGIntent.OPEN, [], config)
	assert grip == GripType.OPEN
	assert reason == "emg_open"


def test_select_grip_cancel_returns_open() -> None:
	config = GripConfig(
		vision_threshold=0.7,
		safe_grip=GripType.SAFE,
		empty_palm_label="empty_palm",
		mapping={"pen": GripType.PINCH},
	)
	grip, reason = select_grip(EMGIntent.CANCEL, [], config)
	assert grip == GripType.OPEN


def test_select_grip_empty_palm() -> None:
	config = GripConfig(
		vision_threshold=0.7,
		safe_grip=GripType.SAFE,
		empty_palm_label="empty_palm",
		mapping={},
	)
	detections = [Detection(label="empty_palm", confidence=0.95, bbox=(0, 0, 0, 0))]
	grip, reason = select_grip(EMGIntent.CLOSE, detections, config)
	assert grip == GripType.OPEN
	assert reason == "vision"


def test_select_grip_unknown_object_uses_safe() -> None:
	config = GripConfig(
		vision_threshold=0.7,
		safe_grip=GripType.SAFE,
		empty_palm_label="empty_palm",
		mapping={"pen": GripType.PINCH},
	)
	detections = [Detection(label="unknown_thing", confidence=0.85, bbox=(0, 0, 10, 10))]
	grip, reason = select_grip(EMGIntent.CLOSE, detections, config)
	assert grip == GripType.SAFE
	assert reason == "fallback"


def test_select_grip_no_detections() -> None:
	config = GripConfig(
		vision_threshold=0.7,
		safe_grip=GripType.SAFE,
		empty_palm_label="empty_palm",
		mapping={},
	)
	grip, reason = select_grip(EMGIntent.CLOSE, [], config)
	assert grip == GripType.OPEN
	assert reason == "vision_empty"
