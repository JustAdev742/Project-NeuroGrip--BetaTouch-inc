import numpy as np

from modules.vision.vision_mock import MockVision


def test_mock_vision_returns_detections() -> None:
	vision = MockVision(labels=["pen"])
	frame = np.zeros((240, 320, 3), dtype=np.uint8)
	detections = vision.detect(frame)
	assert isinstance(detections, list)
