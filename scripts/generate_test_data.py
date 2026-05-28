from pathlib import Path

import cv2
import numpy as np


def generate_frame(label: str, size: tuple[int, int]) -> np.ndarray:
	width, height = size
	frame = np.zeros((height, width, 3), dtype=np.uint8)
	center = (width // 2, height // 2)

	if label == "pen":
		cv2.rectangle(frame, (center[0] - 140, center[1] - 8), (center[0] + 140, center[1] + 8), (0, 200, 255), -1)
	elif label == "ball":
		cv2.circle(frame, center, 70, (0, 120, 255), -1)
	elif label == "box":
		cv2.rectangle(frame, (center[0] - 110, center[1] - 80), (center[0] + 110, center[1] + 80), (0, 160, 80), -1)
	elif label == "mug":
		cv2.rectangle(frame, (center[0] - 70, center[1] - 90), (center[0] + 70, center[1] + 70), (255, 140, 0), -1)
		cv2.circle(frame, (center[0] + 90, center[1] - 10), 30, (255, 140, 0), 8)

	cv2.putText(frame, label, (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
	return frame


def main() -> None:
	out_dir = Path("tests/fixtures/sample_frames")
	out_dir.mkdir(parents=True, exist_ok=True)

	labels = ["pen", "ball", "box", "mug"]
	for label in labels:
		frame = generate_frame(label, (320, 240))
		path = out_dir / f"{label}.png"
		cv2.imwrite(str(path), frame)


if __name__ == "__main__":
	main()
