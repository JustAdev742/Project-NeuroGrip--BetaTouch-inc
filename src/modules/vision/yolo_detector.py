"""
YOLO Detector — Object detection for grasp classification.

Uses YOLOv8n (nano) which runs at ~8-12 FPS on Raspberry Pi 4 CPU.
If no model file exists at the configured path, it automatically
downloads yolov8n.pt from Ultralytics on first run.

The detector maps COCO class names to our grip labels via the
label_map in the config, so we can use the pretrained model directly
for common objects (cups, bottles, keyboards, balls, etc.).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from app.models.status import Detection
from app.utils.logging import get_logger
from .interface import VisionInterface


# COCO classes that map well to prosthetic grasp types
COCO_GRIP_MAP: Dict[str, str] = {
	# Pinch grip objects
	"cell phone": "pinch", "remote": "pinch", "scissors": "pinch",
	"toothbrush": "pinch", "fork": "pinch", "knife": "pinch", "spoon": "pinch",
	"mouse": "pinch",
	# Power grip objects
	"bottle": "power", "cup": "power", "wine glass": "power",
	"banana": "power", "hot dog": "power",
	# Cylindrical grip objects
	"baseball bat": "cylindrical", "tennis racket": "cylindrical",
	"umbrella": "cylindrical",
	# Tripod grip objects
	"apple": "tripod", "orange": "tripod", "donut": "tripod",
	# Hook grip objects
	"handbag": "hook", "suitcase": "hook", "backpack": "hook",
	# Sport objects
	"sports ball": "power", "frisbee": "power",
	# Keyboard/mouse (for keyboard mode context)
	"keyboard": "typing", "laptop": "typing",
	# Large objects
	"book": "power", "clock": "pinch", "vase": "cylindrical",
}


class YoloDetector(VisionInterface):
	"""
	YOLO-based object detector optimised for Raspberry Pi CPU inference.

	Auto-downloads YOLOv8n if the model file doesn't exist.
	Maps COCO class names to prosthetic-relevant grip labels.
	"""

	def __init__(
		self,
		model_path: str = "assets/models/yolov8n.pt",
		device: str = "cpu",
		confidence_threshold: float = 0.25,
		iou_threshold: float = 0.45,
		input_size: int = 320,
		labels: Sequence[str] | None = None,
		grip_label_map: Dict[str, str] | None = None,
	) -> None:
		self.logger = get_logger("vision")

		try:
			from ultralytics import YOLO
		except ImportError as exc:
			raise RuntimeError(
				"ultralytics is required for YoloDetector.\n"
				"Install: pip install ultralytics"
			) from exc

		model_file = Path(model_path)

		# Auto-download YOLOv8n if model doesn't exist
		if not model_file.exists():
			self.logger.info(f"Model not found at {model_path} — downloading YOLOv8n...")
			model_file.parent.mkdir(parents=True, exist_ok=True)
			# YOLO() auto-downloads when given a model name
			self.model = YOLO("yolov8n.pt")
			# Save to configured path for next time
			try:
				import shutil
				# Ultralytics downloads to ~/.cache/ultralytics or CWD
				cached = Path("yolov8n.pt")
				if cached.exists():
					shutil.move(str(cached), str(model_file))
					self.logger.info(f"Model saved to {model_file}")
			except Exception as exc:
				self.logger.warning(f"Could not move model file: {exc}")
		else:
			self.model = YOLO(str(model_file))

		self.device = device
		self.confidence_threshold = confidence_threshold
		self.iou_threshold = iou_threshold
		self.input_size = input_size
		self.labels = list(labels) if labels else []
		self.grip_label_map = grip_label_map or COCO_GRIP_MAP

		# Get COCO class names from the model
		self._class_names: Dict[int, str] = {}
		if hasattr(self.model, "names"):
			self._class_names = dict(self.model.names)

		self.logger.info(
			f"YOLOv8 ready: device={device}, imgsz={input_size}, "
			f"conf={confidence_threshold}, classes={len(self._class_names)}"
		)

	def detect(self, frame: np.ndarray) -> List[Detection]:
		"""Run inference and return detections with grip-relevant labels."""
		results = self.model.predict(
			source=frame,
			conf=self.confidence_threshold,
			iou=self.iou_threshold,
			device=self.device,
			imgsz=self.input_size,
			verbose=False,
		)

		detections: List[Detection] = []
		for result in results:
			boxes = result.boxes
			if boxes is None:
				continue
			for box in boxes:
				conf = float(box.conf[0])
				cls_id = int(box.cls[0])

				# Get COCO class name
				coco_name = self._class_names.get(cls_id, str(cls_id))

				# Map to grip-relevant label
				# If the COCO name is in our grip map, use the grip label
				# Otherwise use the raw COCO name (for training/display)
				grip_label = self.grip_label_map.get(coco_name, coco_name)

				# Also check custom labels list (for fine-tuned models)
				if self.labels and cls_id < len(self.labels):
					grip_label = self.labels[cls_id]

				x1, y1, x2, y2 = box.xyxy[0].tolist()
				detections.append(
					Detection(
						label=grip_label,
						confidence=conf,
						bbox=(int(x1), int(y1), int(x2), int(y2)),
					)
				)

		# If no objects detected, report empty palm
		if not detections:
			detections.append(
				Detection(label="empty_palm", confidence=0.95, bbox=(0, 0, 0, 0))
			)

		return detections
