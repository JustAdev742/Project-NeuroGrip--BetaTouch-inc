"""
OpenCV Camera — Linux V4L2 and CSI camera support for Raspberry Pi.

On Ubuntu 26.04 (kernel 6.18), the Pi camera appears as /dev/video0
via the V4L2 driver. We prefer the V4L2 backend explicitly for
reliability on headless Linux systems.
"""

from __future__ import annotations

import platform
import sys
from typing import Optional

import cv2
import numpy as np

from app.utils.logging import get_logger
from .interface import CameraInterface


class OpenCVCamera(CameraInterface):
	"""
	Real camera capture via OpenCV.

	On Linux (Raspberry Pi), uses the V4L2 backend explicitly for
	reliable CSI and USB camera access. Falls back to the default
	backend on other platforms.
	"""

	def __init__(
		self,
		device_index: int = 0,
		width: int = 640,
		height: int = 480,
		fps: int = 30,
	) -> None:
		self.logger = get_logger("camera")

		# Choose backend: V4L2 on Linux for reliability, default elsewhere
		if sys.platform == "linux":
			backend = cv2.CAP_V4L2
			self.logger.info(f"Opening camera {device_index} with V4L2 backend")
		else:
			backend = cv2.CAP_ANY
			self.logger.info(f"Opening camera {device_index} with default backend")

		self.cap: Optional[cv2.VideoCapture] = cv2.VideoCapture(device_index, backend)
		if not self.cap.isOpened():
			# Retry with default backend
			self.logger.warning("V4L2 open failed, retrying with default backend")
			self.cap = cv2.VideoCapture(device_index)

		if not self.cap.isOpened():
			raise RuntimeError(
				f"Camera {device_index} could not be opened. "
				f"On Linux, check: ls /dev/video* and permissions (sudo usermod -aG video $USER)"
			)

		self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
		self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
		self.cap.set(cv2.CAP_PROP_FPS, fps)

		# On Pi, reduce buffer size to minimize latency
		self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

		actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
		actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
		self.logger.info(f"Camera ready: {actual_w}x{actual_h} @ {fps}fps")

	def capture_frame(self) -> np.ndarray:
		if not self.cap or not self.cap.isOpened():
			raise RuntimeError("Camera is not connected")
		ok, frame = self.cap.read()
		if not ok or frame is None:
			raise RuntimeError("Camera read failed — check cable or driver")
		return frame

	def is_connected(self) -> bool:
		return bool(self.cap and self.cap.isOpened())

	def close(self) -> None:
		if self.cap:
			self.cap.release()
			self.cap = None
			self.logger.info("Camera released")
