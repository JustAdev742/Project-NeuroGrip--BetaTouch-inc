from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from app.models.grip_types import GripType, HandMode, MODE_GRIP_MAP
from app.models.status import (
	Detection,
	EMGIntent,
	EventRecord,
	GripConfig,
	SystemState,
	SystemStatus,
	TrainingCapture,
)
from app.utils.config import load_config
from app.utils.logging import get_logger
from app.utils.timing import FpsTracker, IntervalTimer, RateController
from modules.camera.camera_mock import MockCamera
from modules.camera.camera_opencv import OpenCVCamera
from modules.emg.emg_mock import MockEMG
from modules.emg.emg_reader import EMGReader
from modules.grip.grip_library import GripLibrary
from modules.grip.grip_selector import select_grip
from modules.servo.servo_driver import HandController
from modules.servo.servo_mock import ServoMock
from modules.vision.vision_mock import MockVision
from modules.vision.yolo_detector import YoloDetector

from .state_machine import StateInputs, StateMachine


class MainController:
	"""
	Orchestrates all modules: camera → vision → grip selector → servos.

	DESIGN PRINCIPLE: The user is always in control.
	  - EMG intent is the primary control signal.
	  - Vision only *recommends* a grip; the user must confirm with EMG CLOSE.
	  - Manual override from dashboard always takes priority.
	  - On any failure → SAFE grip immediately.
	  - Hand modes (keyboard, sports) bypass vision and use direct EMG→grip mapping.
	"""

	def __init__(self, config_path: str, dashboard: Optional[object] = None) -> None:
		self.config_path = config_path
		self.config = load_config(config_path)
		self.logger = get_logger("main")
		self.dashboard = dashboard
		self.state_machine = StateMachine()
		self.status = SystemStatus(
			timestamp=time.time(),
			system_state=SystemState.IDLE,
			emg_intent=EMGIntent.REST,
			emg_signal_level=0.0,
			emg_quality=1.0,
		)
		self.running = False
		self.camera = None
		self.vision = None
		self.emg = None
		self.servo = None
		self.grip_library: Optional[GripLibrary] = None
		self.grip_config: Optional[GripConfig] = None
		self._target_grip = GripType.OPEN
		self._motion_end_time: Optional[float] = None
		self._last_detections: list[Detection] = []
		self._start_time: float = 0.0
		self._last_frame: Optional[np.ndarray] = None

		# Hand mode (keyboard, sports, normal)
		self._hand_mode = HandMode.NORMAL

		# EMG override support (dashboard can inject intents)
		self._emg_override: Optional[EMGIntent] = None

		# Manual grip override (user explicitly picks a grip from dashboard)
		self._manual_grip: Optional[GripType] = None

		# Training captures
		self._training_captures: list[TrainingCapture] = []
		self._training_dir = Path("training_data")

		# Wire up state machine event callback → dashboard
		self.state_machine.set_transition_callback(self._on_state_transition)

	# ------------------------------------------------------------------
	# Hand Mode Management
	# ------------------------------------------------------------------

	def handle_mode_change(self, mode_str: str) -> bool:
		"""Switch hand operating mode. Returns True on success."""
		try:
			new_mode = HandMode(mode_str)
			old_mode = self._hand_mode
			self._hand_mode = new_mode
			self.status.hand_mode = new_mode
			self.status.available_grips = [
				g.value for g in MODE_GRIP_MAP.get(new_mode, [])
			]
			self._add_event("mode_change", f"{old_mode.value} → {new_mode.value}")
			self.logger.info(f"Hand mode changed: {old_mode.value} → {new_mode.value}")

			# On mode change, return to open hand for safety
			self._set_grip(GripType.OPEN, "mode_change")
			return True
		except (ValueError, KeyError):
			return False

	def handle_manual_grip(self, grip_str: str) -> bool:
		"""User explicitly selects a grip from the dashboard."""
		try:
			grip = GripType(grip_str)
			# Verify grip is available in current mode
			available = MODE_GRIP_MAP.get(self._hand_mode, [])
			if available and grip not in available:
				return False
			self._manual_grip = grip
			self._add_event("manual_grip", f"User selected {grip.value}")
			return True
		except (ValueError, KeyError):
			return False

	# ------------------------------------------------------------------
	# EMG Override (from dashboard)
	# ------------------------------------------------------------------

	def handle_emg_override(self, intent_str: str) -> bool:
		"""Accept an EMG intent override from the dashboard."""
		try:
			intent = EMGIntent(intent_str)
			self._emg_override = intent
			self._add_event("emg_override", f"Dashboard override → {intent.value}")
			self.logger.info(f"EMG override set: {intent.value}")
			return True
		except (ValueError, KeyError):
			return False

	# ------------------------------------------------------------------
	# Training Capture (auto-train for unknown objects)
	# ------------------------------------------------------------------

	def handle_training_capture(self, label: str) -> dict:
		"""
		Capture the current camera frame and save it with the given label
		for later fine-tuning. Returns metadata about the saved capture.
		"""
		if self._last_frame is None:
			return {"ok": False, "error": "No frame available"}

		# Create training data directory
		label_dir = self._training_dir / label
		label_dir.mkdir(parents=True, exist_ok=True)

		# Save frame
		timestamp = time.time()
		filename = f"{label}_{int(timestamp * 1000)}.jpg"
		filepath = label_dir / filename

		try:
			cv2.imwrite(str(filepath), self._last_frame)
		except Exception as exc:
			return {"ok": False, "error": str(exc)}

		# Determine what the model currently thinks this is
		conf_before = 0.0
		if self._last_detections:
			conf_before = self._last_detections[0].confidence

		capture = TrainingCapture(
			timestamp=timestamp,
			label=label,
			image_path=str(filepath),
			confidence_before=conf_before,
		)
		self._training_captures.append(capture)
		self.status.training_captures_count = len(self._training_captures)

		self._add_event("training", f"Captured '{label}' ({len(self._training_captures)} total)")
		self.logger.info(f"Training capture: label={label}, path={filepath}")

		return {
			"ok": True,
			"label": label,
			"path": str(filepath),
			"total_captures": len(self._training_captures),
			"confidence_before": conf_before,
		}

	def get_training_stats(self) -> dict:
		"""Return stats about captured training data."""
		if not self._training_dir.exists():
			return {"total": 0, "labels": {}}
		labels = {}
		for d in self._training_dir.iterdir():
			if d.is_dir():
				count = len(list(d.glob("*.jpg")))
				if count > 0:
					labels[d.name] = count
		return {"total": sum(labels.values()), "labels": labels}

	# ------------------------------------------------------------------
	# Service Construction
	# ------------------------------------------------------------------

	def _build_services(self, mock_mode: bool) -> None:
		vision_config = self.config.get("vision", {})
		emg_config = self.config.get("emg", {})
		servo_config = self.config.get("servo", {})

		if mock_mode:
			self.camera = MockCamera()
			self.vision = MockVision(labels=vision_config.get("labels", []))
			self.emg = MockEMG(emg_config)
		else:
			self.camera = OpenCVCamera()
			self.vision = YoloDetector(
				model_path=vision_config.get("model_path", ""),
				device=vision_config.get("device", "cpu"),
				confidence_threshold=vision_config.get("confidence_threshold", 0.25),
				iou_threshold=vision_config.get("iou_threshold", 0.45),
				input_size=vision_config.get("input_size", 416),
				labels=vision_config.get("labels", []),
			)
			self.emg = EMGReader(emg_config)

		self.grip_library = GripLibrary(servo_config)
		if mock_mode or servo_config.get("driver") == "mock":
			self.servo = ServoMock(servo_config, self.grip_library)
		else:
			self.servo = HandController(servo_config, self.grip_library)

		mapping = {
			key: GripType[value]
			for key, value in self.config.get("grip_mapping", {}).items()
			if value in GripType.__members__
		}
		grip_config = self.config.get("grip", {})
		self.grip_config = GripConfig(
			vision_threshold=float(grip_config.get("vision_threshold", 0.7)),
			safe_grip=GripType[grip_config.get("safe_grip", "SAFE")],
			empty_palm_label=grip_config.get("empty_palm_label", "empty_palm"),
			mapping=mapping,
		)

		# Set initial available grips for current mode
		self.status.available_grips = [
			g.value for g in MODE_GRIP_MAP.get(self._hand_mode, [])
		]

	# ------------------------------------------------------------------
	# Grip Helpers
	# ------------------------------------------------------------------

	def _set_grip(self, grip: GripType, reason: str) -> None:
		if self.servo is None:
			return
		if grip != self._target_grip:
			duration_ms = float(self.config.get("servo", {}).get("motion_duration_ms", 500))
			self.servo.move_to_grip(grip, duration_ms)
			self._motion_end_time = time.perf_counter() + (duration_ms / 1000.0)
			self._target_grip = grip
			self._add_event("grip_change", f"{grip.value} ({reason})")
		self.status.selected_grip = grip
		self.status.grip_selection_reason = reason

	# ------------------------------------------------------------------
	# Mode-specific Grip Selection
	# ------------------------------------------------------------------

	def _select_grip_for_mode(self, intent: EMGIntent) -> Optional[tuple]:
		"""
		For non-NORMAL modes, map EMG intent directly to mode-specific grips.
		Returns (GripType, reason) or None if NORMAL mode.
		"""
		mode = self._hand_mode
		if mode == HandMode.NORMAL:
			return None  # Use standard vision-based selection

		# User always controls: OPEN/CANCEL → open hand
		if intent in (EMGIntent.OPEN, EMGIntent.CANCEL):
			return GripType.OPEN, "user_open"

		grips = MODE_GRIP_MAP.get(mode, [GripType.OPEN])

		# For keyboard modes
		if mode == HandMode.KEYBOARD_TYPING:
			if intent == EMGIntent.CLOSE:
				return GripType.TYPING_TAP, "keyboard_tap"
			if intent == EMGIntent.HOLD:
				return GripType.TYPING_HOME, "keyboard_home"
			return GripType.TYPING_HOME, "keyboard_rest"

		if mode == HandMode.KEYBOARD_GAMING:
			if intent == EMGIntent.CLOSE:
				return GripType.TYPING_TAP, "gaming_action"
			if intent == EMGIntent.HOLD:
				return GripType.GAMING_WASD, "gaming_wasd"
			return GripType.GAMING_WASD, "gaming_rest"

		if mode == HandMode.MOUSE:
			if intent == EMGIntent.CLOSE:
				return GripType.MOUSE_CLICK, "mouse_click"
			return GripType.MOUSE_REST, "mouse_rest"

		# For sports modes: CLOSE = active grip, REST/HOLD = ready position
		sport_grips = {
			HandMode.SPORT_BASKETBALL: (GripType.BASKETBALL_PALM, GripType.BASKETBALL_SHOOT),
			HandMode.SPORT_TENNIS: (GripType.TENNIS_FOREHAND, GripType.TENNIS_BACKHAND),
			HandMode.SPORT_BASEBALL: (GripType.BASEBALL_BAT, GripType.POWER),
			HandMode.SPORT_CRICKET: (GripType.CRICKET_BAT, GripType.POWER),
		}
		if mode in sport_grips:
			hold_grip, action_grip = sport_grips[mode]
			if intent == EMGIntent.CLOSE:
				return action_grip, f"{mode.value}_action"
			if intent in (EMGIntent.HOLD, EMGIntent.REST):
				return hold_grip, f"{mode.value}_hold"

		# Fallback
		return GripType.OPEN, "mode_default"

	# ------------------------------------------------------------------
	# Event Helpers
	# ------------------------------------------------------------------

	def _on_state_transition(self, event: EventRecord) -> None:
		if self.dashboard and hasattr(self.dashboard, "add_event"):
			self.dashboard.add_event(event)

	def _add_event(self, event_type: str, detail: str) -> None:
		event = EventRecord(
			timestamp=time.time(),
			event_type=event_type,
			detail=detail,
		)
		if self.dashboard and hasattr(self.dashboard, "add_event"):
			self.dashboard.add_event(event)

	# ------------------------------------------------------------------
	# Main Loop
	# ------------------------------------------------------------------

	def run(self, mock_mode: bool = False) -> None:
		self._build_services(mock_mode)
		self.running = True
		self.status.mock_mode_active = mock_mode
		self._start_time = time.monotonic()

		control_hz = float(self.config.get("system", {}).get("control_hz", 50))
		vision_fps = float(self.config.get("system", {}).get("vision_fps", 12))
		control_rate = RateController(control_hz)
		vision_timer = IntervalTimer(vision_fps)
		control_fps_tracker = FpsTracker()
		vision_fps_tracker = FpsTracker()

		self._add_event("system", f"Started in {'mock' if mock_mode else 'hardware'} mode")
		self.logger.info(
			f"Main loop started: control={control_hz}Hz, vision={vision_fps}FPS, mock={mock_mode}"
		)

		while self.running:
			error_flag = False
			detection_ready = False
			grip_complete = False

			# ── EMG ──
			try:
				# Apply override if set (dashboard injection)
				if self._emg_override is not None:
					if hasattr(self.emg, "set_override"):
						self.emg.set_override(self._emg_override)
					self._emg_override = None

				signal = self.emg.read_signal()
				intent = self.emg.get_intent()
				quality = self.emg.signal_quality()
			except Exception as exc:
				self.status.last_error = str(exc)
				error_flag = True
				signal = 0.0
				intent = EMGIntent.REST
				quality = 0.0

			# ── Manual grip override (user explicitly picked a grip) ──
			if self._manual_grip is not None:
				self._set_grip(self._manual_grip, "manual")
				self._manual_grip = None
				# Skip vision-based selection this tick
				detection_ready = False

			# ── Vision (only in NORMAL mode + DETECTING state) ──
			elif self._hand_mode == HandMode.NORMAL:
				if self.state_machine.state == SystemState.DETECTING and vision_timer.ready():
					try:
						frame = self.camera.capture_frame()
						self._last_frame = frame
						self._last_detections = self.vision.detect(frame)
						detection_ready = True
						vision_fps_tracker.tick()
					except Exception as exc:
						self.status.last_error = str(exc)
						error_flag = True
			else:
				# Non-NORMAL modes: still capture frames for training/display
				if vision_timer.ready():
					try:
						frame = self.camera.capture_frame()
						self._last_frame = frame
						vision_fps_tracker.tick()
					except Exception:
						pass  # Camera failure in non-vision mode is non-critical

			# ── Grip Selection ──
			if intent in (EMGIntent.OPEN, EMGIntent.CANCEL):
				self._set_grip(GripType.OPEN, "emg_open")
			elif self._hand_mode != HandMode.NORMAL:
				# Mode-specific grip mapping (no AI needed)
				mode_result = self._select_grip_for_mode(intent)
				if mode_result:
					grip, reason = mode_result
					self._set_grip(grip, reason)
				if intent == EMGIntent.CLOSE:
					detection_ready = True  # Let state machine progress
			elif detection_ready and intent == EMGIntent.CLOSE and self.grip_config:
				# NORMAL mode: vision-based grip selection
				grip, reason = select_grip(intent, self._last_detections, self.grip_config)
				self._set_grip(grip, reason)

			# ── Servo motion ──
			if hasattr(self.servo, "update_motion"):
				self.servo.update_motion()

			if self._motion_end_time and time.perf_counter() >= self._motion_end_time:
				grip_complete = True
				self._motion_end_time = None

			# ── State machine ──
			self.state_machine.update(
				StateInputs(
					emg_intent=intent,
					detection_ready=detection_ready,
					grip_complete=grip_complete,
					error=error_flag,
				)
			)

			# ── Update status ──
			self.status.timestamp = time.time()
			self.status.system_state = self.state_machine.state
			self.status.emg_intent = intent
			self.status.emg_signal_level = signal
			self.status.emg_quality = quality
			self.status.detected_objects = list(self._last_detections)
			self.status.emg_signal_history.append(round(signal, 3))
			self.status.vision_fps = vision_fps_tracker.value
			self.status.control_fps = control_fps_tracker.tick()
			self.status.uptime_seconds = time.monotonic() - self._start_time
			self.status.hand_mode = self._hand_mode

			if self.servo:
				self.status.servo_positions = self.servo.get_position()

			# ── Error safe fallback ──
			if error_flag and self.servo and self.grip_config:
				self._set_grip(self.grip_config.safe_grip, "error_safe")
				self.servo.emergency_stop()

			# ── Push to dashboard ──
			if self.dashboard:
				self.dashboard.update_status(self.status)

			control_rate.sleep()

	# ------------------------------------------------------------------
	# Shutdown
	# ------------------------------------------------------------------

	def shutdown(self) -> None:
		self.running = False
		self._add_event("system", "Shutting down")
		self.logger.info("Shutdown complete")
		if self.camera:
			self.camera.close()
