from dataclasses import dataclass, field
from collections import deque
from enum import Enum
from typing import Dict, List, Optional, Tuple

from .grip_types import GripType, HandMode

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EMG_HISTORY_LENGTH = 150  # ~3 seconds at 50 Hz control loop
EVENT_LOG_MAX = 200


class SystemState(str, Enum):
	IDLE = "IDLE"
	DETECTING = "DETECTING"
	GRIPPING = "GRIPPING"
	HOLDING = "HOLDING"
	ERROR = "ERROR"
	MOCK_MODE = "MOCK_MODE"


class EMGIntent(str, Enum):
	REST = "REST"
	CLOSE = "CLOSE"
	OPEN = "OPEN"
	HOLD = "HOLD"
	CANCEL = "CANCEL"


@dataclass(frozen=True)
class Detection:
	label: str
	confidence: float
	bbox: Tuple[int, int, int, int]


@dataclass
class GripConfig:
	vision_threshold: float
	safe_grip: GripType
	empty_palm_label: str
	mapping: Dict[str, GripType]


@dataclass
class EventRecord:
	"""A single logged event (state transition, error, override, etc.)."""
	timestamp: float
	event_type: str  # "state_change", "grip_change", "error", "emg_override", "mode_change"
	detail: str

	def to_dict(self) -> dict:
		return {
			"timestamp": self.timestamp,
			"event_type": self.event_type,
			"detail": self.detail,
		}


@dataclass
class TrainingCapture:
	"""A captured frame + label for auto-training."""
	timestamp: float
	label: str
	image_path: str
	confidence_before: float  # what the model thought before correction

	def to_dict(self) -> dict:
		return {
			"timestamp": self.timestamp,
			"label": self.label,
			"image_path": self.image_path,
			"confidence_before": self.confidence_before,
		}


@dataclass
class SystemStatus:
	timestamp: float
	system_state: SystemState
	emg_intent: EMGIntent
	emg_signal_level: float
	emg_quality: float
	detected_objects: List[Detection] = field(default_factory=list)
	selected_grip: GripType = GripType.OPEN
	grip_selection_reason: str = "vision"
	servo_positions: Dict[str, float] = field(default_factory=dict)
	servo_feedback_available: bool = False
	last_error: Optional[str] = None
	mock_mode_active: bool = False

	# --- Metrics ---
	vision_fps: float = 0.0
	control_fps: float = 0.0
	uptime_seconds: float = 0.0
	emg_signal_history: deque = field(
		default_factory=lambda: deque(maxlen=EMG_HISTORY_LENGTH)
	)

	# --- Hand mode ---
	hand_mode: HandMode = HandMode.NORMAL
	available_grips: List[str] = field(default_factory=list)

	# --- Training ---
	training_captures_count: int = 0

	def to_dict(self) -> Dict[str, object]:
		return {
			"timestamp": self.timestamp,
			"system_state": self.system_state.value,
			"emg_intent": self.emg_intent.value,
			"emg_signal_level": self.emg_signal_level,
			"emg_quality": self.emg_quality,
			"detected_objects": [
				{
					"label": det.label,
					"confidence": det.confidence,
					"bbox": list(det.bbox),
				}
				for det in self.detected_objects
			],
			"selected_grip": self.selected_grip.value,
			"grip_selection_reason": self.grip_selection_reason,
			"servo_positions": self.servo_positions,
			"servo_feedback_available": self.servo_feedback_available,
			"last_error": self.last_error,
			"mock_mode_active": self.mock_mode_active,
			"vision_fps": round(self.vision_fps, 1),
			"control_fps": round(self.control_fps, 1),
			"uptime_seconds": round(self.uptime_seconds, 1),
			"emg_signal_history": list(self.emg_signal_history),
			"hand_mode": self.hand_mode.value,
			"available_grips": self.available_grips,
			"training_captures_count": self.training_captures_count,
		}
