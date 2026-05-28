from typing import List, Tuple

from app.models.grip_types import GripType
from app.models.status import Detection, EMGIntent, GripConfig


def select_grip(
	emg_intent: EMGIntent,
	detections: List[Detection],
	config: GripConfig,
) -> Tuple[GripType, str]:
	if emg_intent in (EMGIntent.OPEN, EMGIntent.CANCEL):
		return GripType.OPEN, "emg_open"

	if emg_intent != EMGIntent.CLOSE:
		return GripType.OPEN, "no_action"

	if not detections:
		return GripType.OPEN, "vision_empty"

	best = max(detections, key=lambda det: det.confidence)
	if best.label == config.empty_palm_label:
		return GripType.OPEN, "vision"
	if best.confidence < config.vision_threshold:
		return config.safe_grip, "fallback"

	grip = config.mapping.get(best.label, config.safe_grip)
	reason = "vision" if grip != config.safe_grip else "fallback"
	return grip, reason
