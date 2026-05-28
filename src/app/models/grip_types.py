from enum import Enum


class GripType(str, Enum):
	OPEN = "OPEN"
	PINCH = "PINCH"
	POWER = "POWER"
	CYLINDRICAL = "CYLINDRICAL"
	TRIPOD = "TRIPOD"
	HOOK = "HOOK"
	SAFE = "SAFE"

	# --- Keyboard / Input mode grips ---
	TYPING_HOME = "TYPING_HOME"       # Fingers hovering over keys
	TYPING_TAP = "TYPING_TAP"         # Single finger tap
	MOUSE_CLICK = "MOUSE_CLICK"       # Index finger click position
	MOUSE_REST = "MOUSE_REST"         # Hand cupped over mouse
	GAMING_WASD = "GAMING_WASD"       # Fingers spread on WASD

	# --- Sports mode grips ---
	BASKETBALL_PALM = "BASKETBALL_PALM"     # Wide spread for ball control
	BASKETBALL_SHOOT = "BASKETBALL_SHOOT"   # Shooting release form
	TENNIS_FOREHAND = "TENNIS_FOREHAND"     # Racket grip forehand
	TENNIS_BACKHAND = "TENNIS_BACKHAND"     # Racket grip backhand
	BASEBALL_BAT = "BASEBALL_BAT"          # Bat grip
	CRICKET_BAT = "CRICKET_BAT"            # Cricket bat grip


class HandMode(str, Enum):
	"""Operating mode for the prosthetic hand."""
	NORMAL = "NORMAL"               # Standard EMG + vision grasp
	KEYBOARD_TYPING = "KEYBOARD_TYPING"
	KEYBOARD_GAMING = "KEYBOARD_GAMING"
	MOUSE = "MOUSE"
	SPORT_BASKETBALL = "SPORT_BASKETBALL"
	SPORT_TENNIS = "SPORT_TENNIS"
	SPORT_BASEBALL = "SPORT_BASEBALL"
	SPORT_CRICKET = "SPORT_CRICKET"


# Human-readable labels for the UI
MODE_LABELS = {
	HandMode.NORMAL: "Normal (AI + EMG)",
	HandMode.KEYBOARD_TYPING: "Keyboard — Typing",
	HandMode.KEYBOARD_GAMING: "Keyboard — Gaming",
	HandMode.MOUSE: "Mouse Control",
	HandMode.SPORT_BASKETBALL: "🏀 Basketball",
	HandMode.SPORT_TENNIS: "🎾 Tennis",
	HandMode.SPORT_BASEBALL: "⚾ Baseball",
	HandMode.SPORT_CRICKET: "🏏 Cricket",
}

# Which grips are available in each mode
MODE_GRIP_MAP = {
	HandMode.NORMAL: [
		GripType.OPEN, GripType.PINCH, GripType.POWER,
		GripType.CYLINDRICAL, GripType.TRIPOD, GripType.HOOK, GripType.SAFE,
	],
	HandMode.KEYBOARD_TYPING: [
		GripType.TYPING_HOME, GripType.TYPING_TAP, GripType.OPEN,
	],
	HandMode.KEYBOARD_GAMING: [
		GripType.GAMING_WASD, GripType.TYPING_TAP, GripType.OPEN,
	],
	HandMode.MOUSE: [
		GripType.MOUSE_REST, GripType.MOUSE_CLICK, GripType.OPEN,
	],
	HandMode.SPORT_BASKETBALL: [
		GripType.BASKETBALL_PALM, GripType.BASKETBALL_SHOOT,
		GripType.POWER, GripType.OPEN,
	],
	HandMode.SPORT_TENNIS: [
		GripType.TENNIS_FOREHAND, GripType.TENNIS_BACKHAND,
		GripType.POWER, GripType.OPEN,
	],
	HandMode.SPORT_BASEBALL: [
		GripType.BASEBALL_BAT, GripType.POWER, GripType.OPEN,
	],
	HandMode.SPORT_CRICKET: [
		GripType.CRICKET_BAT, GripType.POWER, GripType.OPEN,
	],
}
