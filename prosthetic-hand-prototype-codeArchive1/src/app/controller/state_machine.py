from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from app.models.status import EMGIntent, EventRecord, SystemState


@dataclass
class StateInputs:
	emg_intent: EMGIntent
	detection_ready: bool
	grip_complete: bool
	error: bool


# Minimum time (seconds) spent in ERROR before auto-recovering to IDLE.
ERROR_COOLDOWN_S = 1.0


class StateMachine:
	def __init__(self, initial_state: SystemState = SystemState.IDLE) -> None:
		self.state = initial_state
		self._prev_state: Optional[SystemState] = None
		self._error_entered: float = 0.0
		self._transition_history: List[EventRecord] = []
		self._on_transition: Optional[Callable[[EventRecord], None]] = None

	# ------------------------------------------------------------------
	# Public helpers
	# ------------------------------------------------------------------

	def set_transition_callback(self, callback: Callable[[EventRecord], None]) -> None:
		"""Register a callback invoked on every state transition."""
		self._on_transition = callback

	@property
	def transition_history(self) -> List[EventRecord]:
		return list(self._transition_history)

	# ------------------------------------------------------------------
	# Core update
	# ------------------------------------------------------------------

	def update(self, inputs: StateInputs) -> SystemState:
		old_state = self.state

		# --- Error takes priority ---
		if inputs.error:
			if self.state != SystemState.ERROR:
				self._error_entered = time.monotonic()
			self.state = SystemState.ERROR
			self._emit(old_state)
			return self.state

		# --- Error recovery with cooldown ---
		if self.state == SystemState.ERROR:
			elapsed = time.monotonic() - self._error_entered
			if elapsed >= ERROR_COOLDOWN_S:
				self.state = SystemState.IDLE
			self._emit(old_state)
			return self.state

		# --- OPEN / CANCEL always returns to IDLE ---
		if inputs.emg_intent in (EMGIntent.OPEN, EMGIntent.CANCEL):
			self.state = SystemState.IDLE
			self._emit(old_state)
			return self.state

		# --- IDLE / HOLDING + CLOSE → DETECTING ---
		if (
			self.state in (SystemState.IDLE, SystemState.HOLDING)
			and inputs.emg_intent == EMGIntent.CLOSE
		):
			self.state = SystemState.DETECTING
			self._emit(old_state)
			return self.state

		# --- DETECTING + detection ready → GRIPPING ---
		if self.state == SystemState.DETECTING and inputs.detection_ready:
			self.state = SystemState.GRIPPING
			self._emit(old_state)
			return self.state

		# --- GRIPPING + grip complete → HOLDING ---
		if self.state == SystemState.GRIPPING and inputs.grip_complete:
			self.state = SystemState.HOLDING
			self._emit(old_state)
			return self.state

		self._emit(old_state)
		return self.state

	# ------------------------------------------------------------------
	# Internal
	# ------------------------------------------------------------------

	def _emit(self, old_state: SystemState) -> None:
		if old_state == self.state:
			return
		record = EventRecord(
			timestamp=time.time(),
			event_type="state_change",
			detail=f"{old_state.value} → {self.state.value}",
		)
		self._transition_history.append(record)
		# Keep bounded
		if len(self._transition_history) > 500:
			self._transition_history = self._transition_history[-250:]
		if self._on_transition:
			self._on_transition(record)
		self._prev_state = old_state
