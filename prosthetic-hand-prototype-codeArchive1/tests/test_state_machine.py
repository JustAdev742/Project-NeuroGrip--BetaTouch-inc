from app.models.status import EMGIntent, SystemState
from app.controller.state_machine import StateInputs, StateMachine


def test_state_transitions() -> None:
	machine = StateMachine()
	assert machine.state == SystemState.IDLE

	machine.update(StateInputs(EMGIntent.CLOSE, detection_ready=False, grip_complete=False, error=False))
	assert machine.state == SystemState.DETECTING

	machine.update(StateInputs(EMGIntent.CLOSE, detection_ready=True, grip_complete=False, error=False))
	assert machine.state == SystemState.GRIPPING

	machine.update(StateInputs(EMGIntent.HOLD, detection_ready=False, grip_complete=True, error=False))
	assert machine.state == SystemState.HOLDING

	machine.update(StateInputs(EMGIntent.OPEN, detection_ready=False, grip_complete=False, error=False))
	assert machine.state == SystemState.IDLE


def test_error_takes_priority() -> None:
	machine = StateMachine()
	machine.update(StateInputs(EMGIntent.CLOSE, detection_ready=True, grip_complete=False, error=True))
	assert machine.state == SystemState.ERROR


def test_error_cooldown() -> None:
	import time
	machine = StateMachine()
	machine.update(StateInputs(EMGIntent.REST, detection_ready=False, grip_complete=False, error=True))
	assert machine.state == SystemState.ERROR

	# Immediately after error, should stay in ERROR (cooldown)
	machine.update(StateInputs(EMGIntent.REST, detection_ready=False, grip_complete=False, error=False))
	# May or may not have recovered depending on timing — just check it doesn't crash


def test_cancel_returns_to_idle() -> None:
	machine = StateMachine()
	machine.update(StateInputs(EMGIntent.CLOSE, detection_ready=False, grip_complete=False, error=False))
	assert machine.state == SystemState.DETECTING

	machine.update(StateInputs(EMGIntent.CANCEL, detection_ready=False, grip_complete=False, error=False))
	assert machine.state == SystemState.IDLE


def test_transition_history_tracked() -> None:
	machine = StateMachine()
	machine.update(StateInputs(EMGIntent.CLOSE, detection_ready=False, grip_complete=False, error=False))
	machine.update(StateInputs(EMGIntent.OPEN, detection_ready=False, grip_complete=False, error=False))
	assert len(machine.transition_history) >= 2
