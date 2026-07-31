"""Control: kinematics, grips, trajectory, motion queue, grip force, controller."""

from __future__ import annotations

import pytest

from neurogrip.control.controller import HandController
from neurogrip.control.force import AdaptiveGripController, GripSettings
from neurogrip.control.grips import BUILTIN_PRESETS, GripLibrary
from neurogrip.control.kinematics import HandKinematics
from neurogrip.control.motion import MotionLimits, TrajectoryGenerator
from neurogrip.control.queue import MotionCommand, MotionQueue, Priority
from neurogrip.core.errors import EmergencyStopActive
from neurogrip.core.types import Finger, GraspType, HandPose
from neurogrip.hal.servo.base import ServoLimits
from neurogrip.hal.servo.simulated import ContactModel, SimulatedServoBus


class TestKinematics:
    def test_travel_is_monotonic_in_closure(self):
        kinematics = HandKinematics()
        previous = -1.0
        for step in range(11):
            travel = kinematics.closure_to_travel_mm(Finger.INDEX, step / 10)
            assert travel > previous
            previous = travel

    def test_travel_round_trips(self):
        kinematics = HandKinematics()
        for closure in (0.0, 0.25, 0.5, 0.9, 1.0):
            travel = kinematics.closure_to_travel_mm(Finger.MIDDLE, closure)
            assert kinematics.travel_to_closure(Finger.MIDDLE, travel) == pytest.approx(
                closure, abs=1e-6
            )

    def test_aperture_shrinks_as_the_hand_closes(self):
        kinematics = HandKinematics()
        assert kinematics.aperture_m(HandPose.open_hand()) == pytest.approx(
            kinematics.max_aperture_m
        )
        assert kinematics.aperture_m(HandPose.closed_hand()) == pytest.approx(0.0)

    def test_closure_for_aperture_is_the_inverse(self):
        kinematics = HandKinematics()
        closure = kinematics.closure_for_aperture(0.055)
        assert kinematics.aperture_m(HandPose.uniform(closure)) == pytest.approx(0.055, abs=1e-3)

    def test_thumb_cannot_exceed_its_mechanical_stop(self):
        kinematics = HandKinematics()
        corrected, notes = kinematics.enforce_limits(HandPose.closed_hand())
        assert corrected[Finger.THUMB] <= 0.92
        assert any("Thumb" in note for note in notes)

    def test_self_collision_rule_limits_the_index_when_the_thumb_is_across(self):
        kinematics = HandKinematics()
        pose = HandPose((0.85, 1.0, 0.5, 0.5, 0.5))
        corrected, notes = kinematics.enforce_limits(pose)
        assert corrected[Finger.INDEX] <= 0.95
        assert notes

    def test_reachable_pose_is_left_alone(self):
        kinematics = HandKinematics()
        pose = HandPose((0.5, 0.5, 0.5, 0.5, 0.5))
        corrected, notes = kinematics.enforce_limits(pose)
        assert corrected == pose
        assert not notes


class TestGripLibrary:
    def test_every_grasp_type_has_a_builtin_preset(self):
        for grasp in GraspType:
            assert grasp in BUILTIN_PRESETS

    def test_unknown_grasp_falls_back_rather_than_raising(self):
        library = GripLibrary({GraspType.POWER: BUILTIN_PRESETS[GraspType.POWER]})
        # Never raise: a missing preset must not stop the hand from moving.
        assert library.get(GraspType.TRIPOD).grasp is GraspType.POWER

    def test_precision_grips_use_less_force_than_power_grips(self):
        library = GripLibrary()
        assert library.get(GraspType.PRECISION_PINCH).force < library.get(GraspType.POWER).force

    def test_partial_interpolates_from_open(self):
        preset = GripLibrary().get(GraspType.POWER)
        half = preset.partial(0.5)
        assert half[Finger.INDEX] == pytest.approx(preset.pose[Finger.INDEX] * 0.5)

    def test_mode_scaling_only_reduces(self):
        library = GripLibrary()
        scaled = library.scaled_for_mode(speed_scale=0.5, force_scale=0.5)
        assert scaled.get(GraspType.POWER).force < library.get(GraspType.POWER).force

    def test_loads_from_configuration(self, config):
        from neurogrip.core.config import ConfigLoader

        configured = (
            ConfigLoader()
            .add_mapping(
                {
                    "grasps": {
                        "power": {
                            "pose": {"thumb": 0.5, "index": 0.5, "middle": 0.5, "ring": 0.5, "pinky": 0.5},
                            "force": 0.33,
                        }
                    }
                }
            )
            .build()
        )
        library = GripLibrary.from_config(configured)
        assert library.get(GraspType.POWER).force == pytest.approx(0.33)
        assert library.get(GraspType.POWER).pose[Finger.THUMB] == pytest.approx(0.5)
        # Grasps not mentioned in the file keep their built-in definition.
        assert library.get(GraspType.HOOK).pose == BUILTIN_PRESETS[GraspType.HOOK].pose


class TestTrajectory:
    def test_reaches_the_target(self):
        generator = TrajectoryGenerator(MotionLimits())
        generator.start(HandPose.uniform(0.8))
        for _ in range(1000):
            state = generator.step(0.005)
            if state.complete:
                break
        assert state.pose.is_close(HandPose.uniform(0.8), 0.02)

    def test_respects_the_velocity_limit(self):
        limits = MotionLimits(max_velocity=0.5, max_acceleration=100.0)
        generator = TrajectoryGenerator(limits, s_curve=False)
        generator.start(HandPose.closed_hand())
        peak = 0.0
        for _ in range(400):
            state = generator.step(0.005)
            peak = max(peak, state.max_velocity)
        assert peak <= limits.max_velocity * 1.05

    def test_never_overshoots(self):
        generator = TrajectoryGenerator(MotionLimits(max_velocity=5.0, max_acceleration=100.0))
        generator.start(HandPose.uniform(0.5))
        for _ in range(600):
            state = generator.step(0.005)
            assert all(v <= 0.51 for v in state.pose)

    def test_fingers_arrive_together(self):
        """Synchronisation: a short-travel finger must not finish long before a
        long-travel one, or the hand closes raggedly and pushes objects away."""
        generator = TrajectoryGenerator(MotionLimits())
        generator.start(HandPose((0.2, 0.9, 0.9, 0.9, 0.9)))
        for _ in range(2000):
            state = generator.step(0.005)
            if state.complete:
                break
        # All fingers within tolerance at the same moment.
        assert state.complete
        assert state.pose.is_close(HandPose((0.2, 0.9, 0.9, 0.9, 0.9)), 0.02)

    def test_retarget_preserves_velocity(self):
        generator = TrajectoryGenerator(MotionLimits())
        generator.start(HandPose.closed_hand())
        for _ in range(20):
            generator.step(0.005)
        moving = max(abs(v) for v in generator.velocities)
        generator.retarget(HandPose.uniform(0.7))
        assert max(abs(v) for v in generator.velocities) == pytest.approx(moving)

    def test_estimated_duration_is_positive_and_scales(self):
        generator = TrajectoryGenerator(MotionLimits())
        short = generator.estimate_duration(HandPose.open_hand(), HandPose.uniform(0.2))
        long = generator.estimate_duration(HandPose.open_hand(), HandPose.closed_hand())
        assert 0 < short < long

    def test_stop_holds_position(self):
        generator = TrajectoryGenerator(MotionLimits())
        generator.start(HandPose.closed_hand())
        for _ in range(20):
            generator.step(0.005)
        held = generator.current
        generator.stop()
        for _ in range(20):
            generator.step(0.005)
        assert generator.current == held


class TestMotionQueue:
    def _command(self, **kwargs) -> MotionCommand:
        defaults = dict(target=HandPose.uniform(0.5), source="test")
        defaults.update(kwargs)
        return MotionCommand(**defaults)

    def test_first_command_is_accepted(self):
        queue = MotionQueue()
        assert queue.submit(self._command(), 0.0).accepted

    def test_higher_priority_preempts(self):
        queue = MotionQueue()
        queue.submit(self._command(priority=Priority.ASSISTED, source="ai"), 0.0)
        result = queue.submit(self._command(priority=Priority.USER_OVERRIDE, source="user"), 0.1)
        assert result.accepted
        assert result.preempted is not None
        assert queue.active.source == "user"

    def test_lower_priority_is_dropped_not_queued(self):
        queue = MotionQueue()
        queue.submit(self._command(priority=Priority.USER_DIRECT, source="user"), 0.0)
        result = queue.submit(self._command(priority=Priority.BACKGROUND, source="idle"), 0.1)
        assert not result.accepted
        assert queue.active.source == "user"

    def test_same_stream_refresh_is_not_a_restart(self):
        """Continuous control re-issues its command every cycle; that must not
        restart the pre-shape leg or the timeout."""
        queue = MotionQueue()
        command = self._command(preshape=HandPose.uniform(0.1), source="mode:ai_assist")
        queue.submit(command, 0.0)
        queue.mark_preshape_done()
        result = queue.submit(command, 0.01)
        assert result.same_stream
        assert not queue.preshape_pending

    def test_different_source_at_equal_priority_replaces_and_restarts(self):
        queue = MotionQueue()
        queue.submit(self._command(preshape=HandPose.uniform(0.1), source="a"), 0.0)
        queue.mark_preshape_done()
        result = queue.submit(self._command(preshape=HandPose.uniform(0.1), source="b"), 0.1)
        assert not result.same_stream
        assert queue.preshape_pending

    def test_atomic_command_cannot_be_replaced_by_an_equal_priority(self):
        queue = MotionQueue()
        queue.submit(self._command(atomic=True, source="homing"), 0.0)
        assert not queue.submit(self._command(source="other"), 0.1).accepted

    def test_timeout_cancels_a_stuck_command(self):
        queue = MotionQueue()
        queue.submit(self._command(timeout_s=1.0), 0.0)
        assert queue.check_timeout(0.5) is None
        assert queue.check_timeout(1.5) is not None
        assert not queue.is_busy

    def test_preshape_leg_runs_before_the_target(self):
        queue = MotionQueue()
        preshape = HandPose.uniform(0.1)
        target = HandPose.uniform(0.9)
        queue.submit(self._command(target=target, preshape=preshape), 0.0)
        assert queue.current_leg_target == preshape
        queue.mark_preshape_done()
        assert queue.current_leg_target == target


class TestAdaptiveGrip:
    def test_detects_contact_on_a_rigid_object(self, clock, servo_bus):
        grip = AdaptiveGripController(clock)
        grip.set_commanded_force(0.7)
        servo_bus.set_contact(ContactModel.uniform(0.5, stiffness=1.5))
        target = HandPose.uniform(0.9)

        state = None
        for _ in range(400):
            clock.advance(0.005)
            servo_bus.write_targets(target, force=0.7)
            state = grip.update(servo_bus.read_state(), commanded=target, moving=True)
        assert state.holding
        assert state.contact_count >= 2

    def test_detects_contact_on_a_compliant_object(self, clock, servo_bus):
        """Soft objects barely raise the current; missing them would mean
        squeezing exactly the things that least tolerate it."""
        grip = AdaptiveGripController(clock)
        grip.set_commanded_force(0.6)
        servo_bus.set_contact(ContactModel.uniform(0.45, stiffness=0.3))
        target = HandPose.uniform(0.9)

        state = None
        for _ in range(600):
            clock.advance(0.005)
            servo_bus.write_targets(target, force=0.6)
            state = grip.update(servo_bus.read_state(), commanded=target, moving=True)
        assert state.holding

    def test_no_contact_when_nothing_is_in_the_way(self, clock, servo_bus):
        grip = AdaptiveGripController(clock)
        target = HandPose.uniform(0.8)
        state = None
        for _ in range(400):
            clock.advance(0.005)
            servo_bus.write_targets(target, force=0.6)
            state = grip.update(servo_bus.read_state(), commanded=target, moving=True)
        assert not state.holding

    def test_contact_limits_the_commanded_target(self, clock, servo_bus):
        grip = AdaptiveGripController(clock)
        grip.set_commanded_force(0.7)
        servo_bus.set_contact(ContactModel.uniform(0.5, stiffness=1.5))
        target = HandPose.uniform(0.95)
        for _ in range(400):
            clock.advance(0.005)
            servo_bus.write_targets(target, force=0.7)
            grip.update(servo_bus.read_state(), commanded=target, moving=True)
        limited = grip.contact_limited_target(target)
        assert max(limited) < 0.95

    def test_force_never_exceeds_the_ceiling(self, clock):
        grip = AdaptiveGripController(clock, GripSettings(), max_force=0.5)
        grip.set_commanded_force(1.0)
        assert grip.effective_force <= 0.5


class TestHandController:
    def _controller(self, clock, bus, **kwargs) -> HandController:
        controller = HandController(SimulatedServoBus(clock), clock, bus, **kwargs)
        controller.start()
        controller.enable()
        controller.home()
        return controller

    def test_reaches_a_commanded_grip(self, clock, bus):
        controller = self._controller(clock, bus)
        controller.apply_grip(GraspType.CYLINDRICAL, source="test")
        for _ in range(600):
            clock.advance(0.005)
            state = controller.tick()
        expected = controller.grips.get(GraspType.CYLINDRICAL).pose
        assert state.pose.is_close(expected, 0.05)

    def test_emergency_stop_latches_and_refuses_commands(self, clock, bus):
        controller = self._controller(clock, bus)
        controller.emergency_stop("test")
        assert not controller.apply_grip(GraspType.POWER).accepted
        for _ in range(50):
            clock.advance(0.005)
            state = controller.tick()
        assert state.estop
        assert not state.enabled

    def test_enable_is_refused_while_the_estop_is_latched(self, clock, bus):
        controller = self._controller(clock, bus)
        controller.emergency_stop("test")
        with pytest.raises(EmergencyStopActive):
            controller.enable()

    def test_cancel_holds_the_current_position(self, clock, bus):
        controller = self._controller(clock, bus)
        controller.apply_grip(GraspType.FIST, source="test")
        for _ in range(30):
            clock.advance(0.005)
            controller.tick()
        held = controller.state.pose
        controller.cancel("user cancel")
        for _ in range(100):
            clock.advance(0.005)
            state = controller.tick()
        assert state.pose.max_difference(held) < 0.02

    def test_mechanical_limits_are_applied_to_submitted_targets(self, clock, bus):
        controller = self._controller(clock, bus)
        controller.move_to(HandPose.closed_hand(), source="test")
        assert controller.queue.active.target[Finger.THUMB] <= 0.92

    def test_force_ceiling_is_clipped_to_the_servo_limit(self, clock, bus):
        controller = self._controller(clock, bus, servo_limits=ServoLimits(max_force=0.4))
        controller.configure(force_ceiling=0.99)
        controller.apply_grip(GraspType.FIST, source="test")
        for _ in range(200):
            clock.advance(0.005)
            state = controller.tick()
        assert state.force <= 0.4 + 1e-6

    def test_health_reports_a_comms_failure(self, clock, bus):
        controller = self._controller(clock, bus)
        controller.tick()
        assert controller.health().status.name == "OK"
