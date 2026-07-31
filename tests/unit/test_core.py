"""Core primitives: types, clock, events, state machine, config, ring buffers."""

from __future__ import annotations

import pytest

from neurogrip.core.clock import Deadline, SimulatedClock, Stopwatch
from neurogrip.core.config import Config, ConfigLoader, deep_merge
from neurogrip.core.errors import ConfigurationError
from neurogrip.core.events import EventBus
from neurogrip.core.rate import RateTimer
from neurogrip.core.ringbuffer import RingBuffer, RunningStats, percentile
from neurogrip.core.state import StateMachine, SystemState, build_system_state_machine
from neurogrip.core.types import Finger, GraspType, HandPose, ModeId, clamp, normalise


class TestHandPose:
    def test_clamps_out_of_range_values(self):
        pose = HandPose((-0.5, 1.5, 0.3, 0.0, 1.0))
        assert pose.values == (0.0, 1.0, 0.3, 0.0, 1.0)

    def test_rejects_wrong_length(self):
        with pytest.raises(ValueError):
            HandPose((0.0, 0.0))

    def test_indexing_by_finger_and_int(self):
        pose = HandPose((0.1, 0.2, 0.3, 0.4, 0.5))
        assert pose[Finger.THUMB] == pytest.approx(0.1)
        assert pose[4] == pytest.approx(0.5)

    def test_blend_interpolates(self):
        blended = HandPose.open_hand().blend(HandPose.closed_hand(), 0.25)
        assert all(v == pytest.approx(0.25) for v in blended)

    def test_masked_takes_selected_fingers_from_other(self):
        result = HandPose.open_hand().masked([Finger.INDEX], HandPose.closed_hand())
        assert result[Finger.INDEX] == 1.0
        assert result[Finger.THUMB] == 0.0

    def test_aperture_is_one_when_open(self):
        assert HandPose.open_hand().aperture == pytest.approx(1.0)
        assert HandPose.closed_hand().aperture == pytest.approx(0.0)

    def test_round_trips_through_mapping(self):
        pose = HandPose((0.1, 0.2, 0.3, 0.4, 0.5))
        assert HandPose.from_mapping(pose.as_dict()) == pose

    def test_is_hashable_and_comparable(self):
        # Frozen dataclasses are used as dict keys and compared for equality in
        # the motion queue's same-command detection.
        assert HandPose.uniform(0.5) == HandPose.uniform(0.5)
        assert len({HandPose.uniform(0.5), HandPose.uniform(0.5)}) == 1


class TestScalarHelpers:
    def test_clamp_bounds(self):
        assert clamp(-1.0) == 0.0
        assert clamp(2.0) == 1.0
        assert clamp(5.0, 0.0, 10.0) == 5.0

    def test_normalise_handles_degenerate_range(self):
        assert normalise(5.0, 3.0, 3.0) == 0.0
        assert normalise(5.0, 0.0, 10.0) == pytest.approx(0.5)


class TestEnums:
    def test_only_assistive_modes_enable_ai(self):
        assert ModeId.AI_ASSIST.ai_enabled
        assert ModeId.SPORTS.ai_enabled
        assert not ModeId.MANUAL.ai_enabled
        assert not ModeId.TRAINING.ai_enabled

    def test_precision_grasps_are_flagged(self):
        assert GraspType.PRECISION_PINCH.is_precision
        assert not GraspType.POWER.is_precision


class TestSimulatedClock:
    def test_sleep_advances_virtual_time(self, clock: SimulatedClock):
        start = clock.monotonic()
        clock.sleep(1.5)
        assert clock.monotonic() == pytest.approx(start + 1.5)

    def test_cannot_move_backwards(self, clock: SimulatedClock):
        clock.advance(1.0)
        with pytest.raises(ValueError):
            clock.set(0.5)

    def test_deadline_expiry(self, clock: SimulatedClock):
        deadline = Deadline(clock, 0.5)
        assert not deadline.expired
        clock.advance(0.6)
        assert deadline.expired
        assert deadline.remaining == 0.0

    def test_stopwatch_lap(self, clock: SimulatedClock):
        watch = Stopwatch(clock)
        clock.advance(2.0)
        assert watch.lap() == pytest.approx(2.0)
        assert watch.elapsed == pytest.approx(0.0)


class TestEventBus:
    def test_delivers_to_exact_subscribers(self, bus: EventBus):
        received = []
        bus.subscribe("a.b", received.append)
        bus.publish("a.b", 1)
        bus.publish("a.c", 2)
        assert [e.payload for e in received] == [1]

    def test_wildcard_and_global_subscriptions(self, bus: EventBus):
        prefix, everything = [], []
        bus.subscribe("a.*", prefix.append)
        bus.subscribe("*", everything.append)
        bus.publish("a.b", 1)
        bus.publish("z", 2)
        assert len(prefix) == 1
        assert len(everything) == 2

    def test_handler_exception_does_not_reach_publisher(self, bus: EventBus):
        def explode(_event):
            raise RuntimeError("boom")

        received = []
        bus.subscribe("t", explode)
        bus.subscribe("t", received.append)
        bus.publish("t", 1)  # must not raise
        assert len(received) == 1

    def test_repeatedly_failing_handler_is_quarantined(self, bus: EventBus):
        def explode(_event):
            raise RuntimeError("boom")

        bus.subscribe("t", explode)
        for _ in range(12):
            bus.publish("t")
        assert bus.stats["quarantined"] == 1

    def test_cancel_stops_delivery(self, bus: EventBus):
        received = []
        subscription = bus.subscribe("t", received.append)
        bus.publish("t")
        subscription.cancel()
        bus.publish("t")
        assert len(received) == 1

    def test_history_is_retained(self, bus: EventBus):
        for i in range(5):
            bus.publish("t", i)
        assert [e.payload for e in bus.history("t")] == [0, 1, 2, 3, 4]


class TestStateMachine:
    def test_undeclared_transition_is_refused(self, clock):
        machine: StateMachine[str] = StateMachine("a", clock)
        machine.allow("a", "b")
        assert machine.transition_to("b")
        assert not machine.transition_to("a")

    def test_guard_can_veto(self, clock):
        machine: StateMachine[str] = StateMachine("a", clock)
        allowed = False
        machine.allow("a", "b", guard=lambda *_: allowed)
        assert not machine.transition_to("b")
        allowed = True
        assert machine.transition_to("b")

    def test_force_bypasses_declarations(self, clock):
        machine: StateMachine[str] = StateMachine("a", clock)
        assert machine.transition_to("z", force=True)
        assert machine.state == "z"

    def test_hooks_run_in_order(self, clock):
        order = []
        machine: StateMachine[str] = StateMachine("a", clock)
        machine.allow("a", "b")
        machine.on_exit("a", lambda *_: order.append("exit"))
        machine.on_enter("b", lambda *_: order.append("enter"))
        machine.on_change(lambda _change: order.append("observe"))
        machine.transition_to("b")
        assert order == ["exit", "enter", "observe"]

    def test_system_machine_reaches_ready(self, clock):
        machine = build_system_state_machine(clock)
        assert machine.fire("boot_complete")
        assert machine.fire("selftest_passed")
        assert machine.fire("homed")
        assert machine.state is SystemState.READY

    def test_estop_reachable_from_everywhere_and_needs_reset(self, clock):
        machine = build_system_state_machine(clock)
        machine.fire("boot_complete")
        assert machine.fire("estop")
        assert machine.state is SystemState.ESTOP
        assert not machine.state.motion_allowed
        # Recovery must go via self-test, not straight back to ACTIVE.
        assert not machine.transition_to(SystemState.ACTIVE)
        assert machine.fire("reset")
        assert machine.state is SystemState.SELFTEST


class TestConfig:
    def test_deep_merge_replaces_lists_and_merges_tables(self):
        merged = deep_merge({"a": {"x": 1, "y": 2}, "l": [1, 2]}, {"a": {"y": 3}, "l": [9]})
        assert merged == {"a": {"x": 1, "y": 3}, "l": [9]}

    def test_dotted_lookup_and_defaults(self):
        config = Config({"a": {"b": {"c": 42}}})
        assert config.get("a.b.c") == 42
        assert config.get("a.b.missing", "fallback") == "fallback"

    def test_require_raises_with_a_useful_message(self):
        config = Config({})
        with pytest.raises(ConfigurationError, match=r"servo\.port"):
            config.require("servo.port")

    def test_type_coercion_rejects_wrong_types(self):
        config = Config({"n": "text"})
        with pytest.raises(ConfigurationError, match="must be a number"):
            config.get_float("n", 1.0)

    def test_environment_overrides_are_parsed_as_toml_scalars(self):
        config = (
            ConfigLoader()
            .add_mapping({"servo": {"baud": 9600, "port": "/dev/x"}})
            .add_environment(
                {
                    "NEUROGRIP__SERVO__BAUD": "921600",
                    "NEUROGRIP__SERVO__PORT": "/dev/ttyUSB1",
                    "NEUROGRIP__UI__FULLSCREEN": "true",
                }
            )
            .build()
        )
        assert config.get_int("servo.baud") == 921600
        assert config.get_str("servo.port") == "/dev/ttyUSB1"
        assert config.get_bool("ui.fullscreen") is True

    def test_cli_overrides_win(self):
        config = (
            ConfigLoader()
            .add_mapping({"a": {"b": 1}})
            .add_overrides(["a.b=99"])
            .build()
        )
        assert config.get_int("a.b") == 99

    def test_sections_enumerates_subtables(self):
        config = Config({"grasps": {"power": {"force": 0.8}, "pinch": {"force": 0.3}}})
        sections = config.sections("grasps")
        assert set(sections) == {"power", "pinch"}
        assert sections["power"].get_float("force") == pytest.approx(0.8)


class TestRingBuffer:
    def test_evicts_oldest_when_full(self):
        buffer = RingBuffer(3)
        buffer.extend([1, 2, 3, 4])
        assert buffer.to_list() == [2, 3, 4]

    def test_mean_stays_correct_after_eviction(self):
        buffer = RingBuffer(3)
        buffer.extend([10, 20, 30, 40])
        assert buffer.mean() == pytest.approx(30.0)

    def test_downsample_preserves_peaks(self):
        buffer = RingBuffer(100)
        buffer.extend([0.0] * 50 + [1.0] + [0.0] * 49)
        # A mean-based downsample would hide the spike; max-of-bucket must not.
        assert max(buffer.downsample(10)) == pytest.approx(1.0)

    def test_running_stats_matches_batch(self):
        stats = RunningStats()
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        for value in values:
            stats.add(value)
        assert stats.mean == pytest.approx(3.0)
        assert stats.std == pytest.approx(1.5811, abs=1e-3)
        assert stats.minimum == 1.0
        assert stats.maximum == 5.0

    def test_percentile_interpolates(self):
        assert percentile([0, 10], 0.5) == pytest.approx(5.0)


class TestRateTimer:
    def test_due_fires_at_the_period(self, clock: SimulatedClock):
        timer = RateTimer(clock, 100.0)
        assert not timer.due()
        clock.advance(0.01)
        assert timer.due()

    def test_resynchronises_after_a_long_stall(self, clock: SimulatedClock):
        timer = RateTimer(clock, 100.0)
        clock.advance(5.0)  # a 500-period stall
        assert timer.due()
        assert timer.missed == 1
        # It must not then fire repeatedly to "catch up".
        assert not timer.due()
