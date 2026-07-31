"""Decision fusion and the safety layer.

The tests in :class:`TestTheAiNeverActsAlone` encode the project's central rule.
If any of them ever fails, the system has stopped being a shared-control device.
"""

from __future__ import annotations

import pytest

from neurogrip.ai.grasp.base import GraspPlan, PlanSource
from neurogrip.control.controller import HandState
from neurogrip.core.errors import Severity
from neurogrip.core.types import GraspType, HandPose, IntentKind, ModeId
from neurogrip.emg.intent import IntentEstimate
from neurogrip.emg.quality import SignalQuality
from neurogrip.fusion.evidence import Evidence, EvidenceSet
from neurogrip.fusion.fusion import DecisionAction, DecisionFusion, FusionInputs
from neurogrip.fusion.policy import POLICIES, policy_for_mode
from neurogrip.hal.system import BatteryState
from neurogrip.safety.estop import EmergencyStop, EstopSource
from neurogrip.safety.monitor import SafetyMonitor
from neurogrip.safety.rules import (
    BatteryRule,
    CommunicationRule,
    GripForceRule,
    SafetyContext,
    SensorFailureRule,
    ThermalRule,
)
from neurogrip.safety.watchdog import WatchdogGroup
from neurogrip.vision.types import BoundingBox, Detection, VisionCapability, VisionResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _StubPlanner:
    """A planner that always produces the same plan, so fusion is isolated."""

    name = "stub"

    def __init__(self, plan: GraspPlan | None = None) -> None:
        self.plan_result = plan or GraspPlan(
            grasp=GraspType.CYLINDRICAL,
            target=HandPose.uniform(0.8),
            force=0.6,
            speed=1.0,
            confidence=0.9,
            source=PlanSource.LEARNED,
            label="bottle",
            reasons=("stub",),
        )
        self.calls = 0

    def plan(self, context):
        self.calls += 1
        return self.plan_result


def _intent(kind=IntentKind.CLOSE, confidence=0.9, strength=0.7, timestamp=1.0, **kwargs):
    return IntentEstimate(
        kind=kind,
        confidence=confidence,
        strength=strength,
        timestamp=timestamp,
        quality=kwargs.pop("quality", SignalQuality.GOOD),
        **kwargs,
    )


def _vision(label="bottle", confidence=0.9, timestamp=1.0, age=8):
    return VisionResult(
        timestamp=timestamp,
        detections=(
            Detection(
                label=label,
                confidence=confidence,
                bbox=BoundingBox(0.4, 0.3, 0.6, 0.7),
                track_id=1,
                age=age,
                attributes={"label_agreement": 1.0},
            ),
        ),
        backend="test",
        capabilities=VisionCapability.DETECTION,
    )


def _inputs(**kwargs) -> FusionInputs:
    defaults = dict(
        intent=_intent(),
        vision=_vision(),
        current_pose=HandPose.open_hand(),
        mode=ModeId.AI_ASSIST,
        timestamp=1.0,
    )
    defaults.update(kwargs)
    return FusionInputs(**defaults)


def _hand(**kwargs) -> HandState:
    defaults = dict(
        pose=HandPose.open_hand(),
        commanded=HandPose.open_hand(),
        enabled=True,
        comms_ok=True,
        temperature_c=30.0,
        timestamp=1.0,
    )
    defaults.update(kwargs)
    return HandState(**defaults)


# ---------------------------------------------------------------------------
# The central rule
# ---------------------------------------------------------------------------


class TestTheAiNeverActsAlone:
    """These tests are the executable form of the project's core requirement."""

    def test_no_intent_means_no_motion_however_confident_the_vision(self, clock):
        fusion = DecisionFusion(_StubPlanner(), clock)
        decision = fusion.evaluate(_inputs(intent=None, vision=_vision(confidence=0.99)))
        assert decision.action is DecisionAction.IDLE
        assert not decision.commands_motion

    def test_resting_intent_means_no_motion(self, clock):
        fusion = DecisionFusion(_StubPlanner(), clock)
        decision = fusion.evaluate(_inputs(intent=_intent(kind=IntentKind.REST)))
        assert not decision.commands_motion

    def test_stale_intent_means_no_motion(self, clock):
        fusion = DecisionFusion(_StubPlanner(), clock)
        # Intent from a second ago: the user may have stopped asking.
        decision = fusion.evaluate(_inputs(intent=_intent(timestamp=0.0), timestamp=1.0))
        assert decision.action is DecisionAction.IDLE
        assert "stale" in decision.reasons[0]

    def test_the_planner_is_not_even_consulted_without_intent(self, clock):
        planner = _StubPlanner()
        fusion = DecisionFusion(planner, clock)
        fusion.evaluate(_inputs(intent=None))
        assert planner.calls == 0

    def test_low_confidence_intent_does_not_move_the_hand(self, clock):
        fusion = DecisionFusion(_StubPlanner(), clock)
        decision = fusion.evaluate(_inputs(intent=_intent(confidence=0.1)))
        assert not decision.commands_motion

    def test_cancel_beats_everything_including_a_confident_plan(self, clock):
        fusion = DecisionFusion(_StubPlanner(), clock)
        decision = fusion.evaluate(_inputs(intent=_intent(kind=IntentKind.CANCEL)))
        assert decision.action is DecisionAction.CANCEL

    def test_safety_block_beats_a_confident_user_and_a_confident_ai(self, clock):
        fusion = DecisionFusion(_StubPlanner(), clock)
        decision = fusion.evaluate(_inputs(motion_allowed=False, safety_reason="overheated"))
        assert decision.action is DecisionAction.BLOCKED
        assert "overheated" in decision.reasons[0]


class TestAssistanceDegradesToControlNeverToInaction:
    """The other half of the rule: failing assistance must never disable the hand."""

    def test_no_vision_still_moves_the_hand(self, clock):
        """Without vision the planner may still offer a generic grip, but what
        must never happen is nothing at all."""
        fusion = DecisionFusion(_StubPlanner(), clock)
        decision = fusion.evaluate(_inputs(vision=None))
        assert decision.commands_motion
        assert decision.target is not None

    def test_no_vision_and_no_plan_falls_through_to_direct_control(self, clock):
        class NoPlan:
            name = "none"

            def plan(self, context):
                return None

        fusion = DecisionFusion(NoPlan(), clock)
        decision = fusion.evaluate(_inputs(vision=None))
        assert decision.action is DecisionAction.DIRECT
        assert decision.commands_motion

    def test_stale_vision_still_produces_direct_control(self, clock):
        fusion = DecisionFusion(_StubPlanner(), clock)
        decision = fusion.evaluate(_inputs(vision=_vision(timestamp=0.0), timestamp=1.0))
        assert decision.commands_motion

    def test_ai_suspended_by_safety_still_produces_direct_control(self, clock):
        fusion = DecisionFusion(_StubPlanner(), clock)
        decision = fusion.evaluate(_inputs(ai_allowed=False, safety_reason="camera failed"))
        assert decision.action is DecisionAction.DIRECT
        assert decision.commands_motion

    def test_a_planner_that_raises_degrades_to_direct_control(self, clock):
        class Broken:
            name = "broken"

            def plan(self, context):
                raise RuntimeError("model exploded")

        fusion = DecisionFusion(Broken(), clock)
        decision = fusion.evaluate(_inputs())
        assert decision.action is DecisionAction.DIRECT
        assert decision.commands_motion

    def test_direct_control_is_proportional_to_effort(self, clock):
        class NoPlan:
            name = "none"

            def plan(self, context):
                return None

        fusion = DecisionFusion(NoPlan(), clock)
        weak = fusion.evaluate(_inputs(vision=None, intent=_intent(strength=0.3)))
        strong = fusion.evaluate(_inputs(vision=None, intent=_intent(strength=0.9)))
        assert max(strong.target) > max(weak.target)


class TestFusionBehaviour:
    def test_assists_when_intent_and_vision_agree(self, clock):
        fusion = DecisionFusion(_StubPlanner(), clock)
        decision = fusion.evaluate(_inputs())
        assert decision.action is DecisionAction.ASSISTED
        assert decision.plan is not None
        assert decision.plan.grasp is GraspType.CYLINDRICAL

    def test_opening_never_needs_the_ai(self, clock):
        planner = _StubPlanner()
        fusion = DecisionFusion(planner, clock)
        decision = fusion.evaluate(_inputs(intent=_intent(kind=IntentKind.OPEN)))
        assert decision.action is DecisionAction.RELEASE
        assert decision.target == HandPose.open_hand()
        assert planner.calls == 0

    def test_manual_mode_never_assists(self, clock):
        fusion = DecisionFusion(_StubPlanner(), clock)
        fusion.set_policy(POLICIES[ModeId.MANUAL])
        decision = fusion.evaluate(_inputs(mode=ModeId.MANUAL))
        assert decision.action is DecisionAction.DIRECT
        assert decision.plan is None

    def test_holding_at_rest_maintains_the_grip(self, clock):
        fusion = DecisionFusion(_StubPlanner(), clock)
        decision = fusion.evaluate(_inputs(intent=_intent(kind=IntentKind.REST), holding=True))
        assert decision.action is DecisionAction.IDLE
        assert "holding" in decision.reasons[0]

    def test_force_is_capped_by_the_safety_ceiling(self, clock):
        fusion = DecisionFusion(_StubPlanner(), clock)
        decision = fusion.evaluate(_inputs(safety_force_ceiling=0.2))
        assert decision.force <= 0.2

    def test_the_plan_is_held_briefly_rather_than_recomputed_every_cycle(self, clock):
        planner = _StubPlanner()
        fusion = DecisionFusion(planner, clock)
        for step in range(5):
            fusion.evaluate(_inputs(timestamp=1.0 + step * 0.01))
        assert planner.calls == 1

    def test_evidence_is_recorded_even_when_the_decision_is_idle(self, clock):
        fusion = DecisionFusion(_StubPlanner(), clock)
        decision = fusion.evaluate(_inputs(intent=_intent(kind=IntentKind.REST)))
        assert decision.evidence is not None
        assert "emg" in decision.evidence.sources


class TestFusionPolicy:
    def test_manual_and_training_disable_the_ai(self):
        assert not POLICIES[ModeId.MANUAL].ai_enabled
        assert not POLICIES[ModeId.TRAINING].ai_enabled

    def test_sports_is_faster_and_reacts_sooner(self):
        sports, assist = POLICIES[ModeId.SPORTS], POLICIES[ModeId.AI_ASSIST]
        assert sports.speed_ceiling > assist.speed_ceiling
        assert sports.max_intent_age_s < assist.max_intent_age_s
        assert sports.plan_hold_s < assist.plan_hold_s

    def test_missing_vision_does_not_penalise_a_confident_user(self):
        policy = POLICIES[ModeId.AI_ASSIST]
        assert policy.combined_confidence(0.9, 0.0) == pytest.approx(0.9)

    def test_configuration_cannot_enable_the_ai_in_manual_mode(self, clock):
        from neurogrip.core.config import ConfigLoader

        config = ConfigLoader().add_mapping({"fusion": {"manual": {"ai_enabled": True}}}).build()
        policy = policy_for_mode(ModeId.MANUAL, config)
        assert not policy.ai_enabled

    def test_configuration_cannot_raise_the_force_ceiling(self, clock):
        from neurogrip.core.config import ConfigLoader

        config = ConfigLoader().add_mapping({"fusion": {"ai_assist": {"force_ceiling": 5.0}}}).build()
        policy = policy_for_mode(ModeId.AI_ASSIST, config)
        assert policy.force_ceiling <= POLICIES[ModeId.AI_ASSIST].force_ceiling


class TestEvidence:
    def test_confidence_decays_with_age(self):
        evidence = Evidence(source="vision", label="bottle", confidence=1.0, timestamp=0.0)
        assert evidence.decayed_confidence(0.0, half_life=0.4) == pytest.approx(1.0)
        assert evidence.decayed_confidence(0.4, half_life=0.4) == pytest.approx(0.5)

    def test_weighted_score_ignores_informational_items(self):
        evidence = EvidenceSet(
            items=(
                Evidence("emg", "close", 1.0, weight=1.0, timestamp=0.0),
                Evidence("depth", "30 cm", 0.0, weight=0.0, timestamp=0.0),
            ),
            evaluated_at=0.0,
        )
        assert evidence.weighted_score == pytest.approx(1.0)


class TestSafetyRules:
    def test_grip_force_over_the_limit_is_critical(self):
        from neurogrip.control.force import GripState

        rule = GripForceRule(max_force_n=40.0)
        grip = GripState(holding=True, contacts=(), force=0.9, estimated_force_n=55.0)
        fault = rule.evaluate(SafetyContext(timestamp=1.0, hand=_hand(grip=grip)))
        assert fault is not None and fault.severity is Severity.CRITICAL

    def test_thermal_derates_before_it_stops(self):
        rule = ThermalRule(motor_warn_c=50.0, motor_limit_c=70.0)
        warn = rule.evaluate(SafetyContext(timestamp=1.0, hand=_hand(temperature_c=60.0)))
        assert warn is not None
        assert warn.severity is Severity.DEGRADED
        assert warn.force_ceiling < 1.0

        stop = rule.evaluate(SafetyContext(timestamp=1.0, hand=_hand(temperature_c=75.0)))
        assert stop is not None and stop.severity is Severity.CRITICAL

    def test_lost_communication_is_critical(self):
        fault = CommunicationRule().evaluate(
            SafetyContext(timestamp=1.0, hand=_hand(comms_ok=False))
        )
        assert fault is not None and fault.severity is Severity.CRITICAL
        assert fault.force_ceiling == 0.0

    def test_absent_emg_is_only_a_fault_once_the_watchdog_has_expired(self):
        rule = SensorFailureRule()
        # Start-up: no intent yet, but the watchdog has not fired.
        assert rule.evaluate(SafetyContext(timestamp=0.1, hand=_hand(), intent=None)) is None
        # Data has genuinely stopped.
        fault = rule.evaluate(
            SafetyContext(timestamp=5.0, hand=_hand(), intent=None, expired_watchdogs=("emg",))
        )
        assert fault is not None and fault.severity is Severity.FALLBACK

    def test_low_battery_degrades_before_it_stops(self):
        rule = BatteryRule()
        low = rule.evaluate(
            SafetyContext(timestamp=1.0, hand=_hand(), battery=BatteryState(percentage=10.0))
        )
        assert low is not None and low.severity is Severity.DEGRADED
        critical = rule.evaluate(
            SafetyContext(timestamp=1.0, hand=_hand(), battery=BatteryState(percentage=3.0))
        )
        assert critical is not None and critical.severity is Severity.CRITICAL

    def test_charging_battery_raises_no_fault(self):
        fault = BatteryRule().evaluate(
            SafetyContext(
                timestamp=1.0, hand=_hand(), battery=BatteryState(percentage=3.0, charging=True)
            )
        )
        assert fault is None


class TestEmergencyStop:
    def test_engage_is_idempotent_and_accumulates_reasons(self, clock):
        estop = EmergencyStop(clock)
        estop.engage(EstopSource.USER_UI, "button")
        estop.engage(EstopSource.THERMAL, "too hot")
        assert estop.engaged
        assert estop.engage_count == 1
        assert set(estop.reasons) == {"button", "too hot"}

    def test_release_requires_an_explicit_source(self, clock):
        estop = EmergencyStop(clock)
        estop.engage(EstopSource.USER_UI, "button")
        assert estop.release("user:ui") is not None
        assert not estop.engaged

    def test_release_when_clear_is_a_no_op(self, clock):
        assert EmergencyStop(clock).release("user:ui") is None

    def test_a_broken_listener_cannot_prevent_the_stop(self, clock):
        estop = EmergencyStop(clock)
        estop.add_listener(lambda _record: 1 / 0)
        estop.engage(EstopSource.USER_UI, "button")  # must not raise
        assert estop.engaged


class TestWatchdogs:
    def test_expiry_reported_once_then_recovery(self, clock, bus):
        group = WatchdogGroup(clock)
        group.add("test", 0.1)
        group.kick("test")
        clock.advance(0.2)
        assert len(group.check_all()) == 1
        assert len(group.check_all()) == 0  # not reported twice
        group.kick("test")
        assert not group.expired

    def test_disabled_watchdogs_never_expire(self, clock):
        group = WatchdogGroup(clock)
        group.add("vision", 0.1, enabled=False)
        group.kick("vision")
        clock.advance(5.0)
        assert not group.check_all()


class TestSafetyMonitor:
    def _monitor(self, clock, bus):
        group = WatchdogGroup(clock)
        group.add("control", 0.1, severity=Severity.CRITICAL)
        group.add("emg", 0.3, severity=Severity.FALLBACK)
        monitor = SafetyMonitor(clock, bus, EmergencyStop(clock), group)
        monitor.start()
        group.kick("control")
        group.kick("emg")
        return monitor, group

    def _evaluate(self, monitor, group, clock, hand, **kwargs):
        clock.advance(0.05)
        group.kick("control")
        group.kick("emg")
        return monitor.evaluate(
            SafetyContext(
                timestamp=clock.monotonic(),
                hand=hand,
                intent=_intent(),
                **kwargs,
            )
        )

    def test_nominal_state_permits_everything(self, clock, bus):
        monitor, group = self._monitor(clock, bus)
        state = self._evaluate(monitor, group, clock, _hand(), battery=BatteryState(percentage=80))
        assert state.motion_allowed
        assert state.force_ceiling == pytest.approx(1.0)

    def test_a_critical_fault_stops_motion_and_engages_the_estop(self, clock, bus):
        monitor, group = self._monitor(clock, bus)
        state = self._evaluate(monitor, group, clock, _hand(comms_ok=False))
        assert not state.motion_allowed
        assert state.estop_engaged
        assert monitor.estop.engaged

    def test_a_fallback_fault_keeps_motion_but_drops_the_ai(self, clock, bus):
        monitor, group = self._monitor(clock, bus)
        state = self._evaluate(
            monitor,
            group,
            clock,
            _hand(),
            battery=BatteryState(percentage=80),
            cpu_temperature_c=95.0,
        )
        assert state.motion_allowed
        assert not state.ai_allowed

    def test_acknowledge_is_refused_while_the_condition_persists(self, clock, bus):
        monitor, group = self._monitor(clock, bus)
        self._evaluate(monitor, group, clock, _hand(comms_ok=False))
        assert not monitor.acknowledge("user:test")

    def test_acknowledge_succeeds_once_the_condition_clears(self, clock, bus):
        monitor, group = self._monitor(clock, bus)
        self._evaluate(monitor, group, clock, _hand(comms_ok=False))
        self._evaluate(monitor, group, clock, _hand(), battery=BatteryState(percentage=80))
        clock.advance(2.0)
        self._evaluate(monitor, group, clock, _hand(), battery=BatteryState(percentage=80))
        assert monitor.acknowledge("user:test")
        assert not monitor.estop.engaged

    def test_a_critical_watchdog_expiry_engages_the_estop(self, clock, bus):
        monitor, _group = self._monitor(clock, bus)
        clock.advance(0.5)  # no kicks: the control loop has stalled
        state = monitor.evaluate(
            SafetyContext(timestamp=clock.monotonic(), hand=_hand(), intent=_intent())
        )
        assert monitor.estop.engaged
        assert not state.motion_allowed

    def test_a_rule_that_raises_becomes_a_fault_rather_than_a_crash(self, clock, bus):
        class Broken:
            name = "broken"
            enabled = True

            def evaluate(self, context):
                raise RuntimeError("bad rule")

        group = WatchdogGroup(clock)
        monitor = SafetyMonitor(clock, bus, EmergencyStop(clock), group, rules=(Broken(),))
        monitor.start()
        state = monitor.evaluate(SafetyContext(timestamp=1.0, hand=_hand()))
        assert any(f.code.startswith("rule_error") for f in state.faults)
