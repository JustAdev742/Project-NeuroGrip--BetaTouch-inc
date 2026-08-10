"""Modes, training, UI and diagnostics."""

from __future__ import annotations

import pytest

from neurogrip.control.controller import HandController, HandState
from neurogrip.core.types import HandPose, ModeId
from neurogrip.diagnostics.console import DebugConsole
from neurogrip.diagnostics.metrics import MetricsRegistry
from neurogrip.diagnostics.selftest import (
    SelfTestRunner,
    TestOutcome,
)
from neurogrip.diagnostics.selftest import TestResult as SelfTestResult
from neurogrip.emg.pipeline import ChannelFrame, EmgFrame
from neurogrip.fusion.fusion import DecisionFusion
from neurogrip.hal.servo.simulated import SimulatedServoBus
from neurogrip.modes.base import ModeContext
from neurogrip.modes.manager import ModeManager
from neurogrip.modes.profiles import build_modes
from neurogrip.safety.monitor import SafetyState
from neurogrip.training.achievements import AchievementTracker
from neurogrip.training.exercises import Difficulty, create_exercise
from neurogrip.training.session import TrainingSession
from neurogrip.training.stats import SessionRecord, TrainingStats, Trend
from neurogrip.ui.renderers import NullRenderer, TextRenderer
from neurogrip.ui.screens import ViewModel, build_scene
from neurogrip.ui.theme import AccessibilitySettings, Theme, ThemeMode


def _emg(flexor=0.0, extensor=0.0, timestamp=0.0) -> EmgFrame:
    return EmgFrame(
        timestamp=timestamp,
        channels=(
            ChannelFrame(index=0, name="Flexor", role="flexor", activation=flexor),
            ChannelFrame(index=1, name="Extensor", role="extensor", activation=extensor),
        ),
    )


def _hand(**kwargs) -> HandState:
    defaults = dict(pose=HandPose.open_hand(), commanded=HandPose.open_hand())
    defaults.update(kwargs)
    return HandState(**defaults)


class _StubPlanner:
    name = "stub"

    def plan(self, context):
        return None


@pytest.fixture
def controller(clock, bus) -> HandController:
    controller = HandController(SimulatedServoBus(clock), clock, bus, publish_state=False)
    controller.start()
    controller.enable()
    return controller


@pytest.fixture
def modes(clock, bus, controller):
    from neurogrip.emg.gestures import ThresholdGestureClassifier
    from neurogrip.emg.intent import IntentEngine

    fusion = DecisionFusion(_StubPlanner(), clock)
    engine = IntentEngine(ThresholdGestureClassifier(), clock)
    manager = ModeManager(
        build_modes(controller, fusion, clock, bus), clock, bus, engine, min_dwell_s=0.0
    )
    manager.start()
    return manager


class TestModes:
    def test_starts_in_the_default_mode(self, modes):
        assert modes.current is ModeId.AI_ASSIST

    def test_manual_mode_disables_the_ai(self, modes):
        modes.activate(ModeId.MANUAL)
        assert not modes.active.profile.policy.ai_enabled
        assert modes.active.profile.show_ai_disabled_banner

    def test_manual_mode_does_not_even_run_the_camera(self, modes):
        modes.activate(ModeId.MANUAL)
        # Not merely ignored: not processed at all, so no stale perception can
        # influence anything.
        assert modes.active.profile.vision_rate_hz == 0.0

    def test_sports_mode_is_faster(self, modes):
        modes.activate(ModeId.SPORTS)
        sports = modes.active.profile
        assist = modes.mode(ModeId.AI_ASSIST).profile
        assert sports.policy.speed_ceiling > assist.policy.speed_ceiling
        assert sports.intent_settings.dwell_s < assist.intent_settings.dwell_s
        assert not sports.s_curve

    def test_training_mode_disables_the_ai_and_lowers_force(self, modes):
        modes.activate(ModeId.TRAINING)
        assert not modes.active.profile.policy.ai_enabled
        assert modes.active.profile.policy.force_ceiling < 0.6

    def test_a_critical_fault_blocks_a_mode_change(self, modes):
        from neurogrip.core.errors import Severity

        blocked = SafetyState(motion_allowed=False, severity=Severity.CRITICAL)
        assert not modes.activate(ModeId.SPORTS, safety=blocked)
        assert modes.rejections == 1

    def test_ai_modes_are_refused_when_assistance_is_unavailable(self, modes):
        degraded = SafetyState(ai_allowed=False, motion_allowed=True)
        assert not modes.activate(ModeId.SPORTS, safety=degraded)
        assert modes.activate(ModeId.MANUAL, safety=degraded)

    def test_min_dwell_debounces_rapid_changes(self, clock, bus, controller):
        from neurogrip.emg.gestures import ThresholdGestureClassifier
        from neurogrip.emg.intent import IntentEngine

        manager = ModeManager(
            build_modes(controller, DecisionFusion(_StubPlanner(), clock), clock, bus),
            clock,
            bus,
            IntentEngine(ThresholdGestureClassifier(), clock),
            min_dwell_s=1.0,
        )
        manager.start()
        assert not manager.activate(ModeId.MANUAL)
        clock.advance(1.5)
        assert manager.activate(ModeId.MANUAL)

    def test_fallback_to_manual_is_forced_and_remembers_the_user_choice(self, modes, clock):
        modes.activate(ModeId.SPORTS)
        assert modes.fall_back_to_manual("camera failed")
        assert modes.current is ModeId.MANUAL

        context = ModeContext(
            timestamp=clock.monotonic(),
            hand=_hand(),
            intent=None,
            emg=None,
            vision=None,
            safety=SafetyState(),
        )
        modes.update(context)
        clock.advance(3.0)
        modes.update(
            ModeContext(
                timestamp=clock.monotonic(),
                hand=_hand(),
                intent=None,
                emg=None,
                vision=None,
                safety=SafetyState(),
            )
        )
        # The user picked Sports; once the fault clears they get it back.
        assert modes.current is ModeId.SPORTS

    def test_cycling_walks_the_quick_switch_order(self, modes):
        first = modes.current
        modes.cycle()
        assert modes.current is not first


class TestTrainingExercises:
    def test_every_exercise_starts_and_produces_state(self, clock):
        for key in ("reaction", "accuracy", "isolation", "strength", "consistency"):
            exercise = create_exercise(key, clock)
            exercise.start(Difficulty.MEDIUM, clock.monotonic())
            state = exercise.update(_emg(0.5), _hand(), clock.monotonic())
            assert state.trials_total > 0
            assert isinstance(state.prompt, str)

    def test_difficulty_scales_are_ordered(self):
        assert Difficulty.BEGINNER.scale < Difficulty.MEDIUM.scale < Difficulty.EXPERT.scale
        assert Difficulty.MEDIUM.next() is Difficulty.HARD
        assert Difficulty.BEGINNER.previous() is Difficulty.BEGINNER

    def test_reaction_trainer_penalises_a_false_start(self, clock):
        exercise = create_exercise("reaction", clock)
        exercise.start(Difficulty.MEDIUM, clock.monotonic())
        state = exercise.update(_emg(0.9), clock and _hand(), clock.monotonic())
        assert "early" in state.prompt.lower()
        assert exercise.results and not exercise.results[0].success

    def test_reaction_trainer_scores_a_fast_response_highly(self, clock):
        exercise = create_exercise("reaction", clock)
        exercise.start(Difficulty.MEDIUM, clock.monotonic())
        for _ in range(2000):
            state = exercise.update(_emg(0.0), _hand(), clock.monotonic())
            if state.phase == "prompt":
                break
            clock.advance(0.01)
        clock.advance(0.2)
        exercise.update(_emg(0.8), _hand(), clock.monotonic())
        assert exercise.results[-1].success
        assert exercise.results[-1].score > 0.8

    def test_accuracy_rewards_holding_the_target(self, clock):
        exercise = create_exercise("accuracy", clock)
        exercise.start(Difficulty.EASY, clock.monotonic())
        state = exercise.update(_emg(0.0), _hand(), clock.monotonic())
        for _ in range(400):
            clock.advance(0.01)
            state = exercise.update(_emg(state.target), _hand(), clock.monotonic())
            if exercise.results:
                break
        assert exercise.results[0].success
        assert exercise.results[0].score > 0.7

    def test_harder_difficulty_narrows_the_accuracy_band(self, clock):
        easy = create_exercise("accuracy", clock)
        easy.start(Difficulty.BEGINNER, 0.0)
        hard = create_exercise("accuracy", clock)
        hard.start(Difficulty.EXPERT, 0.0)
        assert hard.tolerance < easy.tolerance


class TestTrainingSession:
    def test_session_completes_and_reports_a_summary(self, clock, bus):
        session = TrainingSession(clock, bus)
        assert session.start("accuracy", Difficulty.EASY)
        for _ in range(20000):
            if not session.active:
                break
            clock.advance(0.01)
            state = session.state
            level = state.target if state else 0.5
            session.update(_emg(level), _hand(), clock.monotonic())
        summary = session.summary
        assert summary is not None
        assert summary.trials > 0
        assert 0 <= summary.mean_score <= 1
        assert summary.advice

    def test_unknown_exercise_is_refused(self, clock, bus):
        assert not TrainingSession(clock, bus).start("nonexistent")

    def test_stopping_early_still_records_what_happened(self, clock, bus):
        session = TrainingSession(clock, bus)
        session.start("reaction")
        for _ in range(100):
            clock.advance(0.01)
            session.update(_emg(0.0), _hand(), clock.monotonic())
        summary = session.stop("user stopped")
        assert summary is not None
        assert not session.active


class TestTrainingStats:
    def _record(self, score=0.9, exercise="accuracy", difficulty=Difficulty.MEDIUM, day=0):
        return SessionRecord(
            exercise=exercise,
            difficulty=difficulty,
            trials=8,
            mean_score=score,
            best_score=score,
            success_rate=1.0,
            duration_s=60.0,
            timestamp=day * 86400.0 + 3600,
        )

    def test_records_accumulate(self, tmp_path):
        stats = TrainingStats()
        stats.record(self._record())
        stats.record(self._record(score=0.7))
        assert stats.total_sessions == 2
        assert stats.progress("accuracy").sessions == 2
        assert stats.progress("accuracy").best_score == pytest.approx(0.9)

    def test_streak_counts_consecutive_days(self):
        stats = TrainingStats()
        for day in range(3):
            stats.record(self._record(day=day))
        assert stats.streak_days == 3

    def test_a_missed_day_resets_the_streak(self):
        stats = TrainingStats()
        stats.record(self._record(day=0))
        stats.record(self._record(day=5))
        assert stats.streak_days == 1

    def test_trend_detects_improvement(self):
        stats = TrainingStats()
        for index, score in enumerate((0.3, 0.35, 0.4, 0.6, 0.7, 0.8)):
            stats.record(self._record(score=score, day=index))
        assert stats.progress("accuracy").trend is Trend.IMPROVING

    def test_qualifying_streak_breaks_on_a_bad_session(self):
        stats = TrainingStats()
        stats.record(self._record(score=0.9))
        stats.record(self._record(score=0.5))
        stats.record(self._record(score=0.9))
        assert stats.qualifying_streak("accuracy", Difficulty.MEDIUM, 0.8) == 1

    def test_persists_across_instances(self, tmp_path):
        path = tmp_path / "training.json"
        stats = TrainingStats(path)
        stats.record(self._record())
        stats.set_difficulty("accuracy", Difficulty.HARD)

        reloaded = TrainingStats(path)
        assert reloaded.total_sessions == 1
        assert reloaded.difficulty_for("accuracy") is Difficulty.HARD

    def test_a_corrupt_file_is_ignored_rather_than_fatal(self, tmp_path):
        path = tmp_path / "training.json"
        path.write_text("{not json", encoding="utf-8")
        stats = TrainingStats(path)  # must not raise
        assert stats.total_sessions == 0


class TestAchievements:
    def _record(self, **kwargs):
        defaults = dict(
            exercise="accuracy",
            difficulty=Difficulty.MEDIUM,
            trials=8,
            mean_score=0.5,
            best_score=0.6,
            success_rate=0.8,
            duration_s=60.0,
            timestamp=3600.0,
        )
        defaults.update(kwargs)
        return SessionRecord(**defaults)

    def test_first_session_unlocks(self):
        tracker = AchievementTracker()
        stats = TrainingStats()
        record = self._record()
        stats.record(record)
        unlocked = tracker.evaluate(record, stats)
        assert any(a.key == "first_session" for a in unlocked)

    def test_achievements_unlock_only_once(self):
        tracker = AchievementTracker()
        stats = TrainingStats()
        record = self._record()
        stats.record(record)
        tracker.evaluate(record, stats)
        assert tracker.evaluate(record, stats) == []

    def test_a_broken_condition_does_not_break_the_session(self):
        from neurogrip.training.achievements import Achievement

        tracker = AchievementTracker(
            (
                Achievement(
                    key="bad", title="Bad", description="", condition=lambda r, s: 1 / 0
                ),
            )
        )
        assert tracker.evaluate(self._record(), TrainingStats()) == []


class TestUi:
    def _view_model(self, **kwargs) -> ViewModel:
        defaults = dict(theme=Theme(), route="dashboard", hand=_hand())
        defaults.update(kwargs)
        return ViewModel(**defaults)

    def test_manual_mode_shows_the_ai_disabled_banner(self):
        """A specification requirement, made checkable."""
        scene = build_scene(self._view_model(mode=ModeId.MANUAL, ai_enabled=False))
        assert scene.banner is not None
        assert "AI DISABLED" in scene.banner.text

    def test_ai_assist_mode_shows_no_disabled_banner(self):
        scene = build_scene(self._view_model(mode=ModeId.AI_ASSIST, ai_enabled=True))
        assert scene.banner is None

    def test_an_engaged_estop_dominates_the_banner(self):
        scene = build_scene(
            self._view_model(
                mode=ModeId.MANUAL, ai_enabled=False, safety=SafetyState(estop_engaged=True)
            )
        )
        assert "EMERGENCY STOP" in scene.banner.text

    def test_every_route_renders(self):
        from neurogrip.ui.screens import ROUTES

        for route in ROUTES:
            scene = build_scene(self._view_model(route=route))
            assert scene.title
            assert scene.nav

    def test_dashboard_exposes_a_stop_button(self):
        scene = build_scene(self._view_model())
        assert any(button.action == "estop" for button in scene.buttons())

    def test_the_hand_graphic_reflects_the_pose(self):
        scene = build_scene(self._view_model(hand=_hand(pose=HandPose.uniform(0.7))))
        graphic = scene.find("hand.graphic")
        assert graphic is not None
        assert graphic.pose == HandPose.uniform(0.7)

    def test_text_renderer_produces_readable_output(self):
        renderer = TextRenderer(width=100, colour=False)
        scene = build_scene(self._view_model(mode=ModeId.MANUAL, ai_enabled=False))
        output = renderer.render_to_string(scene, Theme())
        assert "DASHBOARD" in output
        assert "AI DISABLED" in output

    def test_null_renderer_retains_the_last_scene(self):
        renderer = NullRenderer()
        scene = build_scene(self._view_model())
        renderer.render(scene, Theme())
        assert renderer.last_scene is scene

    def test_scene_text_content_covers_the_banner_and_labels(self):
        scene = build_scene(self._view_model(mode=ModeId.MANUAL, ai_enabled=False))
        assert "AI DISABLED" in scene.text_content()


class TestTheme:
    def test_high_contrast_overrides_the_selected_mode(self):
        theme = Theme().with_accessibility(AccessibilitySettings(high_contrast=True))
        assert theme.with_mode(ThemeMode.LIGHT).palette.background == "#000000"

    def test_font_scale_grows_the_touch_target_but_never_shrinks_it(self):
        small = Theme().with_accessibility(AccessibilitySettings(font_scale=0.5))
        large = Theme().with_accessibility(AccessibilitySettings(font_scale=1.6))
        assert small.touch_target >= 44
        assert large.touch_target > small.touch_target

    def test_reduce_motion_disables_animations(self):
        theme = Theme().with_accessibility(AccessibilitySettings(reduce_motion=True))
        assert not theme.animations_enabled


class TestDiagnostics:
    def test_metrics_registry_tracks_counters_and_histograms(self, clock):
        registry = MetricsRegistry(clock)
        registry.counter("frames").increment(5)
        histogram = registry.histogram("latency")
        for value in (1.0, 2.0, 50.0):
            histogram.observe(value)
        snapshot = registry.snapshot()
        assert snapshot["counters"]["frames"] == 5
        assert snapshot["histograms"]["latency"]["count"] == 3

    def test_selftest_reports_and_aggregates(self, clock):
        runner = SelfTestRunner(clock)
        runner.register("good", "", lambda: SelfTestResult("good", TestOutcome.PASS))
        runner.register("warn", "", lambda: SelfTestResult("warn", TestOutcome.WARN, "hmm"))
        report = runner.run()
        assert report.ok  # warnings do not block
        assert report.passed == 1 and report.warnings == 1

    def test_a_failing_test_blocks(self, clock):
        runner = SelfTestRunner(clock)
        runner.register("bad", "", lambda: SelfTestResult("bad", TestOutcome.FAIL, "broken"))
        assert not runner.run().ok

    def test_a_raising_test_is_a_failure_not_a_crash(self, clock):
        runner = SelfTestRunner(clock)

        def explode():
            raise RuntimeError("boom")

        runner.register("explodes", "", explode)
        report = runner.run()
        assert not report.ok

    def test_motion_tests_are_skipped_unless_explicitly_allowed(self, clock):
        runner = SelfTestRunner(clock)
        runner.register(
            "sweep", "", lambda: SelfTestResult("sweep", TestOutcome.PASS), requires_motion=True
        )
        assert runner.run().skipped == 1
        assert runner.run(allow_motion=True).skipped == 0

    def test_console_refuses_dangerous_commands_until_armed(self):
        console = DebugConsole()
        console.register("boom", "dangerous", lambda args: None, dangerous=True)
        result = console.execute("boom")
        assert not result.ok
        assert "arm" in result.output

    def test_console_reports_unknown_commands_with_a_hint(self):
        console = DebugConsole()
        result = console.execute("helpme")
        assert not result.ok
        assert "help" in result.output

    def test_console_survives_a_command_that_raises(self):
        console = DebugConsole()

        def explode(args):
            raise RuntimeError("nope")

        console.register("explode", "", explode)
        result = console.execute("explode")
        assert not result.ok
