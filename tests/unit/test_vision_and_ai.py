"""Vision backends, HGGD-MCU decoding, tracking, depth, affordances, planners."""

from __future__ import annotations

import math

import pytest

from neurogrip.ai.grasp import build_default_planner
from neurogrip.ai.grasp.base import GraspContext, PlanSource
from neurogrip.ai.grasp.composite import CompositeGraspPlanner
from neurogrip.ai.grasp.heuristic import HeuristicGraspPlanner
from neurogrip.ai.grasp.hggd import HggdGraspPlanner
from neurogrip.ai.objects import AffordanceDatabase
from neurogrip.control.grips import GripLibrary
from neurogrip.control.kinematics import HandKinematics
from neurogrip.core.types import GraspType, HandPose, IntentKind, ModeId
from neurogrip.emg.intent import IntentEstimate
from neurogrip.hal.camera.base import Frame, PixelFormat
from neurogrip.hal.camera.simulated import SceneObject, SimulatedCamera
from neurogrip.vision.backend import available_backends, create_backend
from neurogrip.vision.backends.hggd_mcu import (
    HGGD_MCU_CLASSES,
    ClassicalHeatmapSession,
    HeatmapTensors,
    HggdMcuBackend,
    HggdMcuSettings,
    decode_heatmap,
)
from neurogrip.vision.backends.mock import MockSettings, MockVisionBackend
from neurogrip.vision.backends.null import NullVisionBackend
from neurogrip.vision.depth import OBJECT_SIZE_PRIORS, MonocularDepthEstimator
from neurogrip.vision.pipeline import VisionPipeline
from neurogrip.vision.preprocess import letterbox
from neurogrip.vision.tracking import ObjectTracker
from neurogrip.vision.types import (
    BoundingBox,
    Detection,
    GraspApproach,
    GraspCandidate,
    VisionCapability,
    VisionResult,
)


class TestBoundingBox:
    def test_normalises_reversed_corners(self):
        box = BoundingBox(0.8, 0.9, 0.2, 0.1)
        assert (box.x1, box.y1, box.x2, box.y2) == (0.2, 0.1, 0.8, 0.9)

    def test_iou_of_identical_boxes_is_one(self):
        box = BoundingBox(0.1, 0.1, 0.5, 0.5)
        assert box.iou(box) == pytest.approx(1.0)

    def test_iou_of_disjoint_boxes_is_zero(self):
        assert BoundingBox(0, 0, 0.1, 0.1).iou(BoundingBox(0.5, 0.5, 0.6, 0.6)) == 0.0

    def test_centrality(self):
        centred = BoundingBox(0.4, 0.4, 0.6, 0.6)
        corner = BoundingBox(0.0, 0.0, 0.2, 0.2)
        assert centred.distance_from_center() < corner.distance_from_center()


class TestVisionResult:
    def test_primary_prefers_a_central_stable_detection(self):
        edge = Detection("box", 0.95, BoundingBox(0.0, 0.0, 0.15, 0.15), track_id=1, age=1)
        centre = Detection("bottle", 0.75, BoundingBox(0.4, 0.35, 0.6, 0.65), track_id=2, age=10)
        result = VisionResult(timestamp=0.0, detections=(edge, centre))
        # The most confident detection is not necessarily the one the user is
        # reaching for.
        assert result.primary is centre

    def test_freshness(self):
        result = VisionResult(timestamp=1.0)
        assert result.is_fresh(1.2, max_age=0.5)
        assert not result.is_fresh(2.0, max_age=0.5)

    def test_object_confidence_is_zero_without_detections(self):
        assert VisionResult(timestamp=0.0).object_confidence == 0.0


class TestHggdMcuDecoding:
    def _tensors(self, peak_x=2, peak_y=1, width=8, height=6, bins=12) -> HeatmapTensors:
        heatmap = [0.05] * (width * height)
        heatmap[peak_y * width + peak_x] = 0.9
        angle = [0.0] * (width * height * bins)
        # Bin 3 of 12 → about 52°.
        angle[(peak_y * width + peak_x) * bins + 3] = 1.0
        return HeatmapTensors(
            width=width,
            height=height,
            angle_bins=bins,
            heatmap=heatmap,
            angle=angle,
            grasp_width=[0.2] * (width * height),
            quality=[0.8] * (width * height),
        )

    def _identity_letterbox(self, width=64, height=48):
        _, info = letterbox([0] * (width * height), width, height, width, height)
        return info

    def test_finds_the_single_peak(self):
        settings = HggdMcuSettings(score_threshold=0.3)
        grasps = decode_heatmap(self._tensors(), settings, self._identity_letterbox())
        assert len(grasps) == 1
        assert grasps[0].quality > 0.5

    def test_peak_maps_to_the_right_image_position(self):
        settings = HggdMcuSettings(score_threshold=0.3)
        grasps = decode_heatmap(
            self._tensors(peak_x=2, peak_y=1, width=8, height=6),
            settings,
            self._identity_letterbox(),
        )
        # Grid cell (2, 1) of an 8x6 grid → roughly (0.31, 0.25).
        assert grasps[0].center_x == pytest.approx(0.31, abs=0.08)
        assert grasps[0].center_y == pytest.approx(0.25, abs=0.08)

    def test_angle_comes_from_the_argmax_bin(self):
        settings = HggdMcuSettings(score_threshold=0.3)
        grasps = decode_heatmap(self._tensors(), settings, self._identity_letterbox())
        expected = (3 + 0.5) * math.pi / 12
        assert grasps[0].angle == pytest.approx(expected, abs=1e-6)

    def test_sub_threshold_peaks_are_ignored(self):
        settings = HggdMcuSettings(score_threshold=0.95)
        assert decode_heatmap(self._tensors(), settings, self._identity_letterbox()) == []

    def test_nms_suppresses_neighbouring_duplicates(self):
        settings = HggdMcuSettings(score_threshold=0.3, nms_distance=0.5)
        tensors = self._tensors()
        # A second, weaker peak two cells away.
        tensors.heatmap[1 * 8 + 4] = 0.85
        grasps = decode_heatmap(tensors, settings, self._identity_letterbox())
        assert len(grasps) == 1

    def test_grasp_candidates_are_capped(self):
        settings = HggdMcuSettings(score_threshold=0.01, nms_distance=0.0, max_grasps=3)
        base = self._tensors()
        tensors = HeatmapTensors(
            width=base.width,
            height=base.height,
            angle_bins=base.angle_bins,
            heatmap=[0.4 + 0.05 * ((i * 7) % 5) for i in range(base.width * base.height)],
            angle=base.angle,
            grasp_width=base.grasp_width,
            quality=base.quality,
        )
        assert len(decode_heatmap(tensors, settings, self._identity_letterbox())) <= 3

    def test_angle_maps_to_a_sensible_approach(self):
        settings = HggdMcuSettings(score_threshold=0.3)
        grasps = decode_heatmap(self._tensors(), settings, self._identity_letterbox())
        assert grasps[0].approach in tuple(GraspApproach)


class TestHggdMcuBackend:
    def _frame(self) -> Frame:
        camera = SimulatedCamera(scene=SceneObject(label="bottle", shape="cylinder"))
        camera.open()
        frame = camera.read()
        assert frame is not None
        return frame

    def test_falls_back_to_the_classical_session_without_weights(self):
        backend = HggdMcuBackend(settings=HggdMcuSettings(model_path="does/not/exist.onnx"))
        backend.initialize()
        info = backend.info()
        # Missing weights must degrade, not fail: the user still has a hand.
        assert info.runtime == "classical"
        assert info.is_degraded
        assert VisionCapability.GRASP in backend.capabilities

    def test_processes_a_frame_without_raising(self):
        backend = HggdMcuBackend(settings=HggdMcuSettings(model_path="missing.onnx"))
        backend.initialize()
        result = backend.process(self._frame())
        assert result.ok
        assert result.backend == "hggd_mcu"

    def test_uninitialised_backend_reports_an_error_rather_than_raising(self):
        backend = HggdMcuBackend(settings=HggdMcuSettings())
        result = backend.process(self._frame())
        assert not result.ok

    def test_classical_session_emits_the_declared_tensor_shapes(self):
        settings = HggdMcuSettings(input_width=64, input_height=48, stride=8, angle_bins=12)
        session = ClassicalHeatmapSession(settings)
        pixels = [((x // 8) % 2) * 200 for x in range(64 * 48)]
        tensors = session.run(pixels, 64, 48)
        assert tensors.width == 8 and tensors.height == 6
        assert len(tensors.heatmap) == 48
        assert len(tensors.angle) == 48 * 12

    def test_class_list_and_affordances_agree(self):
        database = AffordanceDatabase()
        # Every class the model can predict should have a handling policy, or it
        # silently falls back to a generic grasp.
        assert database.validate_against(HGGD_MCU_CLASSES) == ()


class TestOtherBackends:
    def test_registry_lists_the_bundled_backends(self):
        names = available_backends()
        assert {"hggd_mcu", "mock", "null", "onnx_detector"} <= set(names)

    def test_unknown_backend_falls_back_to_null(self, config):
        backend = create_backend("does-not-exist", config)
        assert backend.capabilities is VisionCapability.NONE

    def test_null_backend_returns_an_empty_but_successful_result(self):
        backend = NullVisionBackend()
        backend.initialize()
        frame = Frame(4, 4, PixelFormat.GRAY8, bytes(16), timestamp=1.0)
        result = backend.process(frame)
        # Empty is not an error: "I looked and there was nothing" is valid.
        assert result.ok
        assert result.detections == ()

    def test_mock_backend_reads_the_simulated_scene(self):
        camera = SimulatedCamera(scene=SceneObject(label="cup", shape="cylinder"))
        camera.open()
        backend = MockVisionBackend(settings=MockSettings())
        backend.initialize()
        result = backend.process(camera.read())
        assert result.primary is not None
        assert result.primary.label == "cup"
        assert result.best_grasp is not None

    def test_mock_backend_honours_the_false_negative_rate(self):
        camera = SimulatedCamera(scene=SceneObject(label="cup"))
        camera.open()
        backend = MockVisionBackend(settings=MockSettings(false_negative_rate=1.0))
        backend.initialize()
        result = backend.process(camera.read())
        assert result.detections == ()

    def test_real_backends_do_not_read_the_simulation_ground_truth(self):
        """The mock is allowed to cheat; the real ones must not."""
        camera = SimulatedCamera(scene=SceneObject(label="bottle"))
        camera.open()
        frame = camera.read()
        assert "scene" in frame.metadata  # the truth is present…

        backend = HggdMcuBackend(settings=HggdMcuSettings(model_path="missing.onnx"))
        backend.initialize()
        result = backend.process(frame)
        # …and HGGD-MCU never reports it, because it never looks at it.
        assert all(d.label != "bottle" for d in result.detections)


class TestTracking:
    def _detection(self, label="bottle", confidence=0.9, x=0.4):
        return Detection(label, confidence, BoundingBox(x, 0.4, x + 0.2, 0.6))

    def test_assigns_and_keeps_a_track_id(self):
        tracker = ObjectTracker()
        first = tracker.update((self._detection(),), 0.0)
        second = tracker.update((self._detection(x=0.42),), 0.1)
        assert first[0].track_id == second[0].track_id
        assert second[0].age > first[0].age

    def test_survives_a_brief_dropout(self):
        tracker = ObjectTracker(max_missed=3)
        tracker.update((self._detection(),), 0.0)
        for step in range(2):
            tracker.update((), 0.1 * (step + 1))
        recovered = tracker.update((self._detection(),), 0.4)
        assert recovered[0].track_id == 1

    def test_label_voting_rejects_a_single_misclassification(self):
        tracker = ObjectTracker()
        for step in range(6):
            tracker.update((self._detection(label="bottle"),), step * 0.1)
        flipped = tracker.update((self._detection(label="can"),), 0.7)
        # One odd frame must not change what the hand thinks it is holding.
        assert flipped[0].label == "bottle"

    def test_low_confidence_detections_are_ignored(self):
        tracker = ObjectTracker(min_confidence=0.5)
        assert tracker.update((self._detection(confidence=0.2),), 0.0) == ()


class TestDepth:
    def test_larger_apparent_size_means_closer(self):
        estimator = MonocularDepthEstimator(image_width=640, image_height=480)
        near = estimator.estimate(
            Detection("bottle", 0.9, BoundingBox(0.3, 0.1, 0.7, 0.9))
        )
        far = estimator.estimate(Detection("bottle", 0.9, BoundingBox(0.45, 0.4, 0.55, 0.6)))
        assert near.distance_m < far.distance_m

    def test_unknown_classes_report_low_confidence(self):
        estimator = MonocularDepthEstimator()
        known = estimator.estimate(Detection("bottle", 0.9, BoundingBox(0.3, 0.2, 0.7, 0.8)))
        unknown = estimator.estimate(Detection("gizmo", 0.9, BoundingBox(0.3, 0.2, 0.7, 0.8)))
        assert unknown.confidence < known.confidence

    def test_estimates_are_clamped_to_the_working_envelope(self):
        estimator = MonocularDepthEstimator(max_distance_m=1.0)
        tiny = estimator.estimate(Detection("bottle", 0.9, BoundingBox(0.5, 0.5, 0.505, 0.51)))
        assert tiny.distance_m <= 1.0

    def test_every_model_class_has_a_size_prior(self):
        for label in HGGD_MCU_CLASSES:
            if label != "unknown":
                assert label in OBJECT_SIZE_PRIORS, f"no size prior for '{label}'"


class TestAffordances:
    def test_known_object_returns_its_policy(self):
        affordance = AffordanceDatabase().get("bottle")
        assert affordance.primary_grasp is GraspType.CYLINDRICAL
        assert affordance.heavy

    def test_aliases_resolve(self):
        database = AffordanceDatabase()
        assert database.get("mug").label == "cup"
        assert database.get("apple").label == "fruit"

    def test_unknown_object_gets_the_conservative_default(self):
        affordance = AffordanceDatabase().get("unheard-of")
        assert affordance.label == "unknown"
        assert affordance.max_force <= 0.5
        assert affordance.speed_scale < 1.0

    def test_fragile_objects_have_low_force_ceilings(self):
        database = AffordanceDatabase()
        for label in ("fruit", "can", "card"):
            assert database.get(label).max_force <= 0.5

    def test_force_is_the_lower_of_object_and_mode_limits(self):
        affordance = AffordanceDatabase().get("tool")  # 0.78
        assert affordance.force_for(0.5) == pytest.approx(0.5)
        assert affordance.force_for(0.9) == pytest.approx(0.78)


class TestGraspPlanners:
    def _context(self, vision=None, strength=0.7, mode=ModeId.AI_ASSIST):
        return GraspContext(
            intent=IntentEstimate(
                kind=IntentKind.CLOSE, confidence=0.9, strength=strength, timestamp=1.0
            ),
            vision=vision,
            current_pose=HandPose.open_hand(),
            mode=mode,
            timestamp=1.0,
            force_ceiling=0.85,
        )

    def _vision(self, label="bottle", confidence=0.9, grasps=()):
        from neurogrip.vision.types import DepthEstimate

        return VisionResult(
            timestamp=1.0,
            detections=(
                Detection(label, confidence, BoundingBox(0.42, 0.3, 0.58, 0.72), track_id=1, age=10,
                          attributes={"label_agreement": 1.0}),
            ),
            grasps=grasps,
            depth=DepthEstimate(distance_m=0.3, confidence=0.8, method="sensor"),
        )

    def _planners(self):
        grips, affordances, kinematics = GripLibrary(), AffordanceDatabase(), HandKinematics()
        return grips, affordances, kinematics

    def test_heuristic_selects_a_cylindrical_grasp_for_a_bottle(self):
        planner = HeuristicGraspPlanner(*self._planners())
        plan = planner.plan(self._context(vision=self._vision("bottle")))
        assert plan is not None
        assert plan.grasp in (GraspType.CYLINDRICAL, GraspType.POWER)
        assert plan.reasons

    def test_heuristic_selects_a_pinch_for_a_pen(self):
        planner = HeuristicGraspPlanner(*self._planners())
        plan = planner.plan(self._context(vision=self._vision("pen")))
        assert plan.grasp in (GraspType.TRIPOD, GraspType.PRECISION_PINCH)

    def test_heuristic_limits_force_for_fragile_objects(self):
        planner = HeuristicGraspPlanner(*self._planners())
        fragile = planner.plan(self._context(vision=self._vision("fruit")))
        sturdy = planner.plan(self._context(vision=self._vision("tool")))
        assert fragile.force < sturdy.force

    def test_heuristic_still_plans_without_any_vision(self):
        planner = HeuristicGraspPlanner(*self._planners())
        plan = planner.plan(self._context(vision=None))
        assert plan is not None
        assert plan.grasp is GraspType.POWER
        assert plan.source is PlanSource.DEFAULT

    def test_user_effort_scales_the_force(self):
        planner = HeuristicGraspPlanner(*self._planners())
        gentle = planner.plan(self._context(vision=self._vision(), strength=0.2))
        firm = planner.plan(self._context(vision=self._vision(), strength=1.0))
        assert firm.force > gentle.force

    def test_hggd_planner_declines_without_grasp_candidates(self):
        planner = HggdGraspPlanner(*self._planners())
        assert planner.plan(self._context(vision=self._vision())) is None

    def test_hggd_planner_maps_a_narrow_opening_to_a_pinch(self):
        planner = HggdGraspPlanner(*self._planners())
        candidate = GraspCandidate(
            center_x=0.5, center_y=0.5, angle=0.0, width=0.05, quality=0.9,
            depth_m=0.3, width_m=0.015, approach=GraspApproach.TOP_DOWN, label="pen",
        )
        plan = planner.plan(self._context(vision=self._vision("pen", grasps=(candidate,))))
        assert plan is not None
        assert plan.grasp is GraspType.PRECISION_PINCH
        assert plan.source is PlanSource.LEARNED

    def test_hggd_planner_ignores_grasps_far_from_the_aim_point(self):
        planner = HggdGraspPlanner(*self._planners())
        offset = GraspCandidate(
            center_x=0.95, center_y=0.95, angle=0.0, width=0.2, quality=0.95, width_m=0.06
        )
        assert planner.plan(self._context(vision=self._vision(grasps=(offset,)))) is None

    def test_composite_always_returns_a_plan(self):
        grips, _affordances, kinematics = self._planners()
        composite = CompositeGraspPlanner((), grips, kinematics)
        plan = composite.plan(self._context())
        assert plan is not None
        assert plan.source is PlanSource.DEFAULT
        assert plan.force <= 0.5  # the safe default is deliberately gentle

    def test_composite_prefers_the_learned_planner(self):
        grips, affordances, kinematics = self._planners()
        composite = CompositeGraspPlanner(
            (HggdGraspPlanner(grips, affordances, kinematics),
             HeuristicGraspPlanner(grips, affordances, kinematics)),
            grips,
            kinematics,
        )
        candidate = GraspCandidate(
            center_x=0.5, center_y=0.5, angle=0.0, width=0.2, quality=0.9,
            depth_m=0.3, width_m=0.065, label="bottle",
        )
        plan = composite.plan(self._context(vision=self._vision(grasps=(candidate,))))
        assert plan.source is PlanSource.LEARNED
        assert composite.wins["hggd"] == 1

    def test_composite_falls_through_when_the_learned_planner_declines(self):
        grips, affordances, kinematics = self._planners()
        composite = CompositeGraspPlanner(
            (HggdGraspPlanner(grips, affordances, kinematics),
             HeuristicGraspPlanner(grips, affordances, kinematics)),
            grips,
            kinematics,
        )
        plan = composite.plan(self._context(vision=self._vision()))
        assert plan.source is PlanSource.HEURISTIC
        assert composite.wins["heuristic"] == 1

    def test_plans_respect_the_force_ceiling(self):
        grips, affordances, kinematics = self._planners()
        composite = CompositeGraspPlanner(
            (HeuristicGraspPlanner(grips, affordances, kinematics),), grips, kinematics
        )
        context = GraspContext(
            intent=IntentEstimate(IntentKind.CLOSE, 0.9, 1.0, 1.0),
            vision=self._vision("tool"),
            current_pose=HandPose.open_hand(),
            mode=ModeId.AI_ASSIST,
            timestamp=1.0,
            force_ceiling=0.3,
        )
        assert composite.plan(context).force <= 0.3

    def test_default_chain_is_built_from_configuration(self, config):
        grips, affordances, kinematics = self._planners()
        planner = build_default_planner(config, grips, affordances, kinematics)
        assert "hggd" in planner.name and "heuristic" in planner.name


class TestVisionPipeline:
    def test_produces_tracked_detections(self, clock, config):
        camera = SimulatedCamera(clock, scene=SceneObject(label="bottle"))
        pipeline = VisionPipeline(camera, create_backend("mock", config), clock)
        pipeline.start()
        result = None
        for _ in range(6):
            clock.advance(0.05)
            result = pipeline.tick() or result
        assert result is not None
        assert result.primary is not None
        assert result.primary.track_id > 0
        pipeline.stop()

    def test_reports_offline_health_without_a_camera(self, clock, config):
        pipeline = VisionPipeline(None, create_backend("null", config), clock)
        pipeline.start()
        assert pipeline.health().status.name == "OFFLINE"
        pipeline.stop()

    def test_a_backend_that_raises_does_not_propagate(self, clock, config):
        class Exploding:
            def initialize(self):
                pass

            def shutdown(self):
                pass

            @property
            def capabilities(self):
                return VisionCapability.NONE

            def info(self):
                from neurogrip.vision.backend import BackendInfo

                return BackendInfo(name="exploding")

            def process(self, frame):
                raise RuntimeError("inference blew up")

        camera = SimulatedCamera(clock, scene=SceneObject())
        pipeline = VisionPipeline(camera, Exploding(), clock)
        pipeline.start()
        clock.advance(0.05)
        result = pipeline.tick()  # must not raise
        assert result is not None and not result.ok
        pipeline.stop()
