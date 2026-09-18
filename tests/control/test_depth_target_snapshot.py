"""Main-thread raw geometry is immutable across concurrent Depth30 updates."""
from pathlib import Path
from types import SimpleNamespace
import sys
import threading

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import request_0513_modular as runtime
from car_control_modular.control_types import DistanceState, HazardState


RAW = (255.99012756347656, 78.54582214355469, 389.80328369140625, 455.6407470703125)
DISPLAY = (242.54516241976836, 40.28864742651828, 415.25703503457976, 479.0)
NOW = 100.0


@pytest.fixture
def owner(monkeypatch):
    monkeypatch.setattr(runtime.time, "monotonic", lambda: NOW)
    monkeypatch.setattr(runtime, "FOLLOW_ROTATION_ONLY", False)
    obj = object.__new__(runtime.PersonTracker)
    obj.frame_index = 420
    obj._active_capture_frame_id = 1015
    obj._active_capture_timestamp = NOW - .1
    obj._follow_controller = SimpleNamespace(active_target_id=1, search_state="none")
    obj.search_state = "none"
    obj._longitudinal_context_lock = threading.Lock()
    obj._control_update_lock = threading.Lock()
    obj._longitudinal_context = None
    obj._rknn_pipeline = SimpleNamespace(tracker=SimpleNamespace(last_identity_observations=[{
        "raw_track_id": 11, "uid": 1, "display_bbox": DISPLAY, "detector_bbox": RAW,
        "sample_metadata": {
            "capture_frame_id": 1015, "capture_timestamp": NOW - .1,
            "is_fresh": True, "source_detection_index": 0,
        },
        "assignment": {
            "uid": 1, "reason": "mapped", "bbox_quality_ok": True,
            "bbox_quality_tier": "strong", "reacquire_geometry_ok": True,
        },
    }]))
    obj.running = True
    obj._runtime_shutdown_requested = False
    obj._action_runtime_started = True
    obj._last_vision_control_ts = NOW - 1.0
    obj._last_vision_control_frame_index = 419
    obj._last_depth30_dedupe_log_frame = -1
    return obj


def persons():
    return [(DISPLAY, 1, .9, 75800.0)]


def test_production_requires_raw_geometry_without_altering_control_target(owner):
    assert runtime.DISTANCE_RUNTIME_CONFIG.vision_depth_require_detector_bbox
    target, = owner._persons_to_targets(persons(), width=640, height=480)
    assert target.bbox == DISPLAY and target.area == 75800.0
    assert target.depth_observation.bbox == RAW
    assert target.depth_observation.target_id == 1
    assert target.depth_observation.raw_track_id == 11
    assert target.depth_observation.capture_frame_id == 1015


def test_missing_or_ambiguous_raw_detector_never_publishes_context(owner):
    owner._rknn_pipeline.tracker.last_identity_observations *= 2
    owner._publish_longitudinal_context(640, 480, persons())
    assert owner._longitudinal_context is None
    owner._rknn_pipeline.tracker.last_identity_observations.clear()
    owner._publish_longitudinal_context(640, 480, persons())
    assert owner._longitudinal_context is None


def test_context_freezes_geometry_and_capture_before_next_inference(owner):
    owner._publish_longitudinal_context(640, 480, persons())
    context = owner._longitudinal_context
    owner._rknn_pipeline.tracker.last_identity_observations.clear()
    owner._active_capture_frame_id = 1020
    owner._active_capture_timestamp = NOW
    queued = []
    owner._queue_actions_for_persons = lambda *args, **kwargs: queued.append((args, kwargs))
    stop = threading.Event()
    owner._longitudinal_stop_event = SimpleNamespace(
        is_set=stop.is_set, wait=lambda duration: stop.set(),
    )
    owner._longitudinal_control_loop()
    assert len(queued) == 1
    assert queued[0][1]["depth_target_snapshot"] is context["person_targets"]
    assert queued[0][1]["depth_target_snapshot"][0].depth_observation.bbox == RAW
    assert queued[0][1]["evidence_capture_frame_id"] == 1015
    assert queued[0][1]["evidence_capture_timestamp"] == NOW - .1


@pytest.mark.parametrize("age", [.19, -.02])
def test_republishing_does_not_make_old_or_future_capture_fresh(owner, age):
    calls = []
    owner._queue_actions_for_persons_locked = lambda *args, **kwargs: calls.append(kwargs)
    owner._queue_actions_for_persons(
        640, 480, persons(), control_source="depth30", expected_target_id=1,
        context_published_ts=NOW, evidence_capture_timestamp=NOW - age,
    )
    assert calls == []


def test_control_lock_revalidates_uid_before_using_depth_snapshot(owner):
    owner._publish_longitudinal_context(640, 480, persons())
    calls = []
    owner._queue_actions_for_persons_locked = lambda *args, **kwargs: calls.append(kwargs)
    owner._follow_controller.active_target_id = 2
    owner._queue_actions_for_persons(
        640, 480, persons(), control_source="depth30", expected_target_id=1,
        depth_target_snapshot=owner._longitudinal_context["person_targets"],
        evidence_capture_frame_id=1015,
        context_published_ts=NOW, evidence_capture_timestamp=NOW - .1,
    )
    assert calls == []


def test_new_depth_not_delayed_by_just_completed_visual_frame(owner):
    owner._publish_longitudinal_context(640, 480, persons())
    owner._last_vision_control_ts = NOW
    owner._last_vision_control_frame_index = owner.frame_index
    context = owner._longitudinal_context
    calls = []
    owner._queue_actions_for_persons_locked = lambda *args, **kwargs: calls.append(kwargs)
    for _ in range(2):
        owner._queue_actions_for_persons(
            640, 480, persons(), control_source="depth30", expected_target_id=1,
            depth_target_snapshot=context["person_targets"],
            expected_frame_index=owner.frame_index,
            evidence_capture_frame_id=context["capture_frame_id"],
            evidence_capture_timestamp=context["capture_timestamp"],
        )
    # The independent sensor timestamp, not the shared RGB frame number,
    # decides whether these calls contain distinct observations.
    assert len(calls) == 2


@pytest.mark.parametrize("replacement", ["clear", "suspend", "republish", "unchanged"])
def test_copied_context_is_revoked_even_when_uid_and_capture_age_are_unchanged(owner, replacement):
    owner._publish_longitudinal_context(640, 480, persons())
    copied = dict(owner._longitudinal_context)
    if replacement == "clear":
        owner._clear_longitudinal_context()
    elif replacement == "suspend":
        owner._clear_longitudinal_context(revoke_translation=False, reason="visual_frame_begin")
    elif replacement == "republish":
        # Even identical capture metadata represents a different publication;
        # no old copy may survive a clear/re-publish transition.
        owner._publish_longitudinal_context(640, 480, persons())
    calls = []
    owner._queue_actions_for_persons_locked = lambda *args, **kwargs: calls.append(kwargs)
    owner._queue_actions_for_persons(
        640, 480, persons(), control_source="depth30", expected_target_id=1,
        depth_target_snapshot=copied["person_targets"],
        context_published_ts=copied["published_ts"], expected_frame_index=420,
        evidence_capture_frame_id=copied["capture_frame_id"],
        evidence_capture_timestamp=copied["capture_timestamp"],
    )
    assert len(calls) == int(replacement == "unchanged")


def test_depth_control_uses_snapshot_capture_not_current_inference_capture(owner):
    target, = owner._persons_to_targets(persons(), width=640, height=480)
    owner._rknn_pipeline.tracker.last_identity_observations.clear()
    owner._active_capture_frame_id = 1020
    owner._active_capture_timestamp = NOW
    owner._last_dispatched_action = runtime.ACTION_STOP
    owner._get_obstacle_status = lambda: dict(front=False, left=False, right=False)
    owner._action_runtime = SimpleNamespace(get_steering_feedback=lambda: None)
    owner._distance_runtime = SimpleNamespace(
        select_target=lambda targets: targets[0],
        get_frame_distance_state=lambda *args, **kwargs: DistanceState(source="vision_depth"),
    )
    owner._current_hazard_state_for_controller = lambda: HazardState()
    owner._depth_longitudinal_authority_enabled = lambda: False
    owner._follow_controller.set_last_dispatched = lambda command: None
    seen = []

    class CapturedFrame(Exception):
        pass

    def decide(frame_index, frame, **kwargs):
        seen.append(frame)
        raise CapturedFrame()

    owner._follow_controller.decide = decide
    with pytest.raises(CapturedFrame):
        owner._process_detections_modular(
            640, 480, persons(), control_source="depth30", depth_use_latest=True,
            depth_target_snapshot=(target,), evidence_capture_frame_id=1015,
            evidence_capture_timestamp=NOW - .1,
        )
    assert seen[0].persons[0].bbox == DISPLAY
    assert seen[0].persons[0].depth_observation.bbox == RAW
    assert seen[0].capture_frame_id == 1015
    assert seen[0].capture_timestamp == NOW - .1


def test_reacquire_depth_observation_uses_stable_uid_not_raw_track_id(owner):
    seen = []
    requests = []
    owner._action_runtime = SimpleNamespace(get_steering_feedback=lambda: None)

    def distance(width, target, **kwargs):
        seen.append(target)
        requests.append(kwargs)
        return DistanceState(source="vision_depth", source_detail="no_visual_target")

    owner._distance_runtime = SimpleNamespace(get_frame_distance_state=distance)
    owner._confirmed_search_reacquire_depth_streak = 0
    owner._confirmed_search_reacquire_depth_uid = None
    owner._confirmed_search_reacquire_depth_track_id = None
    owner._confirmed_search_reacquire_depth_last_frame = -1
    owner._observe_search_reacquire_depth(
        bbox=DISPLAY, track_id=11, uid=1, score=.9, area=75800, width=640, height=480,
    )
    assert seen[0].track_id == 1
    assert seen[0].depth_observation.raw_track_id == 11
    assert seen[0].depth_observation.bbox == RAW
    assert requests[0]["depth_use_latest"] is False
    assert requests[0]["capture_timestamp"] == seen[0].depth_observation.capture_timestamp
