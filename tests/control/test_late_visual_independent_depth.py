"""CAP216: late RGB processing is not a new Depth/identity deadline."""
from dataclasses import replace
import queue
import threading

import pytest

import request_0513_modular as runtime
from car_control_modular.control_types import HazardState
from test_depth_authority_250 import advance
from test_depth_authority_300 import authority300, seed300
from test_distance_tracking_response import setup
from test_lateral_zero_runtime import owner
from test_visual_depth_publication import full_result


@pytest.fixture
def late(authority300, monkeypatch):
    a, obj = authority300, authority300.owner
    monkeypatch.setattr(runtime, "VISUAL_DEPTH_VISIBILITY_MAX_AGE_SEC", .5)
    a.stamp, a.original = seed300(a)
    a.data, a.records = full_result(obj, cap=213, stamp=a.stamp-.10, now=a.stamp)
    a.visual = obj._validated_visual_observation
    a.deadline = obj._depth30_linear_timing.depth_expires_at
    obj._longitudinal_context_lock = threading.Lock()
    obj._longitudinal_context = {"sentinel": "original ROI"}
    a.context = obj._longitudinal_context
    obj._get_obstacle_status = lambda: dict(front=False, left=False, right=False)
    obj._current_hazard_state_for_controller = lambda: HazardState()
    obj.action_queue = queue.Queue()
    obj.search_direction = None
    obj.lost_confirm_frames = 0
    obj._last_dispatched_action = runtime.ACTION_FORWARD
    obj._rknn_pipeline.last_identity_processing.update(
        capture_frame_id=216, capture_timestamp=a.stamp+.01)
    obj._active_capture_frame_id, obj._active_capture_timestamp = 216, a.stamp+.01
    advance(a, a.stamp+.271)
    return a


def late_update(a):
    return a.owner._update_detector_identity_lease(a.records, 216, a.stamp+.01,
                                                 now=a.clock.now, stale=True)


def handle(a):
    return a.owner._handle_stale_vision_result(width=640, result_age_sec=.241)


def test_late_full_result_retains_only_original_depth_until_300ms_real_writer(late):
    a, obj = late, late.owner
    assert not late_update(a)  # never accepted as a fresh visual result
    assert obj._validated_visual_observation is a.visual
    assert handle(a)
    assert obj._longitudinal_context is a.context
    assert obj._validated_visual_observation is a.visual
    assert obj._depth30_linear_timing.depth_expires_at == a.deadline
    assert obj.search_state == "none" and not obj._queued_calls
    a.action._service_follow_wheels()
    assert a.backend.pairs[-1][0] > 0 and a.backend.pairs[-1][1] < 0
    assert obj._depth30_linear_snapshot[3] == a.stamp
    advance(a, a.stamp+.301)
    a.action._service_follow_wheels()
    assert a.backend.pairs[-1][:2] == (0, 0)
    assert obj._depth30_linear_timing.depth_expires_at == a.deadline


@pytest.mark.parametrize("fault", ["rejected", "pending", "uid0", "quality", "track",
                                   "ambiguous", "predicted", "features", "too_old", "replay"])
def test_identity_or_capture_failures_do_not_use_processing_only_exception(late, fault):
    a, obj = late, late.owner
    if fault == "rejected": a.data["identity_control_rejected"] = True
    if fault == "pending": a.data["identity_recheck_pending"] = True
    if fault == "uid0": a.records[0].reid_uid = 0
    if fault == "quality": a.data["bbox_quality_ok"] = False
    if fault == "track": a.records[0].track_id = 2
    if fault == "ambiguous": a.records.append(a.records[0])
    if fault == "predicted": a.records[0].time_since_update = 1
    if fault == "features": obj._rknn_pipeline.last_identity_processing["full_features_current"] = False
    if fault == "too_old": advance(a, a.stamp+.4)
    if fault == "replay": obj._identity_processing_watermark = (216, a.stamp+.01)
    late_update(a)
    assert obj._late_visual_preserved_evidence is None
    handle(a)
    assert obj._depth30_linear_snapshot is None
    a.action._service_follow_wheels()
    assert not a.backend.pairs or a.backend.pairs[-1][:2] == (0, 0)


@pytest.mark.parametrize("fault", ["expired", "visual_expired", "feedback_missing", "stop",
                                   "brake", "uid", "hazard", "obstacle", "revoked", "shutdown"])
def test_live_gates_rechecked_after_late_classification(late, fault):
    a, obj = late, late.owner
    assert not late_update(a)
    assert obj._late_visual_preserved_evidence is not None
    if fault == "expired": advance(a, a.stamp+.301)
    if fault == "visual_expired":
        obj._validated_visual_observation = replace(a.visual, expires_at=a.clock.now)
    if fault == "feedback_missing": a.feedback = None
    if fault == "stop": obj._explicit_stop_requested = True
    if fault == "brake": obj._brake_hold_active = True
    if fault == "uid": a.controller.active_target_id = 2
    if fault == "hazard": obj._current_hazard_state_for_controller = lambda: HazardState(active=True)
    if fault == "obstacle": obj._get_obstacle_status = lambda: dict(front=True)
    if fault == "revoked": obj._revoke_depth_linear_authority("identity_rejected")
    if fault == "shutdown": obj._runtime_shutdown_requested = True
    handle(a)
    assert obj._depth30_linear_snapshot is None
    assert obj._late_visual_preserved_evidence is None
    if fault == "stop": assert obj._explicit_stop_requested
