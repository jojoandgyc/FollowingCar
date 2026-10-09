"""CAP344: a fresh distance-only request can exit ordinary dwell, not STOP safety.

Use the real controller, release producer and executor's current/FREE service.
Only the backend is fake; preview release never sends a positive wheel command.
"""
from dataclasses import replace

import pytest

from car_control_modular.near_yaw_parking import NearYawParkRequest, ParkSettlingEvidence
from test_cap825_park_resume_runtime import parked, preview, release
from test_depth_authority_250 import authority
from test_distance_tracking_response import setup
from test_lateral_zero_runtime import owner


@pytest.fixture
def cap344(parked):
    a = parked
    a.controller.cfg = replace(a.controller.cfg, distance_target_motion_control_enable=False)
    request = NearYawParkRequest(1, 337, a.clock.now-.2, a.clock.now-.19,
                                 "low_speed_turn_stop:visual_pid_direction_guard_hold")
    a.owner._near_yaw_park_request = request
    a.owner._near_yaw_park_evidence = (337, request.capture_timestamp)
    a.action._near_yaw_park_applied = request
    a.evidence = ParkSettlingEvidence(request, a.clock.now-.1, require_current_release=True)
    a.action._near_yaw_park_settling = a.evidence
    a.owner._brake_hold_stop_mode = a.action._ordinary_park_stop_mode()
    a.io = []
    a.before_free = None
    a.fail_current = a.fail_free = False

    def release_current():
        if a.fail_current:
            raise OSError("one-side current readback failed")
        a.io.append(("current_zero_ack", (0., 0.)))

    def stop(label, *, mode="emergency", **kwargs):
        if label == "ordinary_park_release_free":
            if a.before_free is not None:
                a.before_free()
            if a.fail_free:
                raise OSError("one-side FREE write failed")
        a.io.append(("stop_ack", label, mode))

    a.backend.release_parking_current_only = release_current
    a.backend.send_stop = stop
    return a


def service(a):
    with a.owner.motor_io_lock:
        a.action._service_ordinary_park_exit(a.evidence, "near_yaw_park")


@pytest.mark.parametrize("distance", [1.58, 1.62, 1.95])
def test_new_pure_distance_preview_releases_dwell_only_after_current_and_free_ack(cap344, distance):
    a = cap344
    current, decision = preview(a, distance=distance, cap=344)
    assert a.controller._braking_range_rate is None
    assert a.controller._braking_rate_source != "raw_depth_window"
    assert a.clock.now-a.evidence.sent_at < .5
    assert a.controller.parked_forward_resume_demand_rpm(current, 1) > 0
    assert not release(a, current, decision)  # Cache hint only; current is still held.
    deadline = a.evidence.forward_resume_until
    assert deadline == min(current.capture_timestamp+.19,
                           current.distance_state.sample_timestamp+.18)
    assert a.evidence.current_released_at is None
    assert not a.backend.pairs and not a.io

    def before_free():
        assert a.io == [("current_zero_ack", (0., 0.))]
        assert a.evidence.current_released_at is None
        # Executor owns the non-reentrant serial lock here. Check its cached
        # release gate, not a nested producer call that would reacquire it.
        assert not a.evidence.motion_handoff_ready(
            current.capture_timestamp, a.feedback, a.clock.now)
        assert not a.backend.pairs

    a.before_free = before_free
    service(a)
    assert a.io == [("current_zero_ack", (0., 0.)),
                    ("stop_ack", "ordinary_park_release_free", "free")]
    assert a.evidence.current_released_at == a.clock.now
    assert release(a, current, decision)
    assert a.owner._near_yaw_park_request is None
    assert a.owner._depth30_linear_snapshot is None
    assert a.controller.last_distance_pid_result.output_rpm == 0
    assert not a.backend.pairs  # The new distance must be admitted separately.
    a.owner._refresh_visual_depth_linear_authority(
        current, current.persons[0], decision, is_fresh_depth=True,
        target_steerable=True, low_quality_visible=False)
    assert a.owner._fresh_depth_linear_snapshot(1, quiet=True) is None
    assert not a.backend.pairs


@pytest.mark.parametrize("fault", [
    "pre_stop_image", "pre_stop_depth", "old_depth", "reused_depth", "wrong_uid",
    "hazard", "obstacle", "brake_latched", "safety_distance", "other_brake_owner",
    "identity_unqualified", "explicit_stop", "shutdown", "feedback_stale",
])
def test_unqualified_observation_never_releases_ordinary_dwell(cap344, fault):
    a = cap344
    current, decision = preview(a, distance=1.58, cap=344)
    changes = {}
    if fault == "pre_stop_image": current = replace(current, capture_timestamp=a.evidence.sent_at-.001)
    elif fault == "pre_stop_depth":
        current = replace(current, distance_state=replace(current.distance_state,
                           sample_timestamp=a.evidence.sent_at-.001))
    elif fault == "old_depth": a.clock.now += .181
    elif fault == "reused_depth":
        current = replace(current, distance_state=replace(current.distance_state,
                           source_detail="depth_multiregion_reused_hold"))
    elif fault == "wrong_uid": current = replace(current, persons=[replace(current.persons[0], track_id=2)])
    elif fault == "hazard": current = replace(current, hazard=replace(current.hazard, active=True))
    elif fault == "obstacle": current = replace(current, obstacles=replace(current.obstacles, front=True))
    elif fault == "brake_latched": current = replace(current, distance_state=replace(current.distance_state, brake_latched=True))
    elif fault == "safety_distance": current = replace(current, distance_state=replace(current.distance_state, safety_distance_m=.3))
    elif fault == "other_brake_owner": a.owner._brake_hold_label = "danger"
    elif fault == "identity_unqualified": changes["target_steerable"] = False
    elif fault == "explicit_stop": a.owner._explicit_stop_requested = True
    elif fault == "shutdown": a.owner._runtime_shutdown_requested = True
    elif fault == "feedback_stale": a.feedback = replace(a.feedback, timestamp=a.clock.now-.151)
    assert not release(a, current, decision, **changes)
    assert a.owner._near_yaw_park_request is a.evidence.request
    assert a.owner._depth30_linear_snapshot is None
    assert not a.backend.pairs
    if fault != "feedback_stale":
        assert a.evidence.forward_resume_until == 0


@pytest.mark.parametrize("fault", ["current", "free"])
def test_partial_current_or_free_failure_never_releases_park(cap344, fault):
    a = cap344
    current, decision = preview(a, distance=1.58, cap=344)
    assert not release(a, current, decision)
    a.fail_current, a.fail_free = fault == "current", fault == "free"
    service(a)
    assert a.evidence.fault
    assert a.evidence.current_released_at is None
    assert not release(a, current, decision)
    assert a.owner._near_yaw_park_request is a.evidence.request
    assert a.owner._depth30_linear_snapshot is None
    assert not a.backend.pairs


def test_replaying_same_depth_does_not_extend_resume_hint(cap344):
    a = cap344
    current, decision = preview(a, distance=1.58, cap=344)
    assert not release(a, current, decision)
    deadline = a.evidence.forward_resume_until
    a.clock.now += .04
    assert not release(a, current, decision)
    assert a.evidence.forward_resume_until == deadline
    a.clock.now = deadline+.001
    assert not release(a, current, decision)
    assert not a.evidence.forward_resume_live(a.clock.now)
    service(a)
    assert a.evidence.current_released_at is None
    assert not a.backend.pairs and not a.io
