"""A forward yaw stop is not a chassis parking request; no hardware startup."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

import request_0513_modular as runtime
from car_control_modular.control_types import SteeringFeedback
from test_cap400_turn_stop import attach, feedback
from test_depth_authority_250 import writer
from test_lateral_zero_runtime import NOW, _intent, owner


def forward_zero(o, percent=5, **changes):
    o._depth30_linear_snapshot = ("forward", percent, 1, NOW-.05)
    fields = dict(mode="forward", near_distance_mode=False, base_percent=percent,
                  base_rpm=percent*2, park_requested=True,
                  initial_correction_rpm=0, hold_zero=False)
    fields.update(changes)
    intent = _intent(o, **fields)
    o._publish_lateral_zero(intent, "center_hold")
    return intent


@pytest.mark.parametrize("percent", [1, 2, 3, 5, 6, 47])
@pytest.mark.parametrize("left,right", [(-8, 8), (8, -8), (0, 8), (10, -2)])
def test_ordinary_forward_yaw_stop_preserves_qualified_depth(owner, monkeypatch, percent, left, right):
    monkeypatch.setattr(runtime, "FORWARD_MAX_RPM", 200)
    attach(owner, feedback(left, right))
    intent = forward_zero(owner, percent)
    assert getattr(owner, "_near_yaw_park_request", None) is None
    assert owner._depth30_linear_snapshot == ("forward", percent, 1, NOW-.05)
    assert owner._current_forward_percent == percent
    assert owner._current_steer_correction_rpm == 0
    assert owner._lateral_intent_zero_sequence == intent.sequence
    assert owner._queued_calls[-1][0] == (runtime.ACTION_STEER_RIGHT,)


@pytest.mark.parametrize("changes", [dict(mode="yaw_only"), dict(near_distance_mode=True)])
def test_actual_pivot_and_explicit_near_parking_keep_original_policy(owner, changes):
    attach(owner, feedback())
    forward_zero(owner, **changes)
    assert owner._near_yaw_park_request is not None
    assert owner._depth30_linear_snapshot is None
    assert owner._queued_calls[-1][0] == (runtime.ACTION_STOP,)


@pytest.mark.parametrize("fault", ["expired", "uid", "visual", "safety", "park"])
def test_ordinary_yaw_stop_cannot_rescue_invalid_or_parked_motion(owner, fault):
    attach(owner, feedback())
    intent = forward_zero(owner)
    owner._queued_calls.clear()
    if fault == "expired": owner._depth30_linear_snapshot = ("forward", 5, 1, NOW-1.)
    elif fault == "uid": owner._follow_controller.active_target_id = 2
    elif fault == "visual": owner._vision_control_state = "target_lost"
    elif fault == "safety": owner._explicit_stop_requested = True
    else:
        owner._request_near_yaw_park(intent, "explicit_near_park")
    owner._publish_lateral_zero(intent, "second_yaw_stop")
    assert owner._fresh_depth_linear_snapshot(1) is None or fault == "park"
    assert not owner._queued_calls or owner._queued_calls[-1][0] == (runtime.ACTION_STOP,)


def test_producer_to_writer_keeps_actual_reversal_guard_then_resumes(owner, monkeypatch):
    """Retained forward intent is not permission to reverse a moving wheel."""
    clock = [NOW]
    monkeypatch.setattr(runtime.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(runtime, "FORWARD_MAX_RPM", 200)
    monkeypatch.setattr(runtime, "MOTOR_FORWARD_MAX_TARGET_RPM", 200)
    owner._follow_controller.search_state = "none"
    fb = [SteeringFeedback(timestamp=NOW-.01, trustworthy=True,
                           left_forward_rpm=-8., right_forward_rpm=8.)]
    a = SimpleNamespace(owner=owner, frame=lambda *a, **kw: SimpleNamespace(steering_feedback=fb[0]))
    action, backend = writer(a)
    action.get_steering_feedback = lambda: fb[0]
    forward_zero(owner)
    assert getattr(owner, "_near_yaw_park_request", None) is None
    action._service_follow_wheels()
    assert backend.pairs and backend.pairs[-1][:2] == (0, 0)
    assert owner._depth30_linear_snapshot == ("forward", 5, 1, NOW-.05)
    assert getattr(owner, "_near_yaw_park_request", None) is None
    for offset in (.06, .12):
        clock[0] = NOW+offset
        fb[0] = replace(fb[0], timestamp=clock[0], left_forward_rpm=0., right_forward_rpm=0.)
        action._service_follow_wheels()
    assert backend.pairs[-1][:2] == (10, -10)
    assert getattr(owner, "_near_yaw_park_request", None) is None
    # Neither yaw cancellation nor confirmed zero-crossing refreshes Depth.
    clock[0] = NOW+.30
    fb[0] = replace(fb[0], timestamp=clock[0])
    action._service_follow_wheels()
    assert backend.pairs[-1][:2] == (0, 0)
