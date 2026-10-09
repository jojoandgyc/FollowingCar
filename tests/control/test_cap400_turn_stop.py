"""CAP400: predictive yaw zero must finish an actual pivot, not coast forward.

Real producer methods, cached synthetic encoder feedback; no hardware startup.
"""
from types import SimpleNamespace

import pytest

import request_0513_modular as runtime
from car_control_modular.control_types import ControlAction
from car_control_modular.near_yaw_parking import low_speed_turn_stop_reason
from test_lateral_zero_runtime import owner, _intent, _target, NOW
from test_near_yaw_park_runtime import publish_motion, settled_park


def feedback(left=-8., right=8., **kw):
    fields = dict(timestamp=NOW-.04, trustworthy=True,
                  left_forward_rpm=left, right_forward_rpm=right)
    fields.update(kw)
    return SimpleNamespace(**fields)


def attach(owner, fb):
    owner._action_runtime = SimpleNamespace(get_steering_feedback=lambda: fb)


def zero(owner, **kw):
    fields = dict(near_distance_mode=False, park_requested=True, hold_zero=True)
    fields.update(kw)
    intent = _intent(owner, **fields)
    owner._publish_lateral_zero(intent, "predictive_brake_coast")
    return intent


@pytest.mark.parametrize("percent", [0, 2, 3, 5])
def test_pivot_predictive_stop_overrides_only_small_forward_grant(owner, monkeypatch, percent):
    monkeypatch.setattr(runtime, "FORWARD_MAX_RPM", 200)
    attach(owner, feedback())
    if percent:
        owner._depth30_linear_snapshot = ("forward", percent, 1, NOW-.05)
    zero(owner)
    assert owner._near_yaw_park_request.reason.startswith("low_speed_turn_stop:")
    assert owner._depth30_linear_snapshot is None
    assert owner._current_forward_percent == owner._current_steer_correction_rpm == 0
    assert not owner._use_soft_stop_next
    assert owner._queued_calls[-1][0] == (runtime.ACTION_STOP,)


@pytest.mark.parametrize("left,right,percent", [(60, 80, 35), (4, 8, 3),
                                                    (6, 6, 3), (-8, 8, 20)])
def test_normal_translation_is_not_whole_chassis_park(owner, monkeypatch, left, right, percent):
    monkeypatch.setattr(runtime, "FORWARD_MAX_RPM", 200)
    attach(owner, feedback(left, right))
    owner._depth30_linear_snapshot = ("forward", percent, 1, NOW-.05)
    zero(owner)
    assert getattr(owner, "_near_yaw_park_request", None) is None
    assert owner._current_forward_percent == percent
    assert owner._queued_calls[-1][0] == (runtime.ACTION_STEER_RIGHT,)


@pytest.mark.parametrize("fb", [None, feedback(timestamp=NOW-.16),
    feedback(timestamp=NOW+.01), feedback(trustworthy=False),
    feedback(left=float("nan")), feedback(right=float("inf")),
    feedback(0, 0), feedback(-40, 40), feedback(-30, -20)])
def test_unknown_stale_or_nonpivot_feedback_does_not_invent_park(owner, fb):
    attach(owner, fb)
    zero(owner)
    assert getattr(owner, "_near_yaw_park_request", None) is None
    assert owner._use_soft_stop_next


@pytest.mark.parametrize("changes", [dict(target_id=2), dict(bbox_quality="limited"),
    dict(capture_timestamp=NOW-.3), dict(capture_timestamp=NOW+.01),
    dict(capture_frame_id=0), dict(valid_until=NOW-.01), dict(park_requested=False)])
def test_unqualified_or_nonpredictive_zero_keeps_existing_semantics(owner, changes):
    attach(owner, feedback())
    zero(owner, **changes)
    assert getattr(owner, "_near_yaw_park_request", None) is None


@pytest.mark.parametrize("left,right", [(-8, 8), (8, -8), (-2, 10), (10, -2), (0, 8)])
def test_motion_qualification_is_symmetric(owner, left, right):
    intent = _intent(owner)
    assert low_speed_turn_stop_reason(intent, feedback(left, right), now=NOW,
        active_uid=1, linear_rpm=6, max_image_age=.19) == "low_speed_turn_stop"


@pytest.mark.parametrize("kind", ["stop", "forward"])
def test_main_loop_carries_predictive_stop_outside_near_mode(owner, kind):
    attach(owner, feedback())
    owner._last_control_decision_reason = "visual_pid_image_yaw_zero"
    result = owner._follow_controller.last_steering_pid_result
    result.correction_rpm = 0
    result.predictive_braking = True
    action = ControlAction.stop("predictive") if kind == "stop" else ControlAction.forward(3, "distance_pi")
    owner._depth30_linear_snapshot = ("forward", 3, 1, NOW-.05)
    assert owner._publish_lateral_intent_from_decision(width=640, target=_target(),
        runtime_actions=[action], control_source="vision", target_steerable=True,
        low_quality_visible=False)
    assert not owner._lateral_intent_store.snapshot().near_distance_mode
    assert owner._near_yaw_park_request is not None


def test_fast_loop_carries_new_prediction_outside_near_mode(owner):
    attach(owner, feedback())
    _intent(owner, near_distance_mode=False, initial_correction_rpm=0)
    result = owner._follow_controller.last_steering_pid_result
    result.correction_rpm = 0
    result.predictive_braking = True
    owner._service_lateral_intent(NOW)
    assert owner._near_yaw_park_request is not None


def test_latched_stop_survives_18_and_22_rpm_updates_and_old_motion(owner):
    attach(owner, feedback())
    zero(owner)
    request = owner._near_yaw_park_request
    for percent in (9, 11):
        owner._depth30_linear_snapshot = ("forward", percent, 1, NOW-.01)
        zero(owner, park_requested=False)
        assert owner._near_yaw_park_request is request
        assert owner._depth30_linear_snapshot is None
        assert not owner._use_soft_stop_next
    assert not publish_motion(owner, capture=576, stamp=NOW-.09)
    assert owner._near_yaw_park_request is request


def test_motion_qualified_park_uses_existing_post_write_quiet_release(owner, monkeypatch):
    attach(owner, feedback())
    zero(owner)
    # Merely newer visual output cannot bypass the real executor's quiet proof.
    assert not publish_motion(owner)
    settled_park(owner, monkeypatch)
    assert publish_motion(owner)
    assert owner._near_yaw_park_request is None
    assert owner._depth30_linear_snapshot is None
