"""Deferred visual centering cannot promote a depth gap into chassis parking."""
from dataclasses import replace

import pytest

import request_0513_modular as runtime
from car_control_modular.control_types import ControlAction, HazardState
from car_control_modular.predictive_turn_brake import qualified_countersteer
from test_async_visual_authority import visual_authority, new_visual
from test_depth_authority_250 import authority, advance, decide_commit
from test_distance_tracking_response import setup
from test_lateral_zero_runtime import owner


def _center_with_expired_depth_and_residual_yaw(a, left=-3., right=3.):
    advance(a, a.stamp+.251)
    a.owner._depth_async_scheduler.worker_tick(now=a.clock.now)
    a.feedback = replace(a.feedback, left_forward_rpm=left, right_forward_rpm=right)
    persons = new_visual(a, 810, 240.)
    assert a.owner._fresh_depth_linear_snapshot(1) is None
    assert a.owner._queue_actions_for_persons(640, 480, persons)
    intent = a.owner._lateral_intent_store.snapshot()
    assert intent is not None and intent.distance_control_deferred
    assert intent.initial_correction_rpm == 0 and not intent.park_requested
    assert getattr(a.owner, "_near_yaw_park_request", None) is None
    assert getattr(a.controller, "_normal_parking_uid", None) is None
    return intent


@pytest.mark.parametrize("wheels", [(-3., 3.), (3., -3.), (0., 0.)])
def test_center_during_depth_gap_does_not_latch_park_and_fresh_depth_can_resume(visual_authority, wheels):
    a = visual_authority
    intent = _center_with_expired_depth_and_residual_yaw(a, *wheels)
    # The independent real depth deadline still stops the fake writer. A zero
    # for expired evidence is not permitted to turn into persistent parking.
    a.action._service_follow_wheels()
    assert a.backend.pairs[-1][:2] == (0, 0)
    a.owner._service_lateral_intent(a.clock.now)
    assert getattr(a.owner, "_near_yaw_park_request", None) is None
    assert getattr(a.controller, "_normal_parking_uid", None) is None

    positive = False
    for index in range(1, 4):
        advance(a, a.stamp+.251+index*.05)
        measured = replace(a.frame(3., rpm=3.), persons=[a.target],
                           capture_frame_id=intent.capture_frame_id,
                           capture_timestamp=intent.capture_timestamp)
        _, actions, accepted = decide_commit(a, measured)
        assert getattr(a.owner, "_near_yaw_park_request", None) is None
        assert getattr(a.controller, "_normal_parking_uid", None) is None
        if accepted and any(action.speed_percent > 0 for action in actions):
            a.action._service_follow_wheels()
            positive = a.backend.pairs[-1][0] > 0 and a.backend.pairs[-1][1] < 0
            if positive:
                break
    assert positive, "fresh independent measurements must not be locked by visual centering"


@pytest.mark.parametrize("automatic_request", [False, True])
def test_deferred_center_cannot_promote_tick_park_request(visual_authority, automatic_request):
    a = visual_authority
    intent = _center_with_expired_depth_and_residual_yaw(a)
    a.owner._publish_lateral_zero(intent, "center_hold", park_requested=automatic_request,
                                  pid_result=a.controller.last_steering_pid_result)
    assert getattr(a.owner, "_near_yaw_park_request", None) is None
    assert getattr(a.controller, "_normal_parking_uid", None) is None


def test_dedicated_near_distance_stop_still_requests_normal_parking(visual_authority):
    a = visual_authority
    _center_with_expired_depth_and_residual_yaw(a)
    a.owner._last_control_decision_reason = "near_distance_rotation_only"
    a.owner._last_command_capture_frame = 811
    a.owner._last_command_capture_timestamp = a.clock.now
    assert a.owner._publish_lateral_intent_from_decision(
        width=640, target=a.target,
        runtime_actions=[ControlAction.stop("near_distance_rotation_only")],
        control_source="vision", target_steerable=True, low_quality_visible=False,
        near_yaw_park_requested=True,
    )
    intent = a.owner._lateral_intent_store.snapshot()
    assert intent.hold_zero and intent.park_requested and not intent.distance_control_deferred
    assert a.owner._near_yaw_park_request.uid == 1
    assert a.controller._normal_parking_uid == 1


@pytest.mark.parametrize("event", ["hazard", "front", "left", "right"])
def test_real_safety_on_requested_deferred_frame_still_revokes_and_stops(visual_authority, event):
    a = visual_authority
    advance(a, a.stamp+.02)
    a.action._service_follow_wheels()
    assert a.backend.pairs[-1][0] > 0 and a.backend.pairs[-1][1] < 0
    advance(a, a.stamp+.04)
    persons = new_visual(a, 812, 240.)
    cleared = []
    a.owner._clear_action_queue = lambda reason: cleared.append(reason)
    if event == "hazard":
        a.owner._current_hazard_state_for_controller = lambda: HazardState(active=True, reason="danger")
    else:
        a.owner._get_obstacle_status = lambda: {event: True}
    a.owner._process_detections_modular(640, 480, persons,
                                      visual_lateral_only=True, range_deferred=True)
    assert a.owner._explicit_stop_requested
    assert a.owner._depth30_linear_snapshot is None
    assert cleared
    a.action._service_follow_wheels()
    assert a.backend.pairs[-1][:2] == (0, 0)


@pytest.mark.parametrize("deferred", [True, False])
def test_predictive_turn_parking_respects_independent_visual_owner(visual_authority, deferred):
    a = visual_authority
    intent = _center_with_expired_depth_and_residual_yaw(a)
    a.controller.cfg = replace(a.controller.cfg,
        visible_steering_pid_execution_response_trial_sec=0.,
        visible_steering_pid_predictive_countersteer_max_correction_rpm=6.)
    result = replace(a.controller.last_steering_pid_result,
        output_floor_reason="predictive_countersteer", predictive_braking=True,
        feedback_used=True, target_rate_valid=True, feedback_age_sec=.01,
        correction_rpm=-3, visual_error_deg=6., measured_yaw_rate_dps=10.,
        target_image_rate_dps=-1.)
    intent = replace(intent, x_ratio=.6, mode="forward" if deferred else "yaw_only",
                     distance_control_deferred=deferred)
    assert qualified_countersteer(result, intent.x_ratio, 6.)
    parked = a.owner._request_predictive_turn_brake(intent, result)
    assert parked is (not deferred)
    assert (getattr(a.owner, "_near_yaw_park_request", None) is not None) is (not deferred)
