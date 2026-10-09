"""Startup soft STOP through real visual admission, PI and fake wheel I/O."""
from dataclasses import replace

import pytest

import request_0513_modular as runtime
from car_control_modular.action_command import ActionCommandSnapshot
from test_depth_authority_250 import authority, advance, writer
from test_distance_pi_controller import configured
from test_distance_tracking_response import setup
from test_lateral_zero_runtime import owner
from test_live_authority_binding import bind_production_reader


@pytest.fixture(params=['queue_empty', 'dispatch_context'])
def startup(authority, setup, request):
    a = authority
    _, a.controller, _ = configured(
        setup, target_distance_m=1.4, forward_start_distance_m=1.48,
        forward_stop_distance_m=1.43, near_distance_rotate_only_distance_m=1.43,
        distance_pi_kp_per_sec=3., distance_pi_launch_request_rpm=180.,
        distance_pi_launch_full_error_m=.5, depth_longitudinal_sample_max_age_sec=.25,
        visible_steering_pid_enable=True, visible_steering_pid_image_error_only=True,
        visible_steering_pid_image_brake_assist=True,
        center_left_ratio=.45, center_right_ratio=.55,
        steer_release_left_ratio=.47, steer_release_right_ratio=.53,
    )
    a.owner._follow_controller = a.controller
    bind_production_reader(a.owner)
    a.action, a.backend = writer(a)
    a.action.symbols = runtime.ACTION_RUNTIME_SYMBOLS
    a.action.config = replace(runtime.ACTION_RUNTIME_CONFIG, follow_wheel_period_sec=.05,
                              motor_forward_max_target_rpm=200, rotation_only=False,
                              rotate_pulse_brake_enable=False, use_percent_speed=True,
                              motor_steer_raw_target=15)
    a.action.get_steering_feedback = lambda: a.feedback
    a.backend.stops = []
    a.backend.send_stop = lambda label, **kwargs: a.backend.stops.append((label, kwargs))
    a.owner._action_command_revision = 1
    a.owner.current_command = runtime.ACTION_STOP
    a.owner._last_action_queue_reason = 'wait_first_person'
    a.owner._last_control_decision_reason = 'wait_first_person'
    a.owner._current_forward_percent = a.owner._current_steer_base_percent = 0
    a.owner._current_steer_correction_rpm = a.owner._current_rotate_raw_target = 0
    a.owner._last_vision_correction_rpm = a.owner._lateral_intent_last_correction_rpm = 0
    a.owner._use_soft_stop_next = True
    a.owner._soft_stop_active = False
    a.command = ActionCommandSnapshot(
        action=runtime.ACTION_STOP, revision=1, enqueued_at=a.clock.now,
        control_frame=3, capture_frame_id=24, capture_timestamp=a.clock.now-.04,
        reason='wait_first_person', soft_stop=True, protected_stop=False,
        source_module='vision', uid=None,
    )
    # This is the executor boundary after queue adoption, before the first
    # send_robot_command(STOP). It reproduces CAP24's interrupted startup zero.
    a.action._current_action_snapshot = a.command
    a.action._dispatch_context.command = (a.command if request.param == 'dispatch_context' else None)
    assert not a.owner._soft_stop_active
    assert a.owner._use_soft_stop_next
    a.action.send_stop_with_brake_hold('stop_signal')
    return a


def visual_sample(a, cap, *, case=None):
    distance = 1.40 if case == 'near_distance' else 1.619
    stamp = a.clock.now - .251 if case == 'expired_depth' else a.clock.now
    current = a.frame(distance, rpm=0., stamp=stamp, capture_frame_id=cap,
                      capture_timestamp=a.clock.now-.04)
    target = current.persons[0]
    if case == 'unconfirmed':
        a.controller.active_target_id = None
        a.controller._has_seen_person = False
        a.owner._vision_control_state = 'initial_wait'
        current = replace(current, persons=[])
        target = None
    elif case == 'hazard':
        current = replace(current, hazard=replace(current.hazard, active=True, reason='bunker'))
    elif case == 'obstacle':
        current = replace(current, obstacles=replace(current.obstacles, front=True))
    a.feedback = current.steering_feedback
    a.owner.frame_index = cap
    a.owner._active_capture_frame_id = a.owner._last_command_capture_frame = cap
    a.owner._last_command_capture_timestamp = current.capture_timestamp
    decision = a.controller.decide(cap, current, longitudinal_only=False)
    admitted = a.owner._refresh_visual_depth_linear_authority(
        current, target, decision, is_fresh_depth=True,
        target_steerable=target is not None, low_quality_visible=False,
    )
    a.owner._last_control_decision_reason = decision.reason
    if not decision.explicit_stop_requested and target is not None:
        a.owner._publish_lateral_intent_from_decision(
            width=current.width, target=target, runtime_actions=decision.actions,
            control_source='vision', target_steerable=True, low_quality_visible=False,
            near_yaw_park_requested=decision.near_yaw_park_requested,
        )
    a.action._service_follow_wheels()
    return current, decision, admitted


def test_adopted_startup_soft_stop_resumes_from_centered_new_depth_without_turn(startup):
    a = startup
    initial = a.clock.now
    grants = []
    for index in range(4):
        advance(a, initial + index * .05)
        current, decision, admitted = visual_sample(a, 26 + index)
        assert decision.reason == 'visual_pid_center_hold'
        assert not any(action.steer_correction_rpm for action in decision.actions)
        grants.append(a.owner._fresh_depth_linear_snapshot(1))
        if grants[-1] is not None:
            assert admitted
            assert grants[-1][3] == current.distance_state.sample_timestamp
    # First PI sample starts at zero. Later physical samples must create a
    # live grant without needing an off-center turn to release a false park.
    assert grants[0] is None
    assert any(grant is not None and grant[1] > 0 for grant in grants[1:])
    assert not a.owner._brake_hold_active
    assert not a.backend.stops
    assert a.backend.pairs[-1][0] > 0
    assert a.backend.pairs[-1][0] == -a.backend.pairs[-1][1]
    assert all(left == -right for left, right, _ in a.backend.pairs)
    assert all(action not in (runtime.ACTION_ROTATE_LEFT, runtime.ACTION_ROTATE_RIGHT)
               for actions, _reason in a.owner._queued_calls for action in actions)
    deadline = a.owner._depth30_linear_timing.depth_expires_at
    assert deadline == pytest.approx(grants[-1][3] + .25)
    advance(a, deadline + .001)
    a.action._service_follow_wheels()
    assert a.owner._fresh_depth_linear_snapshot(1) is None
    assert a.backend.pairs[-1][:2] == (0, 0)
    assert a.owner._depth30_linear_timing.depth_expires_at == deadline


@pytest.mark.parametrize('case', ['near_distance', 'unconfirmed', 'hazard', 'obstacle', 'expired_depth'])
def test_startup_soft_zero_never_grants_forward_for_invalid_evidence(startup, case):
    a = startup
    initial = a.clock.now
    for index in range(4):
        advance(a, initial + index * .05)
        _current, _decision, _admitted = visual_sample(a, 26 + index, case=case)
        assert a.owner._fresh_depth_linear_snapshot(1) is None
    assert all(left == right == 0 for left, right, _ in a.backend.pairs)
