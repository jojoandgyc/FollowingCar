"""Centered CAP825--957: real PI, release, admission and writer; no hardware."""
from dataclasses import replace

import pytest

import request_0513_modular as runtime
from car_control_modular.control_types import ControlAction, ControlDecision
from car_control_modular.near_yaw_parking import NearYawParkRequest, ParkSettlingEvidence
from test_depth_authority_250 import authority, advance, decide_commit, writer
from test_distance_pi_controller import configured
from test_distance_tracking_response import setup
from test_lateral_zero_runtime import owner
from test_live_authority_binding import bind_production_reader


@pytest.fixture
def parked(authority, setup):
    a = authority
    _, a.controller, _ = configured(
        setup, target_distance_m=1.4, distance_pi_kp_per_sec=3.,
        distance_pi_launch_request_rpm=180., distance_pi_launch_full_error_m=.5,
        distance_pid_output_rise_rpm_per_sec=240.,
        depth_longitudinal_sample_max_age_sec=.25,
        visible_steering_pid_enable=True,
        visible_steering_pid_image_error_only=True,
        visible_steering_pid_image_brake_assist=True,
        center_left_ratio=.45, center_right_ratio=.55,
        steer_release_left_ratio=.47, steer_release_right_ratio=.53,
    )
    a.owner._follow_controller = a.controller
    bind_production_reader(a.owner)
    a.action, a.backend = writer(a)
    a.action.get_steering_feedback = lambda: a.feedback
    request = NearYawParkRequest(1, 803, a.clock.now-1., a.clock.now-.99, 'normal')
    a.owner._near_yaw_park_request = request
    a.owner._near_yaw_park_evidence = (803, request.capture_timestamp)
    a.owner._brake_hold_active = True
    a.owner._brake_hold_label = 'near_yaw_park'
    a.action._near_yaw_park_applied = request
    a.evidence = ParkSettlingEvidence(request, a.clock.now-.98, require_current_release=True)
    a.evidence.mark_current_released(a.clock.now-.4)
    a.action._near_yaw_park_settling = a.evidence
    a.controller.set_normal_parking(True, 1)
    return a


def preview(a, distance=1.8, cap=825):
    current = a.frame(distance, rpm=0., capture_timestamp=a.clock.now-.04,
                      capture_frame_id=cap)
    a.feedback = current.steering_feedback
    a.owner._last_command_capture_frame = cap
    a.owner._last_command_capture_timestamp = current.capture_timestamp
    a.owner._active_capture_frame_id = cap
    decision = a.controller.decide(cap, current, longitudinal_only=False)
    assert decision.reason == 'visual_pid_center_hold'
    assert a.controller.last_distance_pid_result.pi_status == 'parked_preview'
    assert a.controller.last_distance_pid_result.output_rpm == 0
    assert not any(x.speed_percent > 0 or x.steer_correction_rpm for x in decision.actions)
    return current, decision


def release(a, current, decision, **changes):
    return a.owner._release_near_yaw_park_for_decision(
        current, current.persons[0] if current.persons else None, decision,
        **dict(dict(control_source='vision', target_steerable=True, low_quality_visible=False), **changes))


@pytest.mark.parametrize('distance', [1.541, 1.885, 2.186])
def test_centered_fresh_demand_releases_park_without_executing_preview(parked, distance):
    a = parked
    current, decision = preview(a, distance)
    result = a.controller.last_distance_pid_result
    assert 2 <= a.controller.parked_forward_resume_demand_rpm(current, 1) <= result.approach_cap_rpm
    before = a.controller._distance_pid._distance_pi.integral_m_s
    assert release(a, current, decision)
    assert a.owner._near_yaw_park_request is None
    assert not a.owner._brake_hold_active
    assert not a.controller._distance_pid._distance_pi._normal_parking
    assert a.owner._depth30_linear_snapshot is None
    assert not a.backend.pairs  # Releasing state never writes motor commands.
    assert a.controller.last_distance_pid_result is result
    assert a.controller._distance_pid._distance_pi.integral_m_s == before

    # Match the actual visual path: it attempts Depth admission immediately
    # after release. The parked preview is still zero, NOT a cached grant.
    a.owner._refresh_visual_depth_linear_authority(
        current, current.persons[0], decision, is_fresh_depth=True,
        target_steerable=True, low_quality_visible=False)
    assert a.owner._fresh_depth_linear_snapshot(1) is None
    assert a.controller._distance_pid._distance_pi._execution_suspended

    # Include the actual center-hold publisher after visual admission. It
    # must not immediately request another park while both wheels are zero.
    a.owner._last_control_decision_reason = decision.reason
    assert a.owner._publish_lateral_intent_from_decision(
        width=current.width, target=current.persons[0], runtime_actions=decision.actions,
        control_source='vision', target_steerable=True, low_quality_visible=False,
        near_yaw_park_requested=decision.near_yaw_park_requested)
    assert a.owner._near_yaw_park_request is None
    assert not a.controller._distance_pid._distance_pi._normal_parking

    approved = []
    for n in range(1, 4):
        advance(a, current.distance_state.sample_timestamp+n*.05)
        _, actions, _ = decide_commit(a, a.frame(distance, rpm=0.))
        approved.append(max([x.speed_percent*2 for x in actions if x.kind == 'forward'] or [0]))
    assert approved[0] == 0 < approved[1] <= approved[2] <= 24
    assert not a.controller.last_distance_pid_result.pi_software_rise_bypassed
    a.action._service_follow_wheels()
    assert a.backend.pairs[-1][:2] == (approved[-1], -approved[-1])


@pytest.mark.parametrize('case', [
    'old_depth', 'future_depth', 'different_depth', 'reused', 'jump', 'raw_near', 'used_near',
    'target_latched', 'brake_latched', 'safety_distance', 'hazard', 'obstacle', 'lost',
    'wrong_uid', 'quality', 'not_steerable', 'search', 'explicit_stop', 'shutdown',
    'new_park', 'rotation_only', 'depth_loop', 'old_capture', 'stop_not_applied',
    'current_not_released', 'feedback_stale', 'other_brake_owner',
    'no_preview', 'no_cap', 'nan_demand', 'zero_cap', 'sub_quantum',
])
def test_preview_demand_does_not_bypass_safety_or_provenance(parked, monkeypatch, case):
    a = parked
    current, decision = preview(a)
    kwargs = {}
    if case == 'old_depth': a.clock.now += .181
    elif case == 'future_depth': a.clock.now -= .01
    elif case == 'different_depth': current = replace(current, distance_state=replace(current.distance_state, sample_timestamp=a.clock.now-.01))
    elif case == 'reused': current = replace(current, distance_state=replace(current.distance_state, source_detail='depth_multiregion_reused_hold'))
    elif case == 'jump': current = replace(current, distance_state=replace(current.distance_state, source_detail='depth_distance_jump_pending'))
    elif case == 'raw_near': current = replace(current, distance_state=replace(current.distance_state, raw_distance_m=1.3))
    elif case == 'used_near': current = replace(current, distance_m=1.3)
    elif case in ('target_latched', 'brake_latched'): current = replace(current, distance_state=replace(current.distance_state, **{case: True}))
    elif case == 'safety_distance': current = replace(current, distance_state=replace(current.distance_state, safety_distance_m=.3))
    elif case == 'hazard': current = replace(current, hazard=replace(current.hazard, active=True))
    elif case == 'obstacle': current = replace(current, obstacles=replace(current.obstacles, front=True))
    elif case == 'lost': current = replace(current, persons=[])
    elif case == 'wrong_uid': current = replace(current, persons=[replace(current.persons[0], track_id=2)])
    elif case == 'quality': kwargs['low_quality_visible'] = True
    elif case == 'not_steerable': kwargs['target_steerable'] = False
    elif case == 'search': a.owner.search_state = 'searching'
    elif case == 'explicit_stop': decision = replace(decision, explicit_stop_requested=True)
    elif case == 'shutdown': decision = replace(decision, shutdown_requested=True)
    elif case == 'new_park': decision = replace(decision, near_yaw_park_requested=True)
    elif case == 'rotation_only': monkeypatch.setattr(runtime, 'FOLLOW_ROTATION_ONLY', True)
    elif case == 'depth_loop': kwargs['control_source'] = 'depth30'
    elif case == 'old_capture': current = replace(current, capture_timestamp=a.clock.now-.2)
    elif case == 'stop_not_applied': a.action._near_yaw_park_applied = None
    elif case == 'current_not_released': a.evidence.current_released_at = None
    elif case == 'feedback_stale': a.feedback = replace(a.feedback, timestamp=a.clock.now-.2)
    elif case == 'other_brake_owner': a.owner._brake_hold_label = 'danger'
    else:
        field, value = {'no_preview': ('pi_status', 'tracking'), 'no_cap': ('approach_cap_rpm', None),
                        'nan_demand': ('unslewed_output_rpm', float('nan')), 'zero_cap': ('approach_cap_rpm', 0.),
                        'sub_quantum': ('unslewed_output_rpm', 1.99)}[case]
        a.controller.last_distance_pid_result = replace(a.controller.last_distance_pid_result, **{field: value})
    assert not release(a, current, decision, **kwargs)
    assert a.owner._near_yaw_park_request is not None
    assert a.owner._depth30_linear_snapshot is None
    assert not a.backend.pairs


def test_non_outward_preview_cannot_skip_normal_dwell(parked):
    a = parked
    current, decision = preview(a)
    a.evidence.sent_at = a.clock.now-.1
    a.controller._braking_range_rate = 0.
    assert not release(a, current, decision)
    assert a.evidence.forward_resume_until == 0
    assert a.evidence.reason == 'minimum_normal_hold_500ms'


def test_original_positive_forward_release_path_unchanged(parked):
    a = parked
    current, _ = preview(a)
    decision = ControlDecision(actions=[ControlAction.forward(10, 'existing')], reason='existing')
    assert release(a, current, decision)
    assert a.owner._depth30_linear_snapshot is None


def test_cached_preview_can_never_renew_release_hint_deadline(parked):
    a = parked
    current, decision = preview(a)
    a.evidence.current_released_at = None
    a.controller._braking_rate_source = 'raw_depth_window'
    a.controller._braking_range_rate = .1
    assert not release(a, current, decision)
    deadline = a.evidence.forward_resume_until
    a.clock.now += .04
    assert not release(a, current, decision)
    assert a.evidence.forward_resume_until == deadline
    a.clock.now += .15
    assert not release(a, current, decision)
    assert not a.evidence.forward_resume_live(a.clock.now)
