"""A lateral correction cannot hide a fresh parked forward PI demand."""
from dataclasses import replace

import pytest

import request_0513_modular as runtime
from car_control_modular.control_types import ControlAction, ControlDecision
from test_cap825_park_resume_runtime import parked, preview, release, authority, setup, owner
from test_depth_authority_250 import advance, decide_commit


def side_preview(a, *, x=.375, cap=2627):
    start = a.clock.now
    a.evidence.sent_at = start-.20
    a.evidence.current_released_at = None
    for offset, distance in ((-.06, 1.60), (0., 1.63)):
        advance(a, start+offset)
        current = a.frame(distance, rpm=0., capture_timestamp=a.clock.now-.04,
                          capture_frame_id=cap if offset == 0 else cap-2)
        target = current.persons[0]
        shift = x*current.width-target.center[0]
        box = target.bbox
        current = replace(current, persons=[replace(target,
            bbox=(box[0]+shift, box[1], box[2]+shift, box[3]))])
        a.feedback = current.steering_feedback
        a.owner._last_command_capture_frame = current.capture_frame_id
        a.owner._last_command_capture_timestamp = current.capture_timestamp
        a.owner._active_capture_frame_id = current.capture_frame_id
        decision = a.controller.decide(current.capture_frame_id, current, longitudinal_only=False)
    assert a.controller.last_distance_pid_result.pi_status == 'parked_preview'
    assert a.controller.last_distance_pid_result.output_rpm == 0
    assert a.controller._braking_rate_source == 'raw_depth_window'
    assert a.controller._braking_range_rate > 0
    if not a.controller.cfg.center_left_ratio <= x <= a.controller.cfg.center_right_ratio:
        assert any(action.steer_correction_rpm != 0 or action.kind.startswith('rotate_')
                   for action in decision.actions)
    assert not any(action.speed_percent > 0 for action in decision.actions
                   if action.kind in {'forward', 'backward', 'steer_left', 'steer_right'})
    return current, decision


@pytest.mark.parametrize('x', [.375, .625, .5425, .5668])
def test_side_and_center_band_observations_request_existing_forward_exit(parked, x, caplog):
    a = parked
    current, decision = side_preview(a, x=x)
    with caplog.at_level('INFO'):
        # Qualified preview requests a release of ordinary holding current;
        # it does not pretend that the executor has completed that release.
        assert not release(a, current, decision)
    assert a.evidence.forward_resume_live(a.clock.now)
    assert a.evidence.reason == 'await_current_release'
    lateral = any(action.steer_correction_rpm != 0 or action.kind.startswith('rotate_')
                  for action in decision.actions)
    assert f'lateral_motion_present={lateral} translation_present=False early_forward=True' in caplog.text
    assert a.owner._near_yaw_park_request is not None
    assert a.owner._depth30_linear_snapshot is None
    assert not a.backend.pairs

    # Existing executor contract: mark only after 0A readback / FREE completes.
    a.evidence.mark_current_released(a.clock.now)
    advance(a, a.clock.now+.01)
    assert release(a, current, decision)
    assert a.owner._near_yaw_park_request is None
    assert a.owner._depth30_linear_snapshot is None
    assert not a.backend.pairs  # no speed write from a release/preview
    a.owner._refresh_visual_depth_linear_authority(
        current, current.persons[0], decision, is_fresh_depth=True,
        target_steerable=True, low_quality_visible=False)
    assert a.owner._fresh_depth_linear_snapshot(1) is None

    # Separate fresh samples start the normal, brake-bounded forward ramp.
    approved = []
    for index in range(1, 4):
        advance(a, current.distance_state.sample_timestamp+index*.05)
        _, actions, _ = decide_commit(a, a.frame(1.63, rpm=0.))
        approved.append(max([action.speed_percent*2 for action in actions
                             if action.kind == 'forward'] or [0]))
    assert approved[0] == 0 < approved[1] <= approved[2]
    assert approved[2] <= 24


@pytest.mark.parametrize('case', [
    'old_depth', 'future_depth', 'replay', 'wrong_uid', 'quality', 'not_steerable',
    'hazard', 'obstacle', 'explicit_stop', 'shutdown', 'search', 'new_park',
    'raw_near', 'target_latched', 'brake_latched', 'depth_loop', 'rotation_only',
    'missing_closure', 'not_outward', 'no_preview', 'zero_cap', 'old_capture',
    'countersteer',
])
def test_lateral_preview_never_creates_an_early_hint_without_current_forward_evidence(parked, monkeypatch, case):
    a = parked
    current, decision = side_preview(a)
    kwargs = {}
    if case == 'old_depth': a.clock.now += .181
    elif case == 'future_depth': current = replace(current, distance_state=replace(
        current.distance_state, sample_timestamp=a.clock.now+.01))
    elif case == 'replay': current = replace(current, distance_state=replace(
        current.distance_state, source_detail='depth_multiregion_reused_hold'))
    elif case == 'wrong_uid': current = replace(current, persons=[replace(current.persons[0], track_id=2)])
    elif case == 'quality': kwargs['low_quality_visible'] = True
    elif case == 'not_steerable': kwargs['target_steerable'] = False
    elif case == 'hazard': current = replace(current, hazard=replace(current.hazard, active=True))
    elif case == 'obstacle': current = replace(current, obstacles=replace(current.obstacles, front=True))
    elif case == 'explicit_stop': decision = replace(decision, explicit_stop_requested=True)
    elif case == 'shutdown': decision = replace(decision, shutdown_requested=True)
    elif case == 'search': a.owner.search_state = 'searching'
    elif case == 'new_park': decision = replace(decision, near_yaw_park_requested=True)
    elif case == 'raw_near': current = replace(current, distance_state=replace(current.distance_state, raw_distance_m=1.3))
    elif case in {'target_latched', 'brake_latched'}: current = replace(current, distance_state=replace(
        current.distance_state, **{case: True}))
    elif case == 'depth_loop': kwargs['control_source'] = 'depth30'
    elif case == 'rotation_only': monkeypatch.setattr(runtime, 'FOLLOW_ROTATION_ONLY', True)
    elif case == 'missing_closure': a.controller._braking_rate_source = 'encoder_fallback'
    elif case == 'not_outward': a.controller._braking_range_rate = -.1
    elif case == 'no_preview': a.controller.last_distance_pid_result = replace(
        a.controller.last_distance_pid_result, pi_status='tracking')
    elif case == 'zero_cap': a.controller.last_distance_pid_result = replace(
        a.controller.last_distance_pid_result, approach_cap_rpm=0.)
    elif case == 'old_capture': current = replace(current, capture_timestamp=a.clock.now-.2)
    elif case == 'countersteer': a.controller.last_steering_pid_result = replace(
        a.controller.last_steering_pid_result, output_floor_reason='predictive_countersteer')
    assert not release(a, current, decision, **kwargs)
    assert a.evidence.forward_resume_until == 0
    assert a.owner._near_yaw_park_request is not None
    assert a.owner._depth30_linear_snapshot is None
    assert not a.backend.pairs


def test_real_reverse_intent_is_not_overridden_by_a_forward_preview(parked):
    a = parked
    current, _ = side_preview(a)
    decision = ControlDecision(actions=[ControlAction.backward(10, 'reverse')], reason='reverse')
    assert not release(a, current, decision)
    assert a.evidence.forward_resume_until == 0


def test_lateral_hint_replay_never_extends_either_physical_deadline(parked):
    a = parked
    current, decision = side_preview(a)
    assert not release(a, current, decision)
    deadline = a.evidence.forward_resume_until
    advance(a, a.clock.now+.04)
    assert not release(a, current, decision)
    assert a.evidence.forward_resume_until == deadline
    advance(a, a.clock.now+.15)
    assert not release(a, current, decision)
    assert not a.evidence.forward_resume_live(a.clock.now)
