"""CAP222 new-endpoint fallback through the real controller/PID interfaces."""
from dataclasses import replace

import pytest

from car_control_modular.control_types import DepthTargetObservation
from test_distance_tracking_response import setup
from test_distance_pi_controller import configured, step
from test_cap224_uncertain_zero_settling import ORIGIN, ZERO, RECOVER


def physical(frame, clock, stamp=ZERO, *, age=.0376, distance=1.759,
             raw=1.7698, rpm=8.5, feedback_age=0.):
    clock.now = stamp+age
    f = frame(distance, rpm=rpm, stamp=stamp)
    return replace(f, distance_state=replace(f.distance_state, raw_distance_m=raw),
                   steering_feedback=replace(f.steering_feedback,
                                             timestamp=clock.now-feedback_age))


def primed(setup):
    clock, c, frame = configured(
        setup, target_distance_m=1.4, distance_pi_launch_request_rpm=180.,
        distance_pi_launch_full_error_m=.5, distance_pi_kp_per_sec=3.,
        distance_approach_deceleration_m_s2=.7, distance_pi_motion_memory_sec=.35,
        depth_longitudinal_sample_max_age_sec=.25)
    # Build the paired old window rather than injecting PI motion evidence.
    step(c, physical(frame, clock, ORIGIN-.05, age=.01, distance=1.74958477,
                     raw=1.759-.1123046667*.05, rpm=0.))
    step(c, physical(frame, clock, ORIGIN, age=.177, distance=1.7552,
                     raw=1.759, rpm=0., feedback_age=.08))
    assert c.last_distance_pid_result.output_rpm == 42
    c.accept_longitudinal_limit(ORIGIN, 36.)
    c.suspend_longitudinal_authority(
        40125.029, 'lateral_zero_no_qualified_depth:visual_pid_center_hold')
    return clock, c, frame


def test_new_endpoint_prevents_first_memory_zero_without_reusing_old_grant(setup, caplog):
    clock, c, frame = primed(setup)
    memory = c._distance_pid._distance_pi._motion_memory
    f = physical(frame, clock)
    with caplog.at_level('INFO'):
        decision = step(c, f)
    r = c.last_distance_pid_result
    assert r.pi_memory_endpoint_fallback and not r.pi_motion_window_used
    assert r.pi_brake_source == 'raw_endpoint_distance_bound'
    assert r.output_rpm == decision.current_forward_percent*2 == 8
    assert r.approach_cap_rpm == 8.5 < r.pi_memory_endpoint_cap_rpm
    assert c._distance_pid._distance_pi._motion_memory == memory
    assert r.pi_motion_origin_ts == ORIGIN
    assert c._distance_pid_last_sample_timestamp == ZERO
    assert 'memory_endpoint_qualified=True memory_endpoint_fallback=True' in caplog.text
    assert 'deadline_renewed=False' in caplog.text
    assert c._raw_closing_window.status == 'warming_up'
    outputs = [r.output_rpm]
    for stamp, age, distance, raw, rpm in (
            (RECOVER, .0723, 1.768, 1.768, 8.5),
            (40125.222647697, .038, 1.768, 1.7552, 7.)):
        step(c, physical(frame, clock, stamp, age=age, distance=distance, raw=raw, rpm=rpm))
        outputs.append(c.last_distance_pid_result.output_rpm)
        assert not c.last_distance_pid_result.pi_brake_settling_limited
    # This zero-yaw replay has slightly more brake room than the recorded turn.
    assert outputs == [8, 22, 38]


def test_same_replay_without_endpoint_permission_still_reproduces_initial_zero(setup, monkeypatch):
    clock, c, frame = primed(setup)
    monkeypatch.setattr(c, '_distance_pi_endpoint_fallback_qualified', lambda *args: False)
    step(c, physical(frame, clock))
    assert c.last_distance_pid_result.output_rpm == 0
    assert c.last_distance_pid_result.pi_brake_source == 'relative_motion_memory'


@pytest.mark.parametrize('case', [
    'old_depth', 'old_feedback', 'feedback_skew', 'bad_feedback', 'reverse_wheel',
    'zero_feedback', 'high_yaw', 'old_roi', 'wrong_roi_uid', 'wrong_roi_source',
    'wrong_uid', 'hazard', 'obstacle', 'brake_latched', 'near_raw', 'raw_closer',
])
def test_unqualified_input_cannot_enter_endpoint_fallback(setup, case):
    clock, c, frame = primed(setup)
    f = physical(frame, clock, **(
        {'age': .181} if case == 'old_depth' else
        {'age': .177} if case == 'feedback_skew' else
        {'feedback_age': .151} if case == 'old_feedback' else
        {'rpm': 0.} if case == 'zero_feedback' else
        {'raw': 1.41} if case == 'near_raw' else
        {'raw': 1.748} if case == 'raw_closer' else {}))
    if case in {'feedback_skew', 'bad_feedback', 'reverse_wheel', 'high_yaw'}:
        fb = f.steering_feedback
        fb = replace(fb, **(
            {'timestamp': clock.now} if case == 'feedback_skew' else
            {'trustworthy': False} if case == 'bad_feedback' else
            {'left_forward_rpm': -1.} if case == 'reverse_wheel' else
            {'yaw_rate_right_dps': 36.}))
        f = replace(f, steering_feedback=fb)
    if case in {'old_roi', 'wrong_roi_uid', 'wrong_roi_source'}:
        t = f.persons[0]
        roi = DepthTargetObservation(t.bbox, 2 if case == 'wrong_roi_uid' else 1,
                                    1, 222, ZERO-(.31 if case == 'old_roi' else .1))
        if case == 'wrong_roi_source': roi = replace(roi, source='predicted')
        f = replace(f, persons=[replace(t, depth_observation=roi)])
    if case == 'wrong_uid': c.active_target_id = 2
    if case == 'hazard': f = replace(f, hazard=replace(f.hazard, active=True))
    if case == 'obstacle': f = replace(f, obstacles=replace(f.obstacles, front=True))
    if case == 'brake_latched': f = replace(f, distance_state=replace(f.distance_state, brake_latched=True))
    step(c, f)
    assert c.last_distance_pid_result is None or not c.last_distance_pid_result.pi_memory_endpoint_fallback


@pytest.mark.parametrize('case', [
    'feedback_expired', 'depth_expired', 'raw_state_changed', 'ego_state_changed',
    'feedback_state_changed', 'window_reset', 'uid_changed', 'reverse_active',
    'target_stopped', 'replayed_frame', 'marginal_rotation',
])
def test_qualification_rechecks_current_time_and_exact_observation_pair(setup, case):
    clock, c, frame = primed(setup)
    f = physical(frame, clock, feedback_age=.02 if case == 'feedback_expired' else 0.)
    c._observe_longitudinal_motion(f, f.persons[0])
    assert c._distance_pi_endpoint_fallback_qualified(f, clock.now, ZERO)
    if case == 'feedback_expired': clock.now += .132
    if case == 'depth_expired': clock.now = ZERO+.251
    if case == 'raw_state_changed': c._distance_pi_raw_distance_m += .001
    if case == 'ego_state_changed': c._distance_pi_ego_forward_rpm += 1.
    if case == 'feedback_state_changed': c._distance_pi_feedback_timestamp += .001
    if case == 'window_reset': c._raw_closing_window.reset()
    if case == 'uid_changed': c.active_target_id = 2
    if case == 'reverse_active': c._reverse_active = True
    if case == 'target_stopped': c._target_stop_latched = True
    if case == 'replayed_frame': f = replace(f)
    if case == 'marginal_rotation': c._distance_pi_memory_rotation_bound = .35
    assert not c._distance_pi_endpoint_fallback_qualified(f, clock.now, ZERO)


def test_duplicate_frame_cannot_reinsert_endpoint_or_advance_pi(setup):
    clock, c, frame = primed(setup)
    f = physical(frame, clock)
    step(c, f)
    result = c.last_distance_pid_result
    endpoint = c._raw_closing_window.samples[-1]
    for replay in (f, replace(f)):
        clock.now += .025
        step(c, replay)
        assert c._distance_pi_memory_endpoint is None
        assert c._raw_closing_window.samples[-1] is endpoint
        assert c.last_distance_pid_result is result
        assert c._distance_pid_last_sample_timestamp == ZERO


def test_geometry_age_gap_retains_memory_without_endpoint_permission(setup):
    from test_cap386_geometry_memory import primed as prime_geometry, current
    clock, c, frame = prime_geometry(setup)
    f = current(frame, clock)
    c._observe_longitudinal_motion(f, f.persons[0])
    assert c._distance_pi_motion_memory_allowed
    assert c._distance_pi_memory_endpoint is None
    assert not c._distance_pi_endpoint_fallback_qualified(f, clock.now, clock.now)
    step(c, f)
    assert not c.last_distance_pid_result.pi_memory_endpoint_fallback


def test_rotation_uncertainty_gap_cannot_supply_new_endpoint_permission(setup):
    from test_cap1271_rotation_memory import prime, marginal
    clock, c, frame = prime(setup)
    f = marginal(frame, clock)
    c._observe_longitudinal_motion(f, f.persons[0])
    assert c._distance_pi_motion_memory_allowed
    assert c._distance_pi_memory_endpoint is None
    assert not c._raw_closing_window.samples
    assert not c._distance_pi_endpoint_fallback_qualified(f, clock.now, clock.now)
    step(c, f)
    assert not c.last_distance_pid_result.pi_memory_endpoint_fallback
