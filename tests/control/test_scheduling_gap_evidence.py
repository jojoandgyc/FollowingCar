"""CAP295/340: skipped ROI attempts are not fresh samples or motor leases."""
from dataclasses import replace

import pytest

from test_distance_tracking_response import setup, decide
from test_longitudinal_continuity_metrics import seed


def missing(frame, *, yaw=0):
    current = frame(None, rpm=30, yaw=yaw)
    return replace(current, distance_state=replace(current.distance_state,
        sample_timestamp=None, source_detail="depth_detector_bbox_stale"))


def test_low_yaw_failed_roi_keeps_chain_but_cannot_authorize_or_refresh(setup, caplog):
    clock, controller, frame = seed(setup)
    stamp = controller._longitudinal_motion_stamp
    previous = controller._longitudinal_feedforward._previous
    evidence = controller._longitudinal_motion_evidence
    for age in (.02, .06, .10):
        clock.now = stamp + age
        current = missing(frame)
        controller._observe_longitudinal_motion(current, current.persons[0])
        assert controller._distance_pid_sample_timestamp is None
        assert controller._tracking_base_rpm(1.9, clock.now) is None
        assert controller._longitudinal_motion_stamp == stamp
        assert controller._longitudinal_motion_evidence is evidence
        assert controller._longitudinal_feedforward._previous is previous
    clock.now = stamp + .12
    decide(controller, frame(1.9, rpm=30))
    assert controller._longitudinal_motion_evidence.status == "ready"
    assert controller._longitudinal_motion_evidence.sample_count == 3
    assert 'deadline_renewed=False' in caplog.text


def test_preservation_cannot_be_chained_beyond_original_expiry(setup):
    clock, controller, frame = seed(setup)
    stamp = controller._longitudinal_motion_stamp
    for age in (.10, .17, .181):
        clock.now = stamp + age
        current = missing(frame)
        controller._observe_longitudinal_motion(current, current.persons[0])
    assert controller._longitudinal_motion_evidence is None
    clock.now += .01
    decide(controller, frame(1.9, rpm=30))
    assert controller._longitudinal_motion_evidence.status == "warming_up"


def test_uncompensated_turn_preserves_prior_not_derivative(setup):
    clock, controller, frame = seed(setup)
    origin = controller._longitudinal_bridge.origin
    clock.now += .03
    current = missing(frame, yaw=10)
    controller._observe_longitudinal_motion(current, current.persons[0])
    assert controller._longitudinal_bridge.origin is origin
    assert controller._longitudinal_feedforward._previous is None
    assert controller._tracking_base_rpm(1.9, clock.now) is None


@pytest.mark.parametrize('change', ['hazard', 'obstacle', 'uid', 'search', 'near', 'brake',
    'safety_distance', 'pixels', 'background', 'jump', 'raw', 'stale_encoder',
    'untrusted_encoder', 'reverse', 'fast_yaw', 'bearing', 'corrupt_stamp', 'replay_jump'])
def test_no_observation_preservation_never_bypasses_rejection(setup, change):
    clock, controller, frame = seed(setup)
    clock.now += .03
    current = missing(frame)
    if change == 'hazard': current = replace(current, hazard=replace(current.hazard, active=True))
    elif change == 'obstacle': current = replace(current, obstacles=replace(current.obstacles, front=True))
    elif change == 'uid': controller.active_target_id = 2
    elif change == 'search': controller.search_state = 'searching'
    elif change == 'near': current = replace(current, distance_m=1.4)
    elif change == 'brake': current = replace(current, distance_state=replace(current.distance_state, brake_latched=True))
    elif change == 'safety_distance': current = replace(current, distance_state=replace(current.distance_state, safety_distance_m=.4))
    elif change in {'pixels', 'background', 'jump'}:
        current = replace(current, distance_state=replace(current.distance_state,
            source_detail={'pixels':'depth_invalid_pixels', 'background':'far_background_guard',
                           'jump':'distance_jump_pending_1_of_3'}[change]))
    elif change == 'raw': current = replace(current, distance_state=replace(current.distance_state, raw_distance_m=3.))
    elif change == 'corrupt_stamp': current = replace(current, distance_state=replace(current.distance_state, sample_timestamp=float('nan')))
    elif change == 'replay_jump': current = replace(current, distance_state=replace(current.distance_state,
        source_detail='distance_jump_pending_1_of_3', temporal_status='duplicate',
        observation_timestamp=controller._longitudinal_motion_stamp))
    elif change == 'stale_encoder': current = replace(current, steering_feedback=replace(current.steering_feedback, timestamp=clock.now-.2))
    elif change == 'untrusted_encoder': current = replace(current, steering_feedback=replace(current.steering_feedback, trustworthy=False))
    elif change == 'reverse': current = replace(current, steering_feedback=replace(current.steering_feedback, left_forward_rpm=-10))
    elif change == 'fast_yaw': current = missing(frame, yaw=20)
    elif change == 'bearing': current = replace(current, persons=[replace(current.persons[0], bbox=(450,50,630,460))])
    controller._observe_longitudinal_motion(current, current.persons[0])
    assert controller._longitudinal_motion_evidence is None


def recovery(setup, *, physical_gap=.17, processing_gap=.23, rpm=55, initial_ramp=None):
    clock, controller, frame = setup
    controller.cfg = replace(controller.cfg, depth_measured_recovery_enable=True,
        depth_recovery_stage1_sec=.2, depth_recovery_stage2_sec=.4)
    start = clock.now
    controller._depth_recovery_anchor = (1, start, 2.)
    controller._depth_quality_degraded = False
    controller._depth_recovery_started_at = initial_ramp
    controller._distance_pid._last_output_rpm = 55.
    controller._depth_last_approved_forward_rpm = 55.
    clock.now += .1
    controller._note_depth_quality_failure(missing(frame), clock.now)
    clock.now = start + processing_gap
    current = frame(2.02, rpm=rpm, stamp=start+physical_gap)
    return clock, controller, frame, current


def test_delayed_processing_with_continuous_depth_and_wheels_does_not_restart_25(setup, caplog):
    clock, controller, frame, current = recovery(setup)
    assert controller._limit_depth_quality_forward_percent(current, 55, clock.now) == 55
    assert controller._depth_recovery_started_at is None
    assert 'depth_recovery_continuity_restored' in caplog.text
    assert 'depth_speed_recovery_started' not in caplog.text


def test_repeated_failed_attempt_keeps_original_non_motion_hint(setup):
    clock, controller, frame, current = recovery(setup)
    hint = controller._depth_gap_resume_hint
    controller._note_depth_quality_failure(missing(frame), clock.now)
    assert controller._depth_gap_resume_hint is hint
    assert controller._depth_recovery_anchor is None
    assert controller._limit_depth_quality_forward_percent(current, 55, clock.now) == 55
    assert controller._depth_gap_resume_hint is None


@pytest.mark.parametrize('change', ['long_physical_gap', 'long_processing_gap', 'stopped',
    'slowed', 'freshness', 'jump', 'closing', 'uid', 'hazard', 'encoder', 'reverse', 'pixels', 'background'])
def test_recovery_continuity_requires_new_depth_and_actual_wheel_continuity(setup, change, caplog):
    clock, controller, frame, current = recovery(setup,
        physical_gap=.22 if change=='long_physical_gap' else .17,
        processing_gap=.36 if change=='long_processing_gap' else .23,
        rpm=0 if change=='stopped' else 10 if change=='slowed' else 55)
    if change == 'freshness': current = replace(current, distance_state=replace(current.distance_state, sample_timestamp=clock.now-.2))
    elif change == 'jump': current = replace(current, distance_state=replace(current.distance_state, raw_distance_m=2.5))
    elif change == 'closing': current = replace(current, distance_state=replace(current.distance_state, raw_distance_m=1.8))
    elif change == 'background': current = replace(current, distance_state=replace(current.distance_state, source_detail='depth_far_background_guard'))
    elif change == 'uid': controller.active_target_id = 2
    elif change == 'hazard': current = replace(current, hazard=replace(current.hazard, active=True))
    elif change == 'encoder': current = replace(current, steering_feedback=replace(current.steering_feedback, timestamp=clock.now-.2))
    elif change == 'reverse': current = replace(current, steering_feedback=replace(current.steering_feedback, left_forward_rpm=-10))
    elif change == 'pixels':
        controller._note_depth_quality_failure(replace(missing(frame), distance_state=replace(
            missing(frame).distance_state, source_detail='depth_invalid_pixels')), clock.now)
    controller._limit_depth_quality_forward_percent(current, 55, clock.now)
    assert 'depth_recovery_continuity_restored' not in caplog.text
    # This only selects a recovery cap. Physical TTL and safety authorization
    # remain independently covered by test_depth_authority_three_clocks.


def test_existing_ramp_clock_is_not_restarted_or_advanced_by_failed_attempt(setup):
    clock, controller, frame, current = recovery(setup, initial_ramp=100.)
    controller._limit_depth_quality_forward_percent(current, 55, clock.now)
    assert controller._depth_recovery_started_at == 100.
