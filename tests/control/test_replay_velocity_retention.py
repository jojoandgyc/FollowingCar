"""CAP197: a .45-confidence duplicate cannot erase fresh velocity evidence.

No hardware; retention is deliberately separate from PID/actuation eligibility.
"""
from dataclasses import replace
import pytest

from test_distance_tracking_response import setup, decide
from test_longitudinal_continuity_metrics import seed
from test_compensated_matching_continuity import observation


def replay(frame, stamp, *, yaw=0, confidence=.45):
    f = frame(1.9, rpm=30, yaw=yaw)
    return replace(f, distance_state=replace(f.distance_state,
        raw_distance_m=None, sample_timestamp=None, observation_timestamp=stamp,
        temporal_status='duplicate', fusion_confidence=confidence,
        fusion_mode='depth_radar_hold',
        source_detail='depth_sample_observation_discarded_fused_radar_hold_hold'))


def test_cap197_duplicate_preserves_estimator_but_full_decision_cannot_accelerate(setup, caplog):
    clock, c, frame = seed(setup)
    stamp = c._longitudinal_motion_stamp
    old = c._longitudinal_motion_evidence
    previous = c._longitudinal_feedforward._previous
    origin = c._longitudinal_bridge.origin
    pid_stamp = c._distance_pid_last_sample_timestamp
    approved = c.last_distance_pid_result.output_rpm
    for age in (.037, .06, .1):
        clock.now = stamp+age
        f = replay(frame, stamp)
        decision = decide(c, f)
        # Existing bounded hold may reduce/preserve old output, never raise it.
        assert all(a.speed_percent <= approved for a in decision.actions)
        assert c._distance_pid_last_sample_timestamp == pid_stamp
        assert c._distance_longitudinally_untrusted(f)  # motion gate NOT relaxed
        assert c._distance_pid_sample_timestamp is None
        assert c._tracking_base_rpm(1.9, clock.now) is None
        assert c._longitudinal_motion_evidence is old
        assert c._longitudinal_feedforward._previous is previous
        assert c._longitudinal_bridge.origin is origin
        assert c._longitudinal_motion_stamp == stamp
    clock.now = stamp+.12
    decide(c, frame(1.9, rpm=30))
    assert c._longitudinal_motion_evidence.sample_count == old.sample_count+1
    assert c._longitudinal_motion_evidence.eligible
    assert 'longitudinal_replay_retained' in caplog.text
    assert 'motion_authorized=False deadline_renewed=False' in caplog.text


def test_replays_cannot_renew_original_expiration(setup):
    clock, c, frame = seed(setup)
    stamp = c._longitudinal_motion_stamp
    for age in (.04, .12, .181):
        clock.now = stamp+age
        f = replay(frame, stamp)
        c._observe_longitudinal_motion(f, f.persons[0])
    assert c._longitudinal_motion_evidence is None
    assert c._longitudinal_feedforward._previous is None
    clock.now += .01
    decide(c, frame(1.9, rpm=30))
    assert c._longitudinal_motion_evidence.sample_count == 1


@pytest.mark.parametrize('change', ['confidence_lower','confidence_nan','confidence_string',
    'raw','timestamp','future_observation','older_observation','wrong_temporal','other_hold',
    'jump','pixels','background','large_distance','near','hazard','search','uid','brake',
    'safety_distance','stale_encoder','bad_encoder','reverse','yaw','bearing','stop_latched'])
def test_retention_never_bypasses_safety_or_ambiguous_provenance(setup, change, caplog):
    clock, c, frame = seed(setup)
    stamp = c._longitudinal_motion_stamp
    clock.now += .037
    f = replay(frame, stamp)
    s = f.distance_state
    if change == 'confidence_lower': s=replace(s,fusion_confidence=.2)
    if change == 'confidence_nan': s=replace(s,fusion_confidence=float('nan'))
    if change == 'confidence_string': s=replace(s,fusion_confidence='0.45')
    if change == 'raw': s=replace(s,raw_distance_m=1.9)
    if change == 'timestamp': s=replace(s,sample_timestamp=float('nan'))
    if change == 'future_observation': s=replace(s,observation_timestamp=clock.now+.1)
    if change == 'older_observation': s=replace(s,observation_timestamp=stamp-.01)
    if change == 'wrong_temporal': s=replace(s,temporal_status='new_sample')
    if change == 'other_hold': s=replace(s,fusion_mode='depth_visual_encoder_hold')
    if change in ('jump','pixels','background'):
        s=replace(s,source_detail={'jump':'distance_jump_pending_1_of_3',
            'pixels':'depth_invalid_pixels','background':'far_background_guard'}[change])
    if change == 'large_distance': s=replace(s,used_distance_m=10.)
    if change == 'brake': s=replace(s,brake_latched=True)
    if change == 'safety_distance': s=replace(s,safety_distance_m=.4)
    f=replace(f,distance_state=s)
    if change == 'near': f=replace(f,distance_m=1.4)
    if change == 'hazard': f=replace(f,hazard=replace(f.hazard,active=True))
    if change == 'search': c.search_state='searching'
    if change == 'uid': c.active_target_id=2
    if change == 'stale_encoder': f=replace(f,steering_feedback=replace(f.steering_feedback,timestamp=clock.now-.2))
    if change == 'bad_encoder': f=replace(f,steering_feedback=replace(f.steering_feedback,trustworthy=False))
    if change == 'reverse': f=replace(f,steering_feedback=replace(f.steering_feedback,left_forward_rpm=-1))
    if change == 'yaw': f=replace(f,steering_feedback=replace(f.steering_feedback,raw_yaw_rate_right_dps=16))
    if change == 'bearing': f=replace(f,persons=[replace(f.persons[0],bbox=(450,50,630,460))])
    if change == 'stop_latched': c._target_stop_latched=True
    caplog.clear()
    c._observe_longitudinal_motion(f, f.persons[0])
    assert 'longitudinal_replay_retained' not in caplog.text
    assert c._tracking_base_rpm(f.distance_m,clock.now) is None


def test_stable_compensated_turn_keeps_window_but_new_endpoint_is_rechecked(setup):
    clock,c,frame=setup
    c.cfg=replace(c.cfg,distance_turn_compensation_enable=True)
    for _ in range(4):
        decide(c,observation(frame,clock,distance=1.9,yaw=10))
        clock.now += .03
    stamp=c._longitudinal_motion_stamp
    previous=c._longitudinal_feedforward._previous
    f=replay(frame,stamp,yaw=10)
    c._observe_longitudinal_motion(f,f.persons[0])
    assert c._longitudinal_feedforward._previous is previous
    assert c._tracking_base_rpm(1.9,clock.now) is None
    clock.now += .03
    decide(c,observation(frame,clock,distance=1.9,yaw=20))
    assert c._tracking_base_rpm(1.9,clock.now) is None


def test_uncompensated_turn_retains_prior_only_not_derivative(setup):
    clock,c,frame=seed(setup)
    origin=c._longitudinal_bridge.origin
    clock.now += .03
    f=replay(frame,c._longitudinal_motion_stamp,yaw=10)
    c._observe_longitudinal_motion(f,f.persons[0])
    assert c._longitudinal_bridge.origin is origin
    assert c._longitudinal_feedforward._previous is None
    assert c._tracking_base_rpm(1.9,clock.now) is None
