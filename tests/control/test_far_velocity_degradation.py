"""CAP211/307: fresh far range must not imply a launch-speed reset. No hardware."""
from dataclasses import replace
import configparser
import math
import os
from pathlib import Path

import pytest

from car_control_modular.controllers import FollowSafetyController
from car_control_modular.longitudinal_feedforward import (
    LongitudinalFeedforwardBridge, LongitudinalFeedforwardEvidence,
    LongitudinalFeedforwardEstimator, LongitudinalFeedforwardConfig,
    far_closure_consistent,
)
from test_distance_tracking_response import setup, decide
from test_scheduling_gap_evidence import missing


def test_runtime_wheel_geometry_has_one_shared_scale(monkeypatch):
    from car_control_modular.config_loader import load_config_to_env
    cfg = configparser.ConfigParser()
    cfg.read(Path(__file__).resolve().parents[2] / 'car_control_modular/config/reid_runtime.ini')
    assert float(cfg['mmwave_match']['fusion_encoder_wheel_circumference_m']) == pytest.approx(math.pi*.26, abs=1e-6)
    env = {}
    monkeypatch.setattr(os, 'environ', env)
    load_config_to_env(str(Path(__file__).resolve().parents[2] / 'car_control_modular/config/reid_runtime.ini'))
    assert float(env['VISION_MMWAVE_FUSION_ENCODER_WHEEL_CIRCUMFERENCE_M']) == pytest.approx(.816814)


@pytest.mark.parametrize('rpm', [30., 60., 90.])
def test_static_plane_motion_with_26cm_wheel_does_not_look_like_moving_person(rpm):
    circumference = math.pi*.26
    e = LongitudinalFeedforwardEstimator(LongitudinalFeedforwardConfig(wheel_circumference_m=circumference))
    for n in range(5):
        t = 100+n*.05
        result = e.update(now=t, sample_timestamp=t, distance_m=3-rpm*circumference/60*n*.05,
            target_id=1, trusted=True, feedback_timestamp=t, ego_forward_rpm=rpm,
            yaw_rate_dps=0)
    assert not result.eligible
    assert abs(result.target_speed_m_s or 0) < .03


@pytest.mark.parametrize('change', [dict(distance=1.79), dict(previous_distance=1.79),
    dict(distance=2.6), dict(interval=.01), dict(interval=.351), dict(ego_speed=0),
    dict(ego_speed=-.5), dict(ego_speed=float('nan')), dict(distance=1.95,previous_distance=2.09,interval=.2)])
def test_far_closure_requires_bounded_travel_and_time_to_near_band(change):
    args=dict(distance=2.805, previous_distance=2.881, interval=.22, ego_speed=.78, near=1.8)
    args.update(change)
    assert not far_closure_consistent(**args)


def seeded(setup):
    clock, old, frame = setup
    c = FollowSafetyController(replace(old.cfg, distance_matching_base_max_rpm=80,
        distance_feedforward_wheel_circumference_m=.816814, depth_measured_recovery_enable=True))
    c.active_target_id=1
    c._has_seen_person=True
    for _ in range(4):
        decide(c, frame(2.9, rpm=60))
        clock.now += .05
    assert c._longitudinal_motion_evidence.eligible
    return clock,c,frame


def test_warming_after_uncompensated_gap_uses_decaying_prior_not_zero(setup, caplog):
    clock,c,frame=seeded(setup)
    origin=c._longitudinal_bridge.origin
    c._observe_longitudinal_motion(missing(frame,yaw=8),frame(2.9).persons[0])
    assert c._longitudinal_feedforward._previous is None
    assert c._tracking_base_rpm(2.9,clock.now) is None
    clock.now += .04
    decide(c,frame(2.86,rpm=60))
    assert c._longitudinal_motion_evidence.status == 'transient_bridge'
    assert c.last_distance_pid_result.tracking_base_rpm == pytest.approx(origin.target_rpm - 80*.09)
    assert c._longitudinal_bridge.origin is origin
    assert 'origin_renewed=False' in caplog.text


def test_repeated_turn_updates_decay_prior_without_renewing_it(setup):
    clock,c,frame=seeded(setup)
    origin=c._longitudinal_bridge.origin
    for age in (.10,.17,.24,.31):
        clock.now=origin.sample_timestamp+age
        decide(c,frame(2.9,rpm=60,yaw=8))  # No compensated geometry: no derivative.
        assert c._longitudinal_motion_evidence.status == 'transient_bridge'
        assert c._longitudinal_bridge.origin is origin
        assert c.last_distance_pid_result.tracking_base_rpm <= origin.target_rpm-80*age+1e-6
    clock.now=origin.sample_timestamp+.351
    decide(c,frame(2.9,rpm=60,yaw=8))
    assert c._tracking_base_rpm(2.9,clock.now) is None


def test_extended_memory_never_authorizes_missing_or_expired_depth(setup):
    clock,c,frame=seeded(setup)
    origin=c._longitudinal_bridge.origin
    clock.now=origin.sample_timestamp+.24
    result=decide(c,missing(frame,yaw=8))
    assert c._longitudinal_bridge.origin is origin
    assert c._tracking_base_rpm(2.9,clock.now) is None
    assert not any(a.kind=='forward' and a.speed_percent>0 for a in result.actions)
    clock.now+=.02
    result=decide(c,frame(2.9,rpm=60,stamp=clock.now-.181))
    # Physical authorization is rechecked by runtime, not inferred from PID requests.
    assert c._tracking_base_rpm(2.9,clock.now) is None


def test_expired_translation_cannot_be_restarted_by_bridge(setup):
    clock,c,frame=seeded(setup)
    origin=c._longitudinal_bridge.origin
    clock.now=origin.sample_timestamp+.23
    c._observe_longitudinal_motion(missing(frame,yaw=8),frame(2.9).persons[0])
    clock.now+=.01
    decide(c,frame(2.86,rpm=57.5))
    assert c._longitudinal_motion_evidence.status != 'transient_bridge'


@pytest.mark.parametrize('bad', ['near','hazard','yaw','reverse','uid','jump','feedback'])
def test_degradation_does_not_bypass_safety(setup,bad):
    clock,c,frame=seeded(setup)
    f=frame(2.9,rpm=60,yaw=8)
    if bad=='near': f=frame(1.6,rpm=60,yaw=8)
    if bad=='hazard': f=replace(f,hazard=replace(f.hazard,active=True))
    if bad=='yaw': f=frame(2.9,rpm=60,yaw=16)
    if bad=='reverse': f=frame(2.9,rpm=-10,yaw=8)
    if bad=='uid': c.active_target_id=2
    if bad=='jump': f=frame(2.9,rpm=60,yaw=8,detail='distance_jump_pending')
    if bad=='feedback': f=replace(f,steering_feedback=replace(f.steering_feedback,trustworthy=False))
    c._observe_longitudinal_motion(f,f.persons[0])
    assert c._tracking_base_rpm(f.distance_m,clock.now) is None


def test_cap307_recovery_keeps_measured_speed_not_24(setup,caplog):
    clock,c,frame=seeded(setup)
    c.cfg=replace(c.cfg,forward_max_rpm=200,depth_recovery_stage1_sec=.2,depth_recovery_stage2_sec=.4)
    hint=(1,clock.now-.23,2.881,54.,None,False)
    f=frame(2.805,rpm=57.5,stamp=clock.now-.01)
    f=replace(f,distance_m=2.871)
    result=c._scheduling_recovery_cap(f,26,clock.now,hint)
    assert result==26  # 52RPM quantized request passes, not launch cap12/24RPM.
    assert 'depth_far_closing_recovery' in caplog.text


@pytest.mark.parametrize('bad',['near','jump','gap','stale','fast_closing','reverse','hazard'])
def test_far_recovery_preserves_strict_rejections(setup,bad):
    clock,c,frame=seeded(setup)
    hint=(1,clock.now-.23,2.881,54.,None,False)
    f=frame(2.805,rpm=57.5,stamp=clock.now-.01)
    if bad=='near': hint=(1,clock.now-.23,1.84,54.,None,False); f=frame(1.79,rpm=57.5)
    if bad=='jump': f=frame(2.4,rpm=57.5)
    if bad=='gap': hint=(1,clock.now-.6,2.881,54.,None,False)
    if bad=='stale': f=frame(2.805,rpm=57.5,stamp=clock.now-.181)
    if bad=='fast_closing':hint=(1,clock.now-.02,2.881,54.,None,False)
    if bad=='reverse': f=frame(2.805,rpm=-10)
    if bad=='hazard': f=replace(f,hazard=replace(f.hazard,active=True))
    assert c._scheduling_recovery_cap(f,60,clock.now,hint) is None
