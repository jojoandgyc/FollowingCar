"""Never add the last encoder speed to a historical average closure rate.

No camera, serial link, motor, or target-speed feedforward is used here.
"""
from dataclasses import replace

import pytest

from car_control_modular.distance_pi import DistancePiConfig, DistancePiController
from car_control_modular.longitudinal_approach import RawDepthClosingWindow
from test_distance_tracking_response import setup
from test_distance_pi_controller import configured, step


CIRCUMFERENCE = .816814


def make_window(*, first_ego=1.4, last_ego=.6, target_speed=0., rotation=0.):
    """Exact linearly changing ego velocity and constant target ground speed."""
    window = RawDepthClosingWindow()
    first_distance = 3.
    for t in (0., .08, .16):
        acceleration = (last_ego-first_ego)/.16
        ego = first_ego+acceleration*t
        raw = first_distance+(target_speed+rotation-first_ego)*t-.5*acceleration*t*t
        window.update(uid=1, stamp=10.+t, raw=raw, rotation=rotation,
                      ego_speed=ego, feedback_stamp=10.+t)
    return window, raw


def evaluate(window, distance, ego, *, evidence=True, **changes):
    config = DistancePiConfig(kp_per_sec=3., physical_ttl_sec=.25,
                              launch_request_rpm=180., launch_full_error_m=.5)
    args = dict(sample_timestamp=window.samples[-1][0], execution_now=10.17,
                deadband_m=.03, max_output_rpm=200.,
                ego_forward_rpm=ego*60./CIRCUMFERENCE, range_rate_m_s=window.rate,
                raw_closure_valid=True, raw_distance_m=distance)
    if evidence:
        args['raw_motion_evidence'] = window.motion_evidence()
    args.update(changes)
    return DistancePiController(config).update(distance, 1.4, **args)


@pytest.mark.parametrize('first,last', [(1.4,.6), (.6,1.4), (1.,1.)])
@pytest.mark.parametrize('target', [0., .4, -.4])
def test_static_walking_and_approaching_target_during_car_accel_or_brake(first,last,target):
    window, distance = make_window(first_ego=first, last_ego=last, target_speed=target)
    result = evaluate(window, distance, last)
    assert result.motion_window_used
    assert result.motion_window_target_speed_m_s == pytest.approx(target)
    assert result.target_velocity_bound_m_s == pytest.approx(target)
    assert result.effective_range_rate_m_s == pytest.approx(target-last)
    assert result.closing_m_s == pytest.approx(max(0., last-target))
    assert result.motion_window_range_rate_m_s == pytest.approx(target-.5*(first+last))


def test_braking_does_not_invent_target_walking_toward_car():
    window, distance = make_window()
    old = evaluate(window, distance, .6, evidence=False)
    fixed = evaluate(window, distance, .6)
    assert old.target_velocity_bound_m_s == pytest.approx(-.4)
    assert fixed.target_velocity_bound_m_s == pytest.approx(0.)
    assert fixed.cap_rpm > old.cap_rpm+20.


def test_acceleration_does_not_invent_target_walking_away():
    window, distance = make_window(first_ego=.6, last_ego=1.4)
    old = evaluate(window, distance, 1.4, evidence=False)
    fixed = evaluate(window, distance, 1.4)
    assert old.target_velocity_bound_m_s == pytest.approx(.4)
    assert fixed.target_velocity_bound_m_s == pytest.approx(0.)
    assert fixed.cap_rpm < old.cap_rpm-20.


@pytest.mark.parametrize('rotation', [.05, .20, .25])
def test_same_window_preserves_conservative_rotation_subtraction(rotation):
    window, distance = make_window(target_speed=.4, rotation=rotation)
    result = evaluate(window, distance, .6)
    assert result.target_velocity_bound_m_s == pytest.approx(.4)
    # If the real rotation was smaller than its upper bound, the estimate
    # remains LOWER, never permission to claim extra target retreat.
    conservative, raw = make_window(target_speed=.4-rotation, rotation=rotation)
    bounded = evaluate(conservative, raw, .6)
    assert bounded.target_velocity_bound_m_s == pytest.approx(.4-rotation)
    assert bounded.cap_rpm < result.cap_rpm


@pytest.mark.parametrize('change', [
    {'sample_timestamp':10.15}, {'range_rate_m_s':-.1},
    {'target_speed_bound_m_s':float('nan')}, {'span_sec':.5},
    {'sample_count':1}, {'sample_count':True},
])
def test_invalid_or_mismatched_window_cannot_supply_target_motion(change):
    window, distance = make_window(target_speed=.8)
    evidence = replace(window.motion_evidence(), **change)
    result = evaluate(window, distance, .6, raw_motion_evidence=evidence)
    assert not result.motion_window_used
    assert result.target_velocity_bound_m_s == 0.
    assert result.brake_source == 'stationary_fallback'


def test_invalid_encoder_cannot_use_otherwise_valid_window():
    window, distance = make_window(target_speed=.8)
    result = evaluate(window, distance, .6, ego_forward_rpm=None)
    assert not result.motion_window_used
    assert result.target_velocity_bound_m_s == 0.


def test_stale_sample_cannot_use_window_to_renew_authority():
    window, distance = make_window(target_speed=.8)
    result = evaluate(window, distance, .6, execution_now=10.42)
    assert result.status == 'stale_sample'
    assert result.output_rpm == 0.
    assert not result.motion_window_used


def test_duplicate_does_not_refresh_motion_origin():
    window, distance = make_window(target_speed=.4)
    p = DistancePiController(DistancePiConfig(physical_ttl_sec=.25))
    kwargs = dict(sample_timestamp=10.16, execution_now=10.17, deadband_m=.03,
                  max_output_rpm=200., ego_forward_rpm=.6*60/CIRCUMFERENCE,
                  range_rate_m_s=window.rate, raw_closure_valid=True,
                  raw_motion_evidence=window.motion_evidence(), raw_distance_m=distance)
    first = p.update(distance,1.4,**kwargs)
    second = p.update(distance,1.4,**dict(kwargs,execution_now=10.19))
    assert second.status == 'duplicate'
    assert second.motion_origin_ts == first.motion_origin_ts == 10.16
    assert second.motion_window_target_speed_m_s == first.motion_window_target_speed_m_s


def test_window_needs_all_encoder_endpoints_and_matching_physical_time():
    window = RawDepthClosingWindow()
    window.update(uid=1,stamp=10.,raw=3.,rotation=0.)
    window.update(uid=1,stamp=10.08,raw=2.95,rotation=0.,ego_speed=.6,feedback_stamp=10.08)
    assert window.rate is not None
    assert window.motion_evidence() is None
    window.update(uid=1,stamp=10.16,raw=2.9,rotation=0.,ego_speed=.6,feedback_stamp=10.5)
    assert window.status == 'invalid_encoder_alignment'
    assert window.motion_evidence() is None


@pytest.mark.parametrize('feedforward', [False,True])
def test_real_controller_receives_independent_window_not_matching_gate(setup, feedforward):
    clock, controller, frame = configured(setup, distance_feedforward_enable=feedforward)
    for t,v,distance in [(0.,1.4,3.),(.08,1.,2.904),(.16,.6,2.84)]:
        clock.now = 100.+t
        step(controller, frame(distance,rpm=v*60/CIRCUMFERENCE))
    result = controller.last_distance_pid_result
    assert result.pi_motion_window_used
    assert result.pi_motion_window_span_sec == pytest.approx(.16)
    assert result.pi_motion_window_target_speed_m_s == pytest.approx(0.)
    assert result.pi_target_velocity_bound_m_s == pytest.approx(0.)
    assert result.pi_effective_range_rate_m_s == pytest.approx(-.6)
    assert result.tracking_base_rpm == 0.


@pytest.mark.parametrize('fault', ['yaw','feedback_age','sample_age','hazard'])
def test_controller_safety_rejection_clears_window_bridge(setup, fault):
    clock, controller, frame = configured(setup)
    step(controller,frame(3.,rpm=40.))
    clock.now += .08
    step(controller,frame(2.96,rpm=40.))
    assert controller.last_distance_pid_result.pi_motion_window_used
    clock.now += .08
    current = frame(2.92,rpm=40.)
    if fault=='yaw':
        current=replace(current,steering_feedback=replace(current.steering_feedback,yaw_rate_right_dps=80.))
    elif fault=='feedback_age':
        current=replace(current,steering_feedback=replace(current.steering_feedback,timestamp=clock.now-.31))
    elif fault=='sample_age':
        current=frame(2.92,rpm=40.,stamp=clock.now-.26)
    else:
        current=replace(current,hazard=replace(current.hazard,active=True))
    step(controller,current)
    assert controller._raw_closing_window.motion_evidence() is None


def test_legacy_pi_api_keeps_original_numerics_when_window_is_not_supplied():
    window, distance = make_window()
    legacy = evaluate(window,distance,.6,evidence=False)
    optional_none = evaluate(window,distance,.6,raw_motion_evidence=None)
    assert legacy == optional_none


@pytest.mark.parametrize('feedforward', [False, True])
def test_production_pi_missing_paired_window_never_uses_legacy_speed_sum(setup, monkeypatch, feedforward):
    clock, controller, frame = configured(setup, distance_feedforward_enable=feedforward)
    # Model a partially rebuilt window: distance rate exists, paired encoder
    # endpoints do not. Its positive trend plus the current .8m/s ego speed
    # must not be silently credited as .9m/s target motion.
    monkeypatch.setattr(controller._raw_closing_window, 'motion_evidence', lambda: None)
    for index in range(3):
        clock.now = 100.+index*.08
        step(controller,frame(3.+index*.008,rpm=.8*60/CIRCUMFERENCE))
    result = controller.last_distance_pid_result
    assert controller._braking_rate_source == 'raw_depth_window'
    assert controller._raw_closing_window.rate == pytest.approx(.1)
    assert controller._braking_motion_evidence is None
    assert not result.pi_motion_window_used
    assert result.pi_target_velocity_bound_m_s == 0.
    assert result.pi_effective_range_rate_m_s == pytest.approx(-.8)
    assert result.approach_closing_m_s == pytest.approx(.8)
    assert result.pi_brake_source in {'motion_unknown_bound', 'stationary_fallback'}
