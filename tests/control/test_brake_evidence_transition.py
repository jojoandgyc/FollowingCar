"""Evidence-switch regression, no sensors, runtime threads or motor writes."""
import pytest

from car_control_modular.distance_pi import DistancePiConfig, DistancePiController
from car_control_modular.longitudinal_approach import closure_rotation_bound
from test_distance_tracking_response import setup
from test_distance_pi_controller import configured, step
from test_cap437_braking_geometry import observation


def controller():
    return DistancePiController(DistancePiConfig(kp_per_sec=3., launch_request_rpm=180.,
                                               motion_memory_sec=.35, physical_ttl_sec=.25))


def update(c, ts, **changes):
    args = dict(sample_timestamp=ts, execution_now=ts+.02, deadband_m=.03,
                max_output_rpm=200., rise_rpm_per_sec=240., ego_forward_rpm=50.,
                range_rate_m_s=.6, raw_closure_valid=True, allow_motion_memory=True,
                raw_distance_m=2.)
    distance = changes.pop('distance', 2.)
    args.update(changes)
    return c.update(distance, 1.4, **args)


@pytest.mark.parametrize('age', [.179, .199, .233, .249])
def test_accepted_roi_crossing_old_age_gate_keeps_controller_closure(setup, caplog, age):
    clock, c, frame = configured(setup, distance_pi_launch_request_rpm=180.,
        distance_pi_kp_per_sec=3., distance_pi_motion_memory_sec=.35,
        depth_longitudinal_sample_max_age_sec=.25)
    for d in (1.85, 1.9):
        step(c, observation(frame, clock, d, rpm=50., yaw=4.9, raw_yaw=4.9, age=.17))
        clock.now += .05
    r0 = c.last_distance_pid_result
    step(c, observation(frame, clock, 1.95, rpm=50., yaw=5.1, raw_yaw=9.7632, age=age))
    r = c.last_distance_pid_result
    assert r.pi_brake_source == 'raw_relative_motion'
    assert r.output_rpm > 80 and r.output_rpm >= r0.output_rpm-10
    assert 'geometry_limit_ms=250' in caplog.text
    assert 'brake_recovery_limited=' in caplog.text


@pytest.mark.parametrize('yaw', [4.99, 5.01, 9.7632, -7.8375])
def test_no_discontinuous_geometry_gate_at_five_dps(yaw):
    args = dict(depth=1.7, bearing_deg=5., yaw_dps=yaw, geometry_age=.233,
                max_yaw_dps=35.)
    assert closure_rotation_bound(**args, turning_geometry_max_age_sec=.25) is not None
    if abs(yaw) > 5:
        assert closure_rotation_bound(**args) is None  # legacy unchanged


@pytest.mark.parametrize('changes', [dict(geometry_age=.251), dict(yaw_dps=35.01),
    dict(depth=4., bearing_deg=35., yaw_dps=30.), dict(bearing_deg=46.),
    dict(turning_geometry_max_age_sec=.251)])
def test_extended_geometry_still_has_hard_bounds(changes):
    args = dict(depth=1.7, bearing_deg=5., yaw_dps=9., geometry_age=.233,
                max_yaw_dps=35., turning_geometry_max_age_sec=.25)
    assert closure_rotation_bound(**dict(args, **changes)) is None


def test_near_but_outside_hold_band_can_use_degrading_evidence():
    c = controller()
    a = update(c, 100., distance=1.69, raw_distance_m=1.69, range_rate_m_s=.2)
    b = update(c, 100.05, distance=1.70, raw_distance_m=1.70,
               raw_closure_valid=False, range_rate_m_s=-.68)
    assert b.brake_source == 'relative_motion_memory'
    assert 0 < b.output_rpm <= a.output_rpm
    assert b.motion_origin_ts == 100.
    assert b.motion_uncertainty_m_s > 0.


def test_raw_near_reading_cannot_hide_behind_filtered_distance():
    c = controller()
    update(c, 100.)
    r = update(c, 100.05, raw_distance_m=1.42, raw_closure_valid=False, range_rate_m_s=-3.)
    assert r.brake_source == 'motion_unknown_bound'
    assert r.output_rpm == 0.


def test_hard_rejection_still_decelerates_but_evidence_return_cannot_rebound():
    c = controller()
    a = update(c, 100.)
    b = update(c, 100.1, raw_closure_valid=False, allow_motion_memory=False,
               range_rate_m_s=-50*.816814/60.)
    r = update(c, 100.2)
    assert a.output_rpm > 130 and b.output_rpm < 45
    assert b.output_rpm < r.output_rpm <= b.output_rpm+24
    assert r.brake_recovery_limited
    assert r.demand_limit_reason == 'brake_evidence_recovery'
    assert r.demand_rpm == 180 and not r.software_rise_bypassed
    assert r.output_rpm <= r.cap_rpm


def test_recovery_uses_final_approval_not_old_high_request():
    c = controller()
    update(c, 100.)
    update(c, 100.1, raw_closure_valid=False, allow_motion_memory=False)
    c.accept_output_limit(100.1, 10.)
    r = update(c, 100.15)
    assert r.brake_recovery_anchor_rpm == 10.
    assert r.output_rpm <= 22


@pytest.mark.parametrize('event', ['expired', 'revoked', 'rejected'])
def test_no_acceleration_credit_during_lost_authorization(event):
    c = controller()
    update(c, 100.)
    update(c, 100.1, raw_closure_valid=False, allow_motion_memory=False)
    ts = 100.15
    if event == 'expired': ts = 100.4
    if event == 'revoked': c.suspend(100.12, 'expired', reset_execution=True)
    if event == 'rejected': c.reject_output(100.1)
    r = update(c, ts, ego_forward_rpm=15.)
    assert r.output_rpm <= 15
    assert r.sample_dt_sec == 0.
    assert r.brake_recovery_limited


def test_duplicate_or_out_of_order_does_not_donate_rise_budget():
    c = controller()
    update(c, 100.)
    b = update(c, 100.1, raw_closure_valid=False, allow_motion_memory=False)
    assert update(c, 100.1, execution_now=100.15).status == 'duplicate'
    assert update(c, 100.09, execution_now=100.15).status == 'out_of_order'
    r = update(c, 100.15)
    assert r.output_rpm <= b.output_rpm+12


def test_recovery_finishes_without_fixed_low_rpm_stage():
    c = controller()
    update(c, 100.)
    update(c, 100.1, raw_closure_valid=False, allow_motion_memory=False)
    outputs = []
    for i in range(1, 14):
        r = update(c, 100.1+i*.05)
        outputs.append(r.output_rpm)
    assert all(a <= b <= a+12 for a, b in zip(outputs, outputs[1:]))
    assert outputs[-1] > 130 and not r.brake_recovery_limited
    assert not c._brake_recovery_pending


def test_repeated_soft_dropouts_do_not_reset_recovery_to_low_fixed_rpm():
    c = controller()
    update(c, 100.)
    update(c, 100.1, raw_closure_valid=False, allow_motion_memory=False)
    last = update(c, 100.15)
    for i in range(1, 6):
        ts = 100.15+i*.1-.05
        warm = update(c, ts, distance=2.+.6*(ts-100.15),
                      raw_distance_m=2.+.6*(ts-100.15), raw_closure_valid=False)
        assert warm.output_rpm <= last.output_rpm
        ts += .05
        last = update(c, ts, distance=2.+.6*(ts-100.15), raw_distance_m=2.+.6*(ts-100.15))
        assert warm.output_rpm <= last.output_rpm <= warm.output_rpm+12
    assert last.output_rpm > 90


def test_actual_fast_approach_overrides_transition_smoothing_now():
    c = controller()
    update(c, 100.)
    update(c, 100.1, raw_closure_valid=False, allow_motion_memory=False)
    update(c, 100.15)
    r = update(c, 100.2, distance=1.55, raw_distance_m=1.55, range_rate_m_s=-2.)
    assert r.output_rpm == 0 and r.cap_rpm == 0


def test_identity_reset_has_no_old_brake_or_rise_memory():
    c = controller()
    update(c, 100.)
    update(c, 100.1, raw_closure_valid=False, allow_motion_memory=False)
    c.reset()
    r = update(c, 100.15)
    assert not r.brake_recovery_limited
    assert r.software_rise_bypassed


def test_initial_launch_and_continuous_valid_samples_keep_motor_ramp_ownership():
    c = controller()
    for ts in (100., 100.05, 100.1):
        r = update(c, ts, rise_rpm_per_sec=1.)
        assert r.output_rpm > 130 and not r.brake_recovery_limited
        assert r.software_rise_bypassed
