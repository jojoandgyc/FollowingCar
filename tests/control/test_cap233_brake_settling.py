"""CAP230/233 and CAP325/327: fresh braking must not reopen in wheel lag.

Pure PI and real controller/admission tests; no camera, motor or serial I/O.
"""
from dataclasses import replace

import pytest

from car_control_modular.distance_pi import DistancePiConfig, DistancePiController
from test_depth_authority_250 import authority, advance, decide_commit
from test_distance_pi_controller import configured
from test_distance_tracking_response import setup
from test_lateral_zero_runtime import owner
from test_live_authority_binding import bind_production_reader


def controller():
    return DistancePiController(DistancePiConfig(
        kp_per_sec=3., launch_request_rpm=180., launch_full_error_m=.5,
        physical_ttl_sec=.25, motion_memory_sec=.35))


def observe(c, *, distance=2., raw=1.93, stamp=100., age=.03,
            ego=140., rate=-1.21185, valid=True, **changes):
    return c.update(distance, **dict(dict(
        target_distance_m=1.4, sample_timestamp=stamp, execution_now=stamp+age,
        deadband_m=.03, max_output_rpm=200., rise_rpm_per_sec=240.,
        fall_rpm_per_sec=300., ego_forward_rpm=ego, range_rate_m_s=rate,
        raw_closure_valid=valid, raw_distance_m=raw), **changes))


def braking(c):
    first = observe(c, distance=2.184, raw=2.011, ego=142.5, rate=-1.72094)
    assert first.output_rpm > 36
    assert c.accept_output_limit(first.sample_timestamp, 36.)
    return first


def test_cap233_raw_distance_closes_filtered_brake_budget_without_changing_p():
    new = observe(controller(), distance=1.7868, raw=1.6749128, ego=99.5,
                  rate=-1.3545, valid=False, age=.0398)
    old = observe(controller(), distance=1.7868, raw=None, ego=99.5,
                  rate=-1.3545, valid=False, age=.0398)
    assert old.cap_rpm == pytest.approx(11.74, abs=.03)
    assert new.cap_rpm == new.output_rpm == 0
    assert new.error_m == old.error_m and new.p_m_s == old.p_m_s
    assert new.pi_demand_rpm == old.pi_demand_rpm
    assert new.braking_distance_input_m == pytest.approx(1.6749128)


@pytest.mark.parametrize('raw', [2., 2.3, None, float('nan'), float('inf'), -.1, 0.])
def test_farther_or_unavailable_raw_never_adds_braking_space(raw):
    result = observe(controller(), raw=raw)
    baseline = observe(controller(), raw=None)
    assert result.braking_distance_input_m == baseline.braking_distance_input_m == 2.
    assert result.cap_rpm == baseline.cap_rpm


def test_cap230_zero_then_cap233_closer_sample_cannot_reopen():
    c = controller()
    first = observe(c, distance=1.819, raw=1.7868, stamp=5550.753076213,
                    age=.091, ego=105.5, rate=-1.4362, valid=False)
    assert first.output_rpm == 0
    second = observe(c, distance=1.7868, raw=1.6749128, stamp=5550.888191132,
                     age=.0398, ego=99.5, rate=-1.3545, valid=False)
    assert second.status == 'tracking'
    assert second.output_rpm == second.cap_rpm == 0


def test_cap327_rebound_uses_final_approval_not_previous_pi_request():
    c = controller()
    first = braking(c)
    r = observe(c, stamp=100.15)
    without_history = observe(controller(), stamp=100.15)
    assert without_history.cap_rpm > first.output_rpm > 36
    assert r.output_rpm == r.cap_rpm == r.brake_settling_anchor_rpm == 36
    assert r.brake_settling_limited and r.demand_limit_reason == 'brake_settling'
    assert r.integral_frozen


def test_brake_settling_never_holds_up_a_tighter_stop():
    c = controller()
    braking(c)
    r = observe(c, stamp=100.15, distance=1.6, raw=1.51, rate=-2.)
    assert r.output_rpm == r.cap_rpm == 0


@pytest.mark.parametrize('case', ['caught_up', 'away', 'flat', 'quantization'])
def test_resolved_braking_or_actual_nonclosing_target_can_accelerate(case):
    c = controller()
    braking(c)
    kwargs = dict(stamp=100.15)
    if case == 'caught_up': kwargs.update(ego=36., rate=-.1)
    if case == 'quantization': kwargs.update(ego=38., rate=-.1)
    if case == 'away': kwargs.update(raw=2.03, distance=2.03, rate=.1)
    if case == 'flat': kwargs.update(raw=2.011, distance=2.011, rate=0.)
    r = observe(c, **kwargs)
    assert not r.brake_settling_limited
    assert r.output_rpm > 36


def test_old_positive_regression_cannot_override_new_raw_closure():
    c = controller()
    braking(c)
    r = observe(c, stamp=100.15, rate=.1)
    assert r.brake_settling_limited and r.output_rpm == 36


@pytest.mark.parametrize('case', ['expired', 'long_gap', 'reset', 'revoked', 'target'])
def test_old_settling_anchor_cannot_survive_real_execution_reset(case):
    c = controller()
    braking(c)
    stamp = 100.15
    kwargs = {}
    if case == 'expired': stamp = 100.26
    if case == 'long_gap': stamp = 100.4
    if case == 'reset': c.reset()
    if case == 'revoked': c.suspend(100.08, 'revoke', reset_execution=True)
    if case == 'target': kwargs['target_distance_m'] = 1.5
    r = observe(c, stamp=stamp, **kwargs)
    assert not r.brake_settling_limited and r.brake_settling_anchor_rpm == 0
    assert r.output_rpm > 36


@pytest.mark.parametrize('case', ['duplicate', 'older', 'jump', 'continuation', 'stale'])
def test_rejected_sample_does_not_move_raw_trend_anchor(case):
    c = controller()
    braking(c)
    kwargs = dict(stamp=100.1, raw=3.)
    if case == 'duplicate': kwargs['stamp'] = 100.
    if case == 'older': kwargs['stamp'] = 99.99
    if case == 'jump': kwargs['measurement_jump_clamped'] = True
    if case == 'continuation': kwargs['age'] = .19
    if case == 'stale': kwargs['age'] = .26
    result = observe(c, **kwargs)
    assert result.status in {'duplicate', 'out_of_order', 'measurement_jump',
                             'continuation_only', 'stale_sample'}
    assert c._last_raw_distance_m == 2.011


def test_production_controller_and_admission_keep_cap233_zero(authority, setup):
    a = authority
    _, a.controller, _ = configured(
        setup, target_distance_m=1.4, distance_pi_kp_per_sec=3.,
        distance_pi_launch_request_rpm=180., distance_pi_launch_full_error_m=.5,
        depth_longitudinal_sample_max_age_sec=.25)
    a.owner._follow_controller = a.controller
    bind_production_reader(a.owner)
    for cap, stamp, age, used, raw, ego in [
        (230, 5550.753076213, .091, 1.819, 1.7868, 105.5),
        (233, 5550.888191132, .0398, 1.7868, 1.6749128, 99.5),
    ]:
        advance(a, stamp+age)
        current = a.frame(used, rpm=ego, stamp=stamp, yaw=40., capture_frame_id=cap)
        current = replace(current, distance_state=replace(
            current.distance_state, raw_distance_m=raw))
        _, actions, _ = decide_commit(a, current)
        result = a.controller.last_distance_pid_result
        assert result.pi_braking_distance_input_m == pytest.approx(raw)
        assert result.output_rpm == 0
        assert not any(x.kind == 'forward' and x.speed_percent > 0 for x in actions)
        assert a.owner._fresh_depth_linear_snapshot(1) is None
