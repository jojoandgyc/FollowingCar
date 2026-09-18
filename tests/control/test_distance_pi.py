"""Pure forward-control regressions: no hardware and no authorization owner."""
from dataclasses import replace
from collections import deque
import math

import pytest

from car_control_modular.distance_pi import DistancePiConfig, DistancePiController
from car_control_modular.steering_pid import DistancePidConfig, LongitudinalDistancePid


SCALE = 60./.816814


def controller(**kwargs):
    return LongitudinalDistancePid(DistancePidConfig(
        pi_profile=DistancePiConfig(), deadband_m=.03,
        max_forward_output_rpm=200., **kwargs))


def sample(c, t, distance=1.8, ego=60., rate=0., **kwargs):
    return c.update(distance, 1.5, now=t, execution_now=t,
                    ego_forward_rpm=ego, braking_range_rate_m_s=rate,
                    raw_closure_valid=True, **kwargs)


def learn_speed(c, count=41):
    for index in range(count):
        result = sample(c, 100.+index*.1)
    assert result.i_rpm > 0
    return 100.+(count-1)*.1, result


def test_integral_holds_matching_speed_at_zero_error_without_feedforward():
    c = controller()
    t, learned = learn_speed(c)
    r = sample(c, t+.1, distance=1.5, ego=learned.i_rpm, tracking_base_rpm=80.)
    assert r.approach_mode == "distance_pi"
    assert r.p_rpm == r.d_rpm == r.tracking_base_rpm == 0
    assert r.i_rpm == pytest.approx(learned.i_rpm)
    assert r.output_rpm == math.floor(r.i_rpm+1e-9) > 20
    assert r.pi_brake_source == "raw_relative_motion"


def test_stopping_person_removes_motion_before_clearing_integral_at_standstill():
    c = controller(output_fall_rpm_per_sec=1.)
    t, learned = learn_speed(c)
    r = sample(c, t+.1, distance=1.5, ego=learned.i_rpm,
               rate=-learned.i_rpm/SCALE)
    assert r.output_rpm == 0  # brake is stronger than comfort fall slew
    assert 0 < r.i_rpm < learned.i_rpm
    for index in range(3):
        r = sample(c, t+.2+index*.05, distance=1.5, ego=0., rate=0.)
    assert r.pi_status == "stationary"
    assert r.i_rpm == r.output_rpm == 0


def test_cap164_recovery_uses_actual_wheel_speed_not_artificial_24rpm_launch():
    c = controller(output_rise_rpm_per_sec=240., output_fall_rpm_per_sec=300.)
    first = sample(c, 100., distance=2.4, ego=62.)
    assert first.output_rpm == 62
    c.suspend(100.20, "expired_authority", retain=True, reset_execution=True)
    resumed = sample(c, 100.25, distance=2.4, ego=58.)
    assert resumed.output_rpm == 58
    assert resumed.pi_sample_dt_sec == 0
    following = sample(c, 100.30, distance=2.4, ego=58.)
    assert 58 < following.output_rpm <= 70


def test_physical_dt_is_not_invented_or_rounded_to_thirty_milliseconds():
    c = controller()
    first = sample(c, 100.)
    assert first.i_rpm == first.pi_sample_dt_sec == 0
    second = sample(c, 100.005)
    assert second.pi_sample_dt_sec == pytest.approx(.005)
    assert second.pi_integral_m_s == pytest.approx(.4*.27*.005)


def test_duplicate_changed_range_and_out_of_order_never_integrate_or_advance_state():
    c = controller(max_measurement_jump_m=.8)
    sample(c, 100.)
    r = sample(c, 100.1)
    changed_duplicate = sample(c, 100.1, distance=5.)
    assert changed_duplicate.actual_distance_m == r.actual_distance_m
    assert changed_duplicate.i_rpm == r.i_rpm
    assert changed_duplicate.pi_sample_dt_sec == 0
    old = sample(c, 100.05, distance=1.6)
    assert old.output_rpm == 0 and old.pi_status == "out_of_order"
    new = sample(c, 100.15)
    assert new.pi_sample_dt_sec == pytest.approx(.05)
    assert not new.measurement_jump_clamped


def test_short_gap_retains_integral_but_does_not_integrate_blind_interval():
    c = controller()
    t, learned = learn_speed(c)
    r = sample(c, t+.25)
    assert r.pi_integral_m_s == learned.pi_integral_m_s
    assert r.pi_sample_dt_sec == 0
    assert sample(c, t+.30).pi_integral_m_s > learned.pi_integral_m_s


def test_long_gap_clears_integral_and_never_catches_up_missing_time():
    c = controller()
    t, learned = learn_speed(c)
    r = sample(c, t+.36)
    assert r.pi_status == "long_gap_reset"
    assert r.pi_integral_m_s == r.pi_sample_dt_sec == 0


def test_stale_and_suspended_duplicates_never_regrant_cached_speed():
    c = controller()
    sample(c, 100.)
    sample(c, 100.1)
    stale = c.update(1.8, 1.5, now=100.1, execution_now=100.281,
                     ego_forward_rpm=60., braking_range_rate_m_s=0., raw_closure_valid=True)
    assert stale.output_rpm == 0 and stale.pi_status == "stale_sample"
    assert sample(c, 100.1).output_rpm == 0


def test_empty_tick_does_not_reset_execution_or_add_integral():
    c = controller(output_rise_rpm_per_sec=240.)
    first = sample(c, 100., distance=2.4, ego=45.)
    c.suspend(100.02, "no_new_depth")
    duplicate = sample(c, 100., distance=2.4, ego=45.)
    assert duplicate.output_rpm == first.output_rpm
    next_r = sample(c, 100.05, distance=2.4, ego=45.)
    assert next_r.output_rpm == first.output_rpm+12


def test_final_output_limits_are_monotonic_and_idempotent_per_sample():
    c = controller()
    t, learned = learn_speed(c)
    assert c.accept_output_limit(t, 20.)
    integral20 = c._distance_pi.integral_m_s
    assert not c.accept_output_limit(t, 20.)
    assert not c.accept_output_limit(t, 30.)
    assert c._distance_pi.integral_m_s == integral20
    assert c.accept_output_limit(t, 10.)
    assert c._distance_pi.integral_m_s <= integral20
    assert c._distance_pi.integral_m_s == pytest.approx(10./SCALE)
    assert not c.accept_output_limit(t-.1, 0.)
    duplicate = sample(c, t)
    assert duplicate.output_rpm == 10
    assert duplicate.i_rpm == pytest.approx(10.)


def test_rejected_sample_rolls_back_increment_without_erasing_old_compensation():
    c = controller()
    t, learned = learn_speed(c)
    sample(c, t+.1)
    assert c.reject_output(t+.1)
    assert c._distance_pi.integral_m_s == pytest.approx(learned.pi_integral_m_s)
    assert not c.reject_output(t+.1)
    c.accept_output_limit(t+.1, 5.)
    reduced = c._distance_pi.integral_m_s
    assert not c.reject_output(t+.1)
    assert c._distance_pi.integral_m_s == reduced


@pytest.mark.parametrize("rejected_execution", [10.21, 10.24])
def test_rejected_request_without_integral_increment_never_becomes_slew_anchor(rejected_execution):
    c = controller(output_rise_rpm_per_sec=240.)

    def observe(stamp, execution, ego):
        return c.update(3., 1.5, now=stamp, execution_now=execution,
                        ego_forward_rpm=ego, braking_range_rate_m_s=0., raw_closure_valid=True)

    assert observe(10., 10., 0.).output_rpm == 0
    assert observe(10.05, 10.05, 0.).output_rpm == 12
    c.accept_output_limit(10.05, 12.)
    rejected = observe(10.10, rejected_execution, 12.)
    assert rejected.i_rpm == 0
    if rejected_execution == 10.21:
        assert rejected.output_rpm > 24  # request, never an issued command
    assert c.reject_output(10.10)
    assert not c.reject_output(10.10)
    # An old live grant may still accept a tighter positive braking command.
    assert c.accept_output_limit(10.10, 5.)
    assert c._distance_pi._execution_suspended
    assert observe(10.10, rejected_execution, 12.).output_rpm == 0
    resumed = observe(10.25, 10.27, 12.)
    assert resumed.pi_status == "recovering"
    assert resumed.pi_sample_dt_sec == 0
    assert resumed.output_rpm == 12


def test_execution_crossing_old_physical_deadline_forces_recovery_without_explicit_pause():
    c = controller(output_rise_rpm_per_sec=240.)

    def observe(stamp, execution, ego):
        return c.update(3., 1.5, now=stamp, execution_now=execution,
                        ego_forward_rpm=ego, braking_range_rate_m_s=0., raw_closure_valid=True)

    assert observe(9.95, 9.95, 60.).output_rpm == 60
    assert observe(10., 10., 60.).output_rpm == 72
    c.accept_output_limit(10., 72.)
    # A new sample is <180ms after the old capture, but processing happens
    # 115ms after that old sample's grant expired. No suspend() was delivered.
    resumed = observe(10.17, 10.295, 0.)
    assert resumed.pi_status == "recovering"
    assert resumed.pi_sample_dt_sec == 0
    assert resumed.output_rpm == 0


def test_percent_quantization_does_not_permanently_prevent_integral_learning():
    c = controller()
    values = []
    for index in range(50):
        stamp = 100.+index*.1
        r = sample(c, stamp)
        c.accept_output_limit(stamp, 2*math.floor(r.output_rpm/2))
        values.append(c._distance_pi.integral_m_s)
    assert values[-1] > .45
    assert values[-1] > values[20] > values[5] > 0


def test_missing_ego_uses_explicit_encoder_bound_without_inventing_max_speed():
    c = controller()
    r = sample(c, 100., distance=2.0, ego=None, rate=-.1)
    assert r.approach_closing_m_s == pytest.approx(.1)
    assert r.output_rpm > 0
    assert r.pi_brake_source == "stationary_fallback"
    t, learned = learn_speed(c)
    missing = c.update(2., 1.5, now=t+.1)
    assert missing.output_rpm == 0 and missing.pi_brake_source == "missing_motion_evidence"
    assert missing.i_rpm == learned.i_rpm


@pytest.mark.parametrize("ego,raw", [(None, True), (40., False)])
def test_unreliable_relative_motion_cannot_exempt_integral_from_stationary_bound(ego, raw):
    c = controller()
    t, learned = learn_speed(c)
    r = c.update(1.5, 1.5, now=t+.1, ego_forward_rpm=ego,
                 braking_range_rate_m_s=0., raw_closure_valid=raw)
    assert r.output_rpm == 0 and r.pi_brake_source == "stationary_fallback"


def test_future_command_closure_is_limited_even_if_present_range_is_not_closing():
    c = controller(kp_rpm_per_m=999.)
    t, learned = learn_speed(c)
    r = sample(c, t+.1, distance=1.5, ego=10., rate=0.)
    assert r.output_rpm <= 10 < learned.i_rpm


def test_approaching_person_does_not_get_clipped_to_a_stationary_target():
    c = controller()
    r = sample(c, 100., distance=1.7, ego=10., rate=-.7)
    assert r.output_rpm == r.approach_cap_rpm == 0


def test_moving_target_converges_without_speed_feedforward_in_ideal_plant():
    c = controller(output_rise_rpm_per_sec=240., output_fall_rpm_per_sec=300.)
    distance, ego = 1.8, 0.
    for index in range(500):
        r = sample(c, 100.+index*.05, distance=distance, ego=ego,
                   rate=.4-ego/SCALE)
        # Real final-percent quantization feeds back its applied limit.
        applied = 2*math.floor(r.output_rpm/2)
        c.accept_output_limit(100.+index*.05, applied)
        ego += max(-.4*SCALE*.05, min(.8*SCALE*.05, applied-ego))
        distance += (.4-ego/SCALE)*.05
    assert abs(distance-1.5) < .07
    assert ego/SCALE == pytest.approx(.4, abs=.03)
    assert c._distance_pi.integral_m_s > .3


def test_simplified_delayed_plant_tracks_then_stops_with_short_authority_gap():
    # Synthetic CLOSED loop: ideal raw range/encoder, 100ms physical samples,
    # 100ms actuator delay, acceleration0.8/deceleration0.4m/s², 2RPM commands.
    # These assumptions are not measured hardware limits or a real-run replay.
    c = controller(output_rise_rpm_per_sec=240., output_fall_rpm_per_sec=300.)
    distance, speed, command = 1.8, 0., 0.
    pending = deque()
    expires_at = -1.
    paused = True
    observed = []
    resumed = None
    integral_before_gap = None
    for tick in range(4001):
        now = tick*.01
        walking = .4 if now < 15. else max(0., .4-.2*(now-15.))
        if tick % 10 == 0 and tick not in {600, 610}:
            r = sample(c, 100.+now, distance=distance, ego=speed*SCALE,
                       rate=walking-speed)
            applied = 2*math.floor(r.output_rpm/2)
            c.accept_output_limit(100.+now, applied)
            pending.append((now+.1, applied/SCALE))
            expires_at, paused = now+.18, False
            observed.append((now, distance, speed, r.output_rpm, r.pi_integral_m_s))
            if tick == 590:
                integral_before_gap = c._distance_pi.integral_m_s
            if tick == 620:
                resumed = r
        if now > expires_at+1e-9:
            command = 0.
            pending.clear()
            if not paused:
                c.suspend(100.+now, "physical_authority_expired", reset_execution=True)
                paused = True
        while pending and pending[0][0] <= now+1e-9:
            _, command = pending.popleft()
        speed += max(-.4*.01, min(.8*.01, command-speed))
        distance += (walking-speed)*.01
    steady = [row for row in observed if 12. <= row[0] <= 14.]
    assert max(abs(row[1]-1.5) for row in steady) < .10
    assert min(row[4] for row in steady) > .2
    assert resumed.pi_sample_dt_sec == 0
    assert resumed.pi_integral_m_s == pytest.approx(integral_before_gap)
    final = [row for row in observed if row[0] >= 35.]
    assert all(row[3] == 0 and row[4] == 0 for row in final)
    assert speed == 0


@pytest.mark.parametrize("field", ["actual_distance_m", "target_distance_m", "now", "execution_now"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_sample_fields_fail_before_advancing_controller(field, bad):
    c = controller()
    sample(c, 100.)
    before = c._distance_pi._last_sample_ts
    values = dict(actual_distance_m=1.8, target_distance_m=1.5, now=100.1, execution_now=100.1)
    values[field] = bad
    with pytest.raises(ValueError):
        c.update(**values)
    assert c._distance_pi._last_sample_ts == before


def test_rejected_depth_jump_cannot_promote_itself_to_the_next_anchor():
    c = controller(max_measurement_jump_m=.8)
    sample(c, 100., distance=1.8)
    for t in (100.05, 100.10):
        r = sample(c, t, distance=5.)
        assert r.output_rpm == 0 and r.pi_status == "measurement_jump"
        assert c._distance_pi._last_sample_ts == 100.
        assert c.last_result.actual_distance_m == 1.8
    recovered = sample(c, 100.15, distance=1.81)
    assert recovered.pi_status == "recovering"
    assert recovered.pi_sample_dt_sec == 0
    assert recovered.output_rpm > 0


def test_pi_ignores_legacy_gains_launch_floor_and_feedforward_and_allows_over60():
    c = controller(kp_rpm_per_m=999., ki_rpm_per_m_s=999., min_forward_output_rpm=80.)
    first = sample(c, 100., distance=2.4, ego=80., tracking_base_rpm=80.)
    second = sample(controller(), 100., distance=2.4, ego=80., tracking_base_rpm=None)
    assert first.output_rpm == second.output_rpm > 60
    assert sample(c, 100.1, distance=1.54, ego=0.).output_rpm < 20


def test_explicit_reverse_keeps_legacy_pid_and_clears_forward_memory():
    c = controller()
    t, _ = learn_speed(c)
    reverse = c.update(1.2, 1.5, now=t+.1, forward_control=False)
    assert reverse.output_rpm < 0 and reverse.approach_mode == "legacy_pid"
    forward = sample(c, t+.2)
    assert forward.i_rpm == forward.pi_sample_dt_sec == 0


@pytest.mark.parametrize("field,value", [("kp_per_sec", 0.), ("ki_per_sec2", float("nan")),
                                         ("integral_max_m_s", -1.), ("physical_ttl_sec", .251),
                                         ("fresh_update_max_age_sec", .181),
                                         ("max_integration_gap_sec", .181),
                                         ("stationary_confirm_samples", 1)])
def test_invalid_config_rejected(field, value):
    with pytest.raises(ValueError):
        replace(DistancePiConfig(), **{field: value})


def extended_pi_sample(c, stamp, execution, *, distance=1.8, ego=30., rate=0.):
    return c.update(distance, 1.5, sample_timestamp=stamp, execution_now=execution,
                    deadband_m=.03, max_output_rpm=200., rise_rpm_per_sec=240.,
                    ego_forward_rpm=ego, range_rate_m_s=rate, raw_closure_valid=True)


@pytest.mark.parametrize("age,status", [(.180, "tracking"), (.200, "continuation_only"),
                                       (.250, "continuation_only"), (.251, "stale_sample")])
def test_extended_physical_ttl_keeps_independent_fresh_update_boundary(age, status):
    c = DistancePiController(DistancePiConfig(physical_ttl_sec=.25))
    extended_pi_sample(c, 100., 100.)
    extended_pi_sample(c, 100.05, 100.05)
    integral, old_stamp, old_execution = c.integral_m_s, c._last_sample_ts, c._last_execution_ts
    r = extended_pi_sample(c, 100.10, 100.10+age)
    if status == "tracking":
        # At the inclusive180ms freshness boundary, the prior100.30 deadline
        # has not yet passed and this is still a normal independent update.
        assert r.status == "tracking"
        assert c._last_sample_ts == 100.10
    else:
        assert r.status == status and r.output_rpm == r.sample_dt_sec == 0
        assert c._last_sample_ts == old_stamp and c._last_execution_ts == old_execution
        assert c.integral_m_s == integral


@pytest.mark.parametrize("age", [.200, .250])
def test_extended_late_sample_cannot_start_an_uninitialized_pi(age):
    c = DistancePiController(DistancePiConfig(physical_ttl_sec=.25))
    r = extended_pi_sample(c, 100., 100.+age, distance=3., ego=0.)
    assert r.status == "continuation_only" and r.output_rpm == 0
    assert c._last_sample_ts is c._last_execution_ts is c.last_result is None
    assert c.integral_m_s == 0


def test_extended_late_duplicates_and_new_timestamps_never_roll_the_pi_timeline():
    c = DistancePiController(DistancePiConfig(physical_ttl_sec=.25))
    first = extended_pi_sample(c, 100., 100., distance=3., ego=30.)
    saved = (c._last_sample_ts, c._last_execution_ts, c.integral_m_s, c._last_output_rpm)
    for stamp, execution in ((100., 100.20), (100., 100.25), (100.05, 100.26),
                             (100.10, 100.31)):
        r = extended_pi_sample(c, stamp, execution, distance=4., ego=80.)
        assert r.status == "continuation_only" and r.output_rpm == 0
        assert (c._last_sample_ts, c._last_execution_ts, c.integral_m_s,
                c._last_output_rpm) == saved
        assert c.last_result is first
    # The newer rejected timestamps never renewed the original100.25 grant.
    resumed = extended_pi_sample(c, 100.32, 100.33, distance=3., ego=12.)
    assert resumed.status == "recovering"
    assert resumed.sample_dt_sec == 0 and resumed.output_rpm == 12


def test_extended_ttl_keeps_live_ramp_but_does_not_integrate_the_missing_interval():
    c = DistancePiController(DistancePiConfig(physical_ttl_sec=.25))
    extended_pi_sample(c, 100., 100.)
    learned = extended_pi_sample(c, 100.05, 100.05)
    resumed = extended_pi_sample(c, 100.25, 100.25)
    assert resumed.status == "tracking_gap_no_integral" and resumed.sample_dt_sec == 0
    assert resumed.integral_m_s == learned.integral_m_s
    assert resumed.output_rpm <= 30
    next_sample = extended_pi_sample(c, 100.30, 100.30)
    assert next_sample.sample_dt_sec == pytest.approx(.05)
    assert next_sample.integral_m_s > resumed.integral_m_s


def test_extended_ttl_real_expiry_retains_rejection_and_recovery_protection():
    c = DistancePiController(DistancePiConfig(physical_ttl_sec=.25))
    extended_pi_sample(c, 100., 100., distance=3., ego=30.)
    stale = extended_pi_sample(c, 100., 100.251, distance=3., ego=30.)
    assert stale.status == "stale_sample" and c._execution_suspended
    recovered = extended_pi_sample(c, 100.26, 100.27, distance=3., ego=12.)
    assert recovered.status == "recovering" and recovered.output_rpm == 12
    assert recovered.sample_dt_sec == 0
    assert c.reject_output(100.26)
    assert extended_pi_sample(c, 100.26, 100.28).output_rpm == 0


def test_default_pi_still_expires_at_180ms():
    c = DistancePiController(DistancePiConfig())
    assert c.config.physical_ttl_sec == .18
    r = extended_pi_sample(c, 100., 100.20)
    assert r.status == "stale_sample" and r.output_rpm == 0
