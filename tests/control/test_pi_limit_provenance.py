"""CAP5240: distinguish request envelope from the tighter recovery ramp."""
import math
from dataclasses import replace

import pytest

from car_control_modular.distance_pi import DistancePiConfig, DistancePiController
from car_control_modular.controllers import FollowPolicyConfig, FollowSafetyController
from car_control_modular.steering_pid import DistancePidConfig, LongitudinalDistancePid
from test_cap233_brake_settling import braking, controller, observe


def test_recovery_does_not_masquerade_as_only_a_braking_limit():
    c = controller()
    r = observe(c, distance=1.8758, raw=1.9226, ego=30.5, rate=.1494, age=.019)
    assert r.status == "recovering"
    assert r.unslewed_output_rpm > r.output_rpm == 30
    assert r.demand_limit_reason == "braking_envelope"  # Legacy field retained.
    assert r.final_limit_reason == "execution_recovery"
    assert r.execution_anchor_rpm == r.ramp_output_rpm == 30.5
    assert r.execution_ramp_dt_sec == 0
    assert r.pre_quantization_rpm == 30.5
    assert r.brake_recovery_cap_rpm is None


def test_braking_zero_wins_even_when_recovering_from_high_measured_speed():
    r = observe(controller(), distance=1.45, raw=1.44, ego=90., rate=-1.4)
    assert r.cap_rpm == r.output_rpm == r.pre_quantization_rpm == 0
    assert r.final_limit_reason == "braking_envelope"


def test_ordinary_slew_tighter_than_braking_is_named():
    c = controller()
    observe(c, distance=2., raw=2., ego=0., rate=.3)
    r = observe(c, stamp=100.05, distance=2.02, raw=2.02, ego=0., rate=.3)
    assert r.output_rpm == 12
    assert r.cap_rpm > r.output_rpm
    assert r.final_limit_reason == "software_slew"
    assert r.execution_anchor_rpm == 0
    assert r.execution_ramp_dt_sec == pytest.approx(.05)


def test_settling_is_named_only_when_it_is_the_binding_limit():
    c = controller()
    braking(c)
    r = observe(c, stamp=100.15)
    assert r.final_limit_reason == "brake_settling"
    assert r.pre_settling_cap_rpm > r.cap_rpm == r.output_rpm == 36


def test_restored_evidence_limit_is_visible_separately():
    c = controller()
    observe(c, ego=60., rate=.2, distance=2.5, raw=2.5)
    c.invalidate_motion_memory()
    r = observe(c, stamp=100.05, ego=60., rate=.2, distance=2.5, raw=2.5,
                rise_rpm_per_sec=0.)
    assert r.brake_recovery_limited
    assert r.final_limit_reason == "brake_evidence_recovery"
    assert r.brake_recovery_cap_rpm == pytest.approx(72.)
    assert r.ramp_output_rpm > r.brake_recovery_cap_rpm


@pytest.mark.parametrize("case", ["duplicate", "older", "late", "revoked_duplicate"])
def test_trace_cannot_make_replayed_or_late_samples_update_state(case):
    c = controller()
    first = observe(c)
    if case == "revoked_duplicate":
        c.suspend(100.04, "revoked", reset_execution=True)
    before = (c._last_sample_ts, c._last_execution_ts, c.integral_m_s)
    kwargs = {"stamp": 100.}
    if case == "older": kwargs["stamp"] = 99.9
    if case == "late": kwargs.update(stamp=100.05, age=.19)
    result = observe(c, **kwargs)
    assert before == (c._last_sample_ts, c._last_execution_ts, c.integral_m_s)
    if case == "duplicate":
        assert result.final_limit_reason == first.final_limit_reason
    else:
        assert result.final_limit_reason == "not_evaluated"
        assert result.output_rpm == 0


def test_wrapper_propagates_the_exact_request_stage_trace():
    config = DistancePiConfig(kp_per_sec=3., launch_request_rpm=180.,
                              launch_full_error_m=.5, physical_ttl_sec=.25)
    wrapped = LongitudinalDistancePid(DistancePidConfig(
        pi_profile=config, output_rise_rpm_per_sec=240., output_fall_rpm_per_sec=300.))
    r = wrapped.update(1.8758, 1.4, now=100., measurement_age_sec=.019,
                       execution_now=100.019, ego_forward_rpm=30.5,
                       braking_range_rate_m_s=.1494, raw_closure_valid=True,
                       braking_raw_distance_m=1.9226)
    pure = wrapped._distance_pi.last_result
    for name in ("final_limit_reason", "pre_settling_cap_rpm", "execution_anchor_rpm",
                 "execution_ramp_dt_sec", "ramp_output_rpm", "brake_recovery_cap_rpm",
                 "pre_quantization_rpm"):
        assert getattr(r, "pi_" + name) == getattr(pure, name)
    assert r.output_rpm == pure.output_rpm


def test_controller_logs_formatted_binding_reason_without_claiming_motor_output(caplog):
    c = FollowSafetyController(FollowPolicyConfig(
        distance_pid_enable=True, distance_control_mode="distance_pi", target_distance_m=1.4,
        distance_pi_kp_per_sec=3., distance_pi_launch_request_rpm=180.,
        distance_pi_launch_full_error_m=.5,
        distance_pid_output_rise_rpm_per_sec=240.,
    ))
    c.active_target_id = 1
    c._distance_approach_sample_trusted = True
    c._distance_pid_sample_timestamp = 100.
    c._distance_pi_feedback_timestamp = 100.
    c._distance_pi_ego_forward_rpm = 30.5
    c._braking_range_rate = .1494
    c._braking_rate_source = "raw_depth_window"
    c._distance_pi_raw_distance_m = 1.9226
    with caplog.at_level("INFO"):
        result = c._update_distance_pid(1.8758, now=100.019)
        c._distance_pid_sample_timestamp = c._distance_pi_feedback_timestamp = 100.05
        c._distance_pi_raw_distance_m = 1.6
        c._braking_range_rate = -1.4
        c._update_distance_pid(1.6, now=100.069)
    lines = [record.getMessage() for record in caplog.records]
    initial = next(line for line in lines if line.startswith("distance_pi "))
    assert "final_limit_reason=execution_recovery" in initial
    assert "execution_anchor_rpm=30.50" in initial
    assert "limit_scope=pi_request" in initial
    assert result.output_rpm == 30
    transition = next(line for line in lines if line.startswith("distance_brake_transition "))
    assert "final_limit_reason=braking_envelope" in transition


@pytest.mark.parametrize("rise", [0., 120., 240.])
@pytest.mark.parametrize("launch", [0., 180.])
def test_pre_quantization_and_trace_do_not_change_rpm_limits(rise, launch):
    c = DistancePiController(replace(controller().config, launch_request_rpm=launch))
    for i, (distance, ego, rate) in enumerate(((2., 20., .2), (1.9, 40., -.2),
                                             (1.5, 40., -.5), (1.8, 10., .3))):
        r = observe(c, distance=distance, raw=distance, ego=ego, rate=rate,
                    stamp=100.+i*.06, rise_rpm_per_sec=rise)
        assert r.output_rpm == math.floor(r.pre_quantization_rpm+1e-9)
        assert r.pre_quantization_rpm <= r.cap_rpm
        assert r.pre_quantization_rpm <= r.ramp_output_rpm
