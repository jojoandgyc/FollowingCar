"""New physical braking evidence may resolve a model zero, not a safety STOP.

Recorded CAP119/122 depth samples from run_20261008_003319. Camera captures and
Depth samples are asynchronous: two of the updates belong to CAP122. No I/O.
"""
from dataclasses import replace

import pytest

from car_control_modular.distance_pi import DistancePiConfig, DistancePiController
from car_control_modular.sample_braking import SampleBrakingAssessment


SAMPLES = (
    # PI check times reconstructed from feedback timestamp + logged age
    # (rounded to 0.1ms), not the earlier ranging-stage sample age.
    (17362.237025417, 17362.286380853+.0217-17362.237025417,
     2.2371695623896684, 2.183, 83.5, 85., 85., 17362.286380853),
    (17362.329838649, 17362.343299593+.0354-17362.329838649,
     2.183, 2.0974, 76.5, 81., 81., 17362.343299593),
    (17362.397264993, 17362.484638782+.0305-17362.397264993,
     2.0974, 2.0525589203423307, 60.5, 61., 76., 17362.484638782),
)


def controller(**changes):
    return DistancePiController(DistancePiConfig(**dict(dict(
        kp_per_sec=3., physical_ttl_sec=.25, use_target_motion=False,
        deceleration_m_s2=1., response_delay_sec=.15,
        stationary_stop_preview_distance_m=1.1,
        launch_request_rpm=180., launch_full_error_m=.5), **changes)))


def arguments(row, **changes):
    stamp, age, distance, raw, ego, outer, bound, feedback = row
    kwargs = dict(actual_distance_m=distance, target_distance_m=1.4,
        sample_timestamp=stamp, execution_now=stamp+age, deadband_m=.03,
        max_output_rpm=200., rise_rpm_per_sec=240., fall_rpm_per_sec=300.,
        ego_forward_rpm=ego, preview_outer_forward_rpm=outer,
        preview_feedback_timestamp=feedback, raw_distance_m=raw)
    kwargs.update(changes)
    kwargs["braking_assessment"] = SampleBrakingAssessment(
        1, stamp, stamp+age, min(distance, raw), bound, outer, feedback,
        0., 1.1, .816814, 1., .15, 200., outer_allowance_rpm=10.)
    return kwargs


def prior_model_stop(pi):
    for row in SAMPLES[:2]:
        result = pi.update(**arguments(row))
        assert result.stationary_preview_status == "momentum_brake"
        assert result.output_rpm == result.cap_rpm == 0
        assert result.stationary_preview_required_stop_m > result.stationary_preview_margin_m
    return result


def test_recorded_three_samples_keep_real_momentum_stop_then_release_resolved_zero():
    pi = controller()
    stopped = prior_model_stop(pi)
    result = pi.update(**arguments(SAMPLES[2]))
    assert result.stationary_preview_cap_rpm == pytest.approx(63.8, abs=.02)
    assert result.stationary_preview_required_stop_m < result.stationary_preview_margin_m
    assert result.brake_settling_preview_released
    assert not result.brake_settling_limited
    assert result.pre_settling_cap_rpm == result.stationary_preview_cap_rpm
    assert 0 < result.output_rpm <= result.cap_rpm == 60.5
    assert result.output_rpm <= 240*(SAMPLES[2][0]+SAMPLES[2][1]
                                    - SAMPLES[1][0]-SAMPLES[1][1])
    assert result.execution_anchor_rpm == stopped.output_rpm == 0
    assert not result.motion_window_used
    assert result.target_velocity_bound_m_s == result.effective_range_rate_m_s == 0


@pytest.mark.parametrize("rise", [0., 60., 240., 1000.])
@pytest.mark.parametrize("launch_full_error", [0., .5])
def test_resolved_stop_cannot_launch_above_measured_speed_even_with_no_rise_limit(rise, launch_full_error):
    pi = controller(launch_full_error_m=launch_full_error)
    prior_model_stop(pi)
    result = pi.update(**arguments(SAMPLES[2], rise_rpm_per_sec=rise))
    assert result.brake_settling_preview_released
    assert 0 < result.output_rpm <= 60.5


def test_following_fresh_samples_continue_normal_ramp_and_new_momentum_stops_immediately():
    pi = controller()
    prior_model_stop(pi)
    recovered = pi.update(**arguments(SAMPLES[2]))
    last_output = recovered.output_rpm
    last_execution = SAMPLES[2][0]+SAMPLES[2][1]
    for i in range(1, 5):
        stamp = SAMPLES[2][0]+.1*i
        row = (stamp, .1175, 2.0974, 2.0525589203423307,
               60.5, 61., 76., stamp+.09)
        result = pi.update(**arguments(row))
        assert not result.brake_settling_limited
        assert not result.brake_settling_preview_released
        assert last_output <= result.output_rpm <= result.stationary_preview_cap_rpm
        assert result.output_rpm <= last_output+240*(stamp+.1175-last_execution)
        last_output, last_execution = result.output_rpm, stamp+.1175
    stamp = SAMPLES[2][0]+.5
    stopped = pi.update(**arguments((stamp, .03, 2., 1.6, 85., 85., 85., stamp+.02)))
    assert stopped.stationary_preview_status == "momentum_brake"
    assert stopped.output_rpm == 0
    assert not stopped.brake_settling_preview_released


@pytest.mark.parametrize("epsilon", [0., .0001])
def test_extra_five_rpm_margin_removed_but_exact_physical_cap_remains_binding(epsilon):
    pi = controller()
    prior_model_stop(pi)
    row = list(SAMPLES[2]); row[5] = 64.
    kwargs = arguments(row)
    cap = kwargs["braking_assessment"].budget(kwargs["execution_now"], 64.).cap_rpm
    kwargs["ego_forward_rpm"] = cap+epsilon
    result = pi.update(**kwargs)
    if epsilon == 0:
        assert result.output_rpm > 0 and result.brake_settling_preview_released
    else:
        assert result.output_rpm == 0 and not result.brake_settling_preview_released


@pytest.mark.parametrize("bad", ["new_momentum", "cap_below_wheels", "close_raw",
    "stale_sample", "continuation_only", "duplicate", "older", "jump",
    "stale_feedback", "assessment_mismatch", "missing_ego", "ego_above_outer"])
def test_new_evidence_must_independently_allow_current_motion(bad):
    pi = controller()
    prior_model_stop(pi)
    kwargs = arguments(SAMPLES[2])
    if bad == "new_momentum":
        row = list(SAMPLES[2]); row[4:7] = [85., 85., 85.]
        kwargs = arguments(row)
    elif bad == "cap_below_wheels":
        # Still a positive physical budget, but it does NOT support the
        # measured common velocity. The previous zero must remain binding.
        row = list(SAMPLES[2]); row[4:7] = [66., 66., 76.]
        kwargs = arguments(row)
    elif bad == "close_raw":
        row = list(SAMPLES[2]); row[3] = 1.2
        kwargs = arguments(row)
    elif bad in ("stale_sample", "continuation_only"):
        kwargs["execution_now"] = kwargs["sample_timestamp"] + (.251 if bad == "stale_sample" else .181)
    elif bad in ("duplicate", "older"):
        kwargs["sample_timestamp"] = SAMPLES[1][0] - (.01 if bad == "older" else 0)
        kwargs["execution_now"] = SAMPLES[1][0] + .10
    elif bad == "jump": kwargs["measurement_jump_clamped"] = True
    elif bad == "stale_feedback": kwargs["preview_feedback_timestamp"] = kwargs["execution_now"]-.151
    elif bad == "assessment_mismatch":
        kwargs["braking_assessment"] = replace(kwargs["braking_assessment"], distance_m=2.2)
    elif bad == "missing_ego": kwargs["ego_forward_rpm"] = None
    elif bad == "ego_above_outer": kwargs["ego_forward_rpm"] = 62.
    result = pi.update(**kwargs)
    assert result.output_rpm == 0
    assert not result.brake_settling_preview_released


@pytest.mark.parametrize("event", ["stop", "rejected", "parking", "expired", "long_gap"])
def test_new_release_does_not_bypass_execution_lifecycle(event):
    pi = controller()
    prior_model_stop(pi)
    kwargs = arguments(SAMPLES[2])
    if event == "stop": pi.suspend(SAMPLES[1][0]+.06, "explicit_stop", reset_execution=True)
    elif event == "rejected": pi.reject_output(SAMPLES[1][0])
    elif event == "parking": pi.set_normal_parking(True)
    else:
        shift = .18 if event == "expired" else .5
        row = list(SAMPLES[2]); row[0] += shift; row[7] += shift
        kwargs = arguments(row)
    result = pi.update(**kwargs)
    # The ordinary controller may preview a measured-speed request while
    # suspended/parked; this branch supplies no release or extra ramp credit.
    # Actual UID/STOP/deadline authority remains owned by the runtime.
    assert not result.brake_settling_preview_released
    assert not result.depth_expiry_recovery_used
    assert not result.fresh_grant_recovery_used


def test_downstream_zero_cannot_claim_it_was_a_model_momentum_stop():
    pi = controller()
    row = (100., .03, 2.2, 2.2, 60.5, 61., 76., 100.02)
    first = pi.update(**arguments(row, rise_rpm_per_sec=0.))
    assert first.output_rpm > 0
    pi.accept_output_limit(first.sample_timestamp, 0.)
    row = (100.1, .1175, 2.0974, 2.0525589203423307, 60.5, 61., 76., 100.2)
    result = pi.update(**arguments(row))
    assert result.output_rpm == 0
    assert not result.brake_settling_preview_released
