"""Distance/encoder control is invariant to the diagnostic target estimator."""
from dataclasses import replace

import pytest

from car_control_modular.distance_pi import DistancePiConfig, DistancePiController
from car_control_modular.longitudinal_approach import RawDepthMotionEvidence
from car_control_modular.sample_braking import SampleBrakingAssessment
from car_control_modular.steering_pid import DistancePidConfig, LongitudinalDistancePid


def controller(**options):
    values = dict(kp_per_sec=3., ki_per_sec2=.4, physical_ttl_sec=.25,
                  deceleration_m_s2=.7, launch_request_rpm=180., launch_full_error_m=.5,
                  motion_memory_sec=.35, use_target_motion=False)
    values.update(options)
    return DistancePiController(DistancePiConfig(**values))


def update(pi, *, stamp=100., age=.04, distance=2.5, wheel=20., **options):
    args = dict(sample_timestamp=stamp, execution_now=stamp+age, deadband_m=.03,
                max_output_rpm=200., rise_rpm_per_sec=240., fall_rpm_per_sec=300.,
                ego_forward_rpm=wheel, preview_outer_forward_rpm=wheel,
                preview_feedback_timestamp=stamp+age-.01, raw_distance_m=distance)
    args.update(options)
    return pi.update(distance, 1.4, **args)


@pytest.mark.parametrize("rate", [-100., -3., -.5, 0., .5, 3., 100., None,
                                  float("nan"), float("inf"), object()])
@pytest.mark.parametrize("memory_sec", [0., .35])
def test_estimator_values_windows_and_resets_do_not_change_pi_or_recovery(rate, memory_sec):
    actual = controller(motion_memory_sec=memory_sec)
    baseline = controller(motion_memory_sec=0.)
    for i, (distance, wheel) in enumerate([(2.5, 20.), (2.51, 24.), (2.52, 28.),
            (2.50, 28.), (2.55, 20.), (2.7, 24.), (1.44, 20.), (1.4, 5.)]):
        stamp = 100.+i*.1
        if i == 3:
            for pi in (actual, baseline):
                pi.suspend(stamp-.02, "no_live_grant_before_pi", reset_execution=True)
        if i == 5:
            for pi in (actual, baseline):
                pi.accept_output_limit(pi._last_sample_ts, 5.)
        # Poisoned/outdated estimator state must have no effect on execution.
        actual._motion_memory = (stamp-.1, -99., 99., .1)
        actual._brake_recovery_pending = True
        actual.invalidate_motion_memory()
        result = update(actual, stamp=stamp, distance=distance, wheel=wheel,
            range_rate_m_s=rate, raw_closure_valid=True, allow_motion_memory=True,
            allow_motion_memory_endpoint_fallback=True, motion_memory_rotation_bound=float("nan"),
            raw_motion_evidence=RawDepthMotionEvidence(stamp-1, -.8, -2., .1, 2))
        expected = update(baseline, stamp=stamp, distance=distance, wheel=wheel)
        assert result == expected
        assert actual.integral_m_s == baseline.integral_m_s
        assert result.brake_source == "ego_distance"
        assert not result.motion_window_used and not result.brake_recovery_limited
        assert result.effective_range_rate_m_s == result.target_velocity_bound_m_s == 0


def test_learned_integral_at_setpoint_does_not_require_matching_person_speed():
    pi = controller(launch_request_rpm=0.)
    update(pi, stamp=99.9, distance=1.5, wheel=10.)
    pi.integral_m_s = .15
    result = update(pi, distance=1.4, wheel=10.)
    assert result.output_rpm > 0
    assert result.cap_rpm > result.output_rpm
    assert result.integral_m_s > 0
    assert result.stationary_preview_status == "bounded"


def test_physical_brake_is_mandatory_even_when_legacy_preview_switch_is_false():
    enabled, disabled = controller(stationary_stop_preview_enabled=True), controller()
    result = update(disabled, distance=1.8, wheel=80.)
    assert result == update(enabled, distance=1.8, wheel=80.)
    assert result.output_rpm == 0 and result.stationary_preview_status == "momentum_brake"


@pytest.mark.parametrize("change", [
    {"preview_outer_forward_rpm": None}, {"preview_feedback_timestamp": 99.8},
    {"preview_feedback_timestamp": 100.05}, {"ego_forward_rpm": None},
    {"raw_distance_m": None}, {"preview_completed_rpm": float("nan")},
])
def test_missing_or_stale_physical_inputs_cannot_be_replaced_by_target_estimates(change):
    result = update(controller(), range_rate_m_s=2., raw_closure_valid=True, **change)
    assert result.output_rpm == 0


def sample():
    return SampleBrakingAssessment(1, 100., 100.04, 2.5, 20., 20., 100.03,
        0., 1.2, .816814, .7, .2, 200., outer_allowance_rpm=10.)


@pytest.mark.parametrize("bad", ["target_speed", "stamp", "feedback", "distance", "object", "model"])
def test_supplied_shared_assessment_must_be_exact_and_target_motion_free(bad):
    evidence = sample()
    changes = dict(target_speed={"target_speed_m_s": -.1}, stamp={"sample_timestamp": 99.99},
                   feedback={"feedback_timestamp": 100.02}, distance={"distance_m": 2.6},
                   model={"deceleration_m_s2": 1.})
    evidence = object() if bad == "object" else replace(evidence, **changes[bad])
    result = update(controller(), braking_assessment=evidence)
    assert result.output_rpm == 0
    assert result.stationary_preview_status == "invalid_shared_assessment"


def test_exact_zero_target_shared_assessment_is_used_not_a_second_relative_cap():
    evidence = sample()
    result = update(controller(), braking_assessment=evidence, range_rate_m_s=-99.)
    assert result.braking_assessment is evidence
    assert result.cap_rpm == evidence.budget(evidence.checked_at, 20.).cap_rpm
    assert result.output_rpm > 0


def test_current_own_wheel_budget_releases_only_a_completed_brake_response():
    pi = controller()
    assert update(pi, distance=1.8, wheel=80.).output_rpm == 0
    assert update(pi, stamp=100.1, distance=1.81, wheel=80.).output_rpm == 0
    recovered = update(pi, stamp=100.2, distance=3.0, wheel=20.)
    assert recovered.output_rpm > 0 and not recovered.brake_settling_limited


def test_fresh_recovery_does_not_require_target_distance_to_be_increasing():
    pi = controller()
    update(pi, stamp=99.9, distance=2.4, wheel=20.)
    pi.suspend(100., "no_live_grant_before_pi", reset_execution=True)
    result = update(pi, stamp=100.05, distance=2.3, wheel=0., fresh_grant_recovery_step_sec=.05)
    assert result.output_rpm > 0 and result.fresh_grant_recovery_used


@pytest.mark.parametrize("age,status", [(.181, "continuation_only"), (.251, "stale_sample")])
def test_physical_depth_deadlines_remain_independent_of_target_estimates(age, status):
    pi = controller()
    result = update(pi, age=age, range_rate_m_s=2., raw_closure_valid=True)
    assert result.output_rpm == 0 and result.status == status


def test_close_raw_sample_wins_over_far_filtered_distance():
    result = update(controller(), distance=2.5, raw_distance_m=1.15, wheel=10.)
    assert result.output_rpm == 0


def test_wrapper_ignores_invalid_target_feedforward_and_derivative():
    wrapped = LongitudinalDistancePid(DistancePidConfig(
        max_forward_output_rpm=200., pi_profile=controller().config))
    result = wrapped.update(2.5, 1.4, now=100., execution_now=100.04,
        ego_forward_rpm=20., preview_outer_forward_rpm=20., preview_feedback_timestamp=100.03,
        braking_raw_distance_m=2.5, tracking_base_rpm=float("nan"),
        braking_range_rate_m_s=float("nan"), raw_closure_valid=True,
        raw_motion_evidence=object(), motion_memory_rotation_bound=float("nan"))
    assert result.output_rpm > 0
    assert result.error_rate_m_s == result.d_rpm == result.tracking_base_rpm == 0
    assert result.pi_brake_source == "ego_distance"


def test_only_boolean_mode_switch_is_accepted():
    with pytest.raises(ValueError):
        controller(use_target_motion="false")
