"""CAP270: expiry recovery is not evidence that measured wheels were braking."""
import pytest

from car_control_modular.distance_pi import DistancePiConfig, DistancePiController
from car_control_modular.longitudinal_approach import RawDepthMotionEvidence


def observe(c, stamp, age, used, raw, ego, rate, target, span, *, expiry=0., rise=240.):
    return c.update(
        used, 1.4, sample_timestamp=stamp, execution_now=stamp+age,
        deadband_m=.03, max_output_rpm=200., rise_rpm_per_sec=rise,
        fall_rpm_per_sec=300., ego_forward_rpm=ego, range_rate_m_s=rate,
        raw_closure_valid=True, raw_distance_m=raw,
        raw_motion_evidence=RawDepthMotionEvidence(stamp, rate, target, span, 2),
        depth_expiry_recovery_step_sec=expiry)


def test_recorded_expiry_then_accelerating_wheels_do_not_latch_34_rpm():
    c = DistancePiController(DistancePiConfig(
        kp_per_sec=3., physical_ttl_sec=.25, deceleration_m_s2=.7,
        launch_request_rpm=180., launch_full_error_m=.5, motion_memory_sec=.35))
    # Last pre-expiry sample (CAP265), with its actual final approval.
    observe(c, 1970.249615809, .099, 2.7299, 2.7429515757, 16.,
            .3724, .5902, .10, rise=0.)
    c.accept_output_limit(1970.249615809, 96.)
    c.suspend(1970.5202, "physical_depth_expired", reset_execution=True)
    recovery = observe(c, 1970.493534444, .0267, 2.7429515757, 2.8156613679,
                       23., .2700017051, .5354662551, .243918635, expiry=.05)
    assert recovery.output_rpm == 35
    assert recovery.final_limit_reason == "depth_expiry_recovery_step"
    assert recovery.pre_settling_cap_rpm == pytest.approx(127.95, abs=.02)
    c.accept_output_limit(recovery.sample_timestamp, 34., quantization_rpm=2.)
    requests = []
    for values in [
        (1970.550691318, .0542, 2.8061887386, 2.8061887386, 41.5,
         -.1870562961, .2519812289, .057156874),
        (1970.616003, .0606, 2.8061887386, 2.7878069255, 48.,
         -.2545782387, .2770869794, .122468556),
        (1970.664150501, .0951, 2.7878069255, 2.7740501743, 48.,
         -.2761082937, .2927161066, .170616057),
    ]:
        r = observe(c, *values)
        assert not r.brake_settling_limited
        assert r.output_rpm <= r.cap_rpm
        assert r.output_rpm <= c.last_result.execution_anchor_rpm+240*r.execution_ramp_dt_sec
        c.accept_output_limit(r.sample_timestamp, 2*(r.output_rpm//2), quantization_rpm=2.)
        requests.append(r.output_rpm)
    assert requests[0] > 34 and requests == sorted(requests)


def test_downstream_real_brake_still_owns_settling_after_recovery():
    c = DistancePiController(DistancePiConfig(physical_ttl_sec=.25))
    first = observe(c, 100., .02, 2.8, 2.8, 48., -.1, .55, .1)
    assert first.output_rpm >= 48
    c.accept_output_limit(100., 30.)
    next_sample = observe(c, 100.1, .02, 2.78, 2.78, 46., -.2, .42, .1)
    assert next_sample.brake_settling_limited
    assert next_sample.output_rpm == 30
