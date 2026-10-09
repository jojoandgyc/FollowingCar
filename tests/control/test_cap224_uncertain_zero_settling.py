"""CAP222/224: a missing-motion zero must not masquerade as measured braking.

Pure controller only: no sensor, serial, runtime thread or motor access.
"""
from dataclasses import replace

import pytest

from car_control_modular.distance_pi import DistancePiConfig, DistancePiController
from car_control_modular.longitudinal_approach import RawDepthMotionEvidence


ORIGIN = 40124.792545741
ZERO = 40125.094278359
RECOVER = 40125.119485572


def controller():
    return DistancePiController(DistancePiConfig(
        kp_per_sec=3., physical_ttl_sec=.25, deceleration_m_s2=.7,
        motion_memory_sec=.35, launch_request_rpm=180., launch_full_error_m=.5))


def observe(c, stamp=RECOVER, *, age=.0723, distance=1.768, raw=1.768,
            ego=8.5, rate=-.07553151429392846,
            target=.040183802372734605, valid=True, evidence=True, **changes):
    args = dict(sample_timestamp=stamp, execution_now=stamp+age,
                deadband_m=.03, max_output_rpm=200., rise_rpm_per_sec=240.,
                ego_forward_rpm=ego, range_rate_m_s=rate,
                raw_closure_valid=valid, allow_motion_memory=True,
                raw_distance_m=raw)
    if evidence:
        args['raw_motion_evidence'] = RawDepthMotionEvidence(
            stamp, rate, target, .025207213, 2)
    args.update(changes)
    return c.update(distance, 1.4, **args)


def uncertainty_zero(c):
    first = observe(c, ORIGIN, age=.177, distance=1.7552, raw=1.759,
                    ego=0., rate=.11230466670656515,
                    target=.11230466670656515, rise_rpm_per_sec=0.)
    assert first.output_rpm == 42
    c.accept_output_limit(ORIGIN, 36.)
    c.suspend(40125.029, 'lateral_zero_no_qualified_depth:visual_pid_center_hold',
              reset_execution=True)
    zero = observe(c, ZERO, age=.0376, distance=1.759, raw=1.7698,
                   ego=8.5, rate=None, valid=False, evidence=False)
    assert zero.output_rpm == 0
    assert zero.brake_source == 'relative_motion_memory'
    assert zero.memory_time_penalty_m_s == pytest.approx(.678665236, abs=1e-8)
    assert c._uncertain_zero_brake_anchor
    return zero


def test_cap224_reliable_window_releases_uncertainty_zero_not_current_envelope():
    c = controller()
    uncertainty_zero(c)
    recovered = observe(c)
    assert recovered.brake_settling_uncertainty_released
    assert not recovered.brake_settling_limited
    assert recovered.pre_settling_cap_rpm == pytest.approx(41.38438, abs=1e-4)
    assert recovered.output_rpm == 14
    assert recovered.output_rpm <= 240.*(RECOVER+.0723-ZERO-.0376)
    assert recovered.output_rpm <= recovered.cap_rpm
    second = observe(c, 40125.222647697, age=.038,
                     raw=1.7552, ego=7., rate=-.12032371479001751,
                     target=-.013394546866453121)
    assert not second.brake_settling_limited
    assert 14 < second.output_rpm <= second.cap_rpm
    assert second.output_rpm <= recovered.output_rpm+240.*second.execution_ramp_dt_sec


def test_quantized_zero_during_tiny_execution_step_does_not_lose_recovery_provenance():
    c = controller()
    uncertainty_zero(c)
    stamp = ZERO+.025
    tiny = observe(c, stamp, age=.0127)
    assert tiny.output_rpm == 0  # only .1ms of new execution time
    assert tiny.brake_settling_uncertainty_released
    assert c._uncertain_zero_brake_anchor
    recovered = observe(c, stamp+.03, age=.03)
    assert recovered.output_rpm > 0
    # The prior zero is now only a quantized recovery ramp, not a new brake
    # latch. It no longer needs to be released again on every fresh sample.
    assert not recovered.brake_settling_limited
    assert not recovered.brake_settling_uncertainty_released
    assert recovered.output_rpm <= recovered.execution_anchor_rpm+240*recovered.execution_ramp_dt_sec


@pytest.mark.parametrize('case', ['no_window', 'bad_stamp', 'bad_rate', 'too_short',
                                 'invalid_encoder', 'invalid_closure'])
def test_missing_or_unpaired_window_cannot_release_uncertain_zero(case):
    c = controller()
    uncertainty_zero(c)
    unchanged = controller()
    uncertainty_zero(unchanged)
    unchanged._uncertain_zero_brake_anchor = False
    evidence = RawDepthMotionEvidence(RECOVER, -.07553151429392846,
                                     .040183802372734605, .025207213, 2)
    changes = {}
    if case == 'no_window': changes['evidence'] = False
    if case == 'bad_stamp': changes['raw_motion_evidence'] = replace(evidence, sample_timestamp=ZERO)
    if case == 'bad_rate': changes['raw_motion_evidence'] = replace(evidence, range_rate_m_s=-1.)
    if case == 'too_short': changes['raw_motion_evidence'] = replace(evidence, span_sec=.01)
    if case == 'invalid_encoder': changes['ego'] = None
    if case == 'invalid_closure': changes['valid'] = False
    r = observe(c, **changes)
    baseline = observe(unchanged, **changes)
    assert not r.brake_settling_uncertainty_released
    # An independent encoder fallback may still request bounded movement;
    # this change cannot add speed through the rejected window.
    assert r.output_rpm == baseline.output_rpm
    assert r.cap_rpm == baseline.cap_rpm


@pytest.mark.parametrize('case', ['close', 'rapid_approach', 'high_wheel_speed'])
def test_new_real_braking_still_wins_over_uncertainty_recovery(case):
    c = controller()
    uncertainty_zero(c)
    changes = {}
    if case == 'close': changes.update(distance=1.42, raw=1.41, target=0.)
    if case == 'rapid_approach': changes.update(rate=-1., target=-.88)
    if case == 'high_wheel_speed': changes.update(ego=100., rate=-1.3, target=.04)
    r = observe(c, **changes)
    assert not r.brake_settling_uncertainty_released
    assert r.output_rpm == 0
    assert not c._uncertain_zero_brake_anchor


@pytest.mark.parametrize('case', ['duplicate', 'older', 'stale', 'late'])
def test_nonfresh_sample_cannot_release_anchor_or_credit_ramp(case):
    c = controller()
    previous = uncertainty_zero(c)
    old_execution = c._last_execution_ts
    changes = {'stamp': ZERO}
    if case == 'older': changes['stamp'] -= .01
    if case == 'stale': changes.update(stamp=RECOVER, age=.26)
    if case == 'late': changes.update(stamp=RECOVER, age=.19)
    r = observe(c, **changes)
    assert r.output_rpm == 0 and not r.brake_settling_uncertainty_released
    assert c._last_execution_ts == old_execution
    assert c._last_sample_ts == previous.sample_timestamp


@pytest.mark.parametrize('case', ['reject', 'geometry_fault', 'identity', 'parking',
                                 'external_zero'])
def test_external_rejection_cannot_inherit_uncertainty_release(case):
    c = controller()
    uncertainty_zero(c)
    if case == 'reject': c.reject_output(ZERO)
    if case == 'geometry_fault': c.invalidate_motion_memory()
    if case == 'identity': c.suspend(ZERO+.04, 'identity_conflict', reset_execution=True)
    if case == 'parking': c.set_normal_parking(True)
    if case == 'external_zero':
        positive = observe(c)
        assert positive.output_rpm > 0
        assert c.accept_output_limit(RECOVER, 0.)
    assert not c._uncertain_zero_brake_anchor
    r = observe(c, RECOVER+.03, age=.03)
    assert not r.brake_settling_uncertainty_released


def test_measured_high_speed_braking_zero_is_never_relabelled_as_uncertainty():
    c = controller()
    first = observe(c, ORIGIN, age=.02, distance=1.7, raw=1.6,
                    ego=100., rate=-1.5, target=-.14)
    assert first.output_rpm == 0
    assert not c._uncertain_zero_brake_anchor
    memory = observe(c, ORIGIN+.30, age=.02, distance=1.7, raw=1.61,
                     ego=8.5, rate=None, valid=False, evidence=False)
    assert memory.output_rpm == 0 and memory.brake_source == 'relative_motion_memory'
    assert not c._uncertain_zero_brake_anchor
    recovered = observe(c, ORIGIN+.33, age=.02, raw=1.6, distance=1.7)
    assert recovered.output_rpm == 0
    assert recovered.brake_settling_limited
    assert not recovered.brake_settling_uncertainty_released


def test_startup_zero_or_memory_no_relaunch_rule_does_not_create_uncertainty_marker():
    c = controller()
    first = observe(c, ORIGIN, age=.02, distance=1.7552, raw=1.759,
                    ego=0., rate=.1123, target=.1123)
    assert first.output_rpm == 0 and first.cap_rpm > 0  # initial rise budget is zero
    memory = observe(c, ORIGIN+.08, age=.02, distance=1.76, raw=1.77,
                     ego=0., rate=None, valid=False, evidence=False)
    assert memory.brake_source == 'relative_motion_memory'
    assert memory.output_rpm == 0
    assert not c._uncertain_zero_brake_anchor


def test_new_close_raw_return_without_window_replaces_uncertainty_stop_provenance():
    c = controller()
    uncertainty_zero(c)
    near = observe(c, raw=1.41, distance=1.42, evidence=False,
                   valid=False, rate=-.3)
    assert near.output_rpm == 0 and near.brake_source != 'relative_motion_memory'
    assert not c._uncertain_zero_brake_anchor
    later = observe(c, RECOVER+.03, age=.03)
    assert not later.brake_settling_uncertainty_released
