"""The caller-qualified far recovery step also starts from measured zero.

The runtime must qualify the old UID/grant and NEW distance/motion window;
the pure PI neither renews the old lease nor awards a step on its own.
"""
import pytest

from car_control_modular.distance_pi import DistancePiConfig, DistancePiController
from car_control_modular.longitudinal_approach import RawDepthMotionEvidence


def observe(c, stamp, *, ego=0., step=.05, raw=2.49, target_speed=0., **changes):
    rate = target_speed-(0. if ego is None else ego*.816814/60.)
    args = dict(sample_timestamp=stamp, execution_now=stamp+.01,
                deadband_m=.03, max_output_rpm=200., rise_rpm_per_sec=240.,
                ego_forward_rpm=ego, raw_closure_valid=True,
                raw_distance_m=raw, range_rate_m_s=rate,
                raw_motion_evidence=RawDepthMotionEvidence(stamp, rate, target_speed, .1, 3),
                depth_expiry_recovery_step_sec=step)
    args.update(changes)
    return c.update(raw, 1.4, **args)


def seeded():
    c = DistancePiController(DistancePiConfig(physical_ttl_sec=.25, deceleration_m_s2=.7))
    first = observe(c, 100., ego=26., step=0.)
    assert first.output_rpm == 26
    return c


def test_caller_qualified_new_far_sample_gets_twelve_rpm_from_zero():
    c = seeded()
    result = observe(c, 100.26)
    assert result.status == 'recovering'
    assert result.depth_expiry_recovery_used
    assert result.depth_expiry_recovery_step_sec == .05
    assert result.output_rpm == 12
    assert result.output_rpm <= result.cap_rpm
    assert result.sample_timestamp == c._last_sample_ts == 100.26
    assert result.sample_dt_sec == 0  # no integral across the old lease gap


@pytest.mark.parametrize('case', ['no_step', 'reverse', 'invalid_encoder', 'close',
                                 'approaching', 'identity', 'parked', 'no_previous'])
def test_zero_speed_recovery_does_not_remove_any_other_guard(case):
    c = seeded() if case != 'no_previous' else DistancePiController(
        DistancePiConfig(physical_ttl_sec=.25))
    kwargs = {}
    if case == 'no_step': kwargs['step'] = 0.
    if case == 'reverse': kwargs['ego'] = -1.
    if case == 'invalid_encoder': kwargs['ego'] = None
    if case == 'close': kwargs['raw'] = 1.39
    if case == 'approaching': kwargs['target_speed'] = -2.
    if case == 'identity': c.suspend(100.2, 'identity_lost', reset_execution=True)
    if case == 'parked': c.set_normal_parking(True)
    result = observe(c, 100.26, **kwargs)
    assert result.output_rpm == 0
    assert not result.depth_expiry_recovery_used


def test_live_prior_grant_does_not_receive_an_expiry_step():
    c = seeded()
    result = observe(c, 100.1)
    assert not result.depth_expiry_recovery_used
    assert result.depth_expiry_recovery_step_sec == 0


@pytest.mark.parametrize('case', ['duplicate', 'older', 'stale'])
def test_duplicate_or_stale_sample_cannot_claim_zero_speed_expiry_recovery(case):
    c = seeded()
    stamp = 100.26 if case == 'stale' else 100. if case == 'duplicate' else 99.9
    result = observe(c, stamp, execution_now=stamp+.26 if case == 'stale' else 100.02)
    assert not result.depth_expiry_recovery_used
    assert c._last_sample_ts == 100.
