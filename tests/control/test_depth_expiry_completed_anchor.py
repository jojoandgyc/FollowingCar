"""A completed positive write may soften only a newly measured, short expiry.

This is controller arithmetic, not proof of a live motor grant. The caller
must validate that the write is still the last motor receipt and that the new
depth/encoder evidence is qualified before passing the bounded recovery step.
"""

from dataclasses import replace

import pytest

from car_control_modular.distance_pi import DistancePiConfig, DistancePiController
from car_control_modular.longitudinal_execution import ForwardExecutionAnchor


def recovered(distance, *, measured=17.5, old_measured=34., proof=None,
              expected_uid=1, old_withdrawal=None, rate=0., step=.05):
    pi = DistancePiController(DistancePiConfig(
        physical_ttl_sec=.25, launch_request_rpm=180., launch_full_error_m=.5))
    args = dict(target_distance_m=1.4, deadband_m=.03,
                max_output_rpm=200., rise_rpm_per_sec=240.,
                fall_rpm_per_sec=300., raw_closure_valid=True,
                range_rate_m_s=rate, raw_distance_m=distance)
    pi.update(distance, sample_timestamp=100., execution_now=100.,
              ego_forward_rpm=old_measured, **args)
    if old_withdrawal is not None:
        pi.suspend(100.2, old_withdrawal, reset_execution=True)
    result = pi.update(
        distance, sample_timestamp=100.23, execution_now=100.264,
        ego_forward_rpm=measured, depth_expiry_recovery_step_sec=step,
        depth_expiry_execution_anchor=proof,
        depth_expiry_expected_uid=expected_uid, **args)
    return pi, result


GOOD_WRITE = ForwardExecutionAnchor(1, 100., 34., 100.239, object())


def test_cap675_new_depth_uses_recent_completed_target_without_reusing_old_lease():
    _, wheel_only = recovered(1.813)
    pi, with_write = recovered(1.813, proof=GOOD_WRITE)
    assert wheel_only.status == with_write.status == 'recovering'
    assert wheel_only.output_rpm == 29
    assert with_write.output_rpm == 40
    assert with_write.depth_expiry_completed_anchor_used
    assert with_write.depth_expiry_completed_anchor_rpm == pytest.approx(34.)
    assert with_write.output_rpm < with_write.cap_rpm
    assert pi._last_sample_ts == pytest.approx(100.23)
    assert with_write.sample_timestamp == pytest.approx(100.23)


@pytest.mark.parametrize(('sent_at', 'expected_rpm'), [
    (100.250, 37),
    (100.239, 40),
    (100.220, 41),
])
def test_completed_write_rise_budget_tracks_actual_time_since_write(sent_at, expected_rpm):
    _, result = recovered(1.813, proof=replace(GOOD_WRITE, sent_at=sent_at))
    assert result.output_rpm == expected_rpm
    assert result.depth_expiry_completed_anchor_used


def test_cap734_after_zero_receipt_uses_only_fresh_wheel_feedback_step():
    # A zero/STOP after the earlier command invalidates its receipt upstream;
    # there is deliberately no completed-write proof to pass here.
    _, result = recovered(2.228, measured=30.5, old_measured=50.)
    assert result.output_rpm == 42
    assert result.depth_expiry_recovery_used
    assert not result.depth_expiry_completed_anchor_used
    assert result.output_rpm <= min(result.cap_rpm, 30.5+240*.05)


@pytest.mark.parametrize('change', [
    'wrong_uid', 'bool_uid', 'missing_uid', 'bool_expected_uid', 'wrong_sample',
    'stale_write', 'after_old_deadline', 'future_write', 'zero_write',
    'missing_receipt', 'nonfinite_rpm', 'invalid_rpm', 'invalid_write_time',
])
def test_unverified_or_late_completed_write_never_adds_recovery_credit(change):
    anchor = GOOD_WRITE
    expected_uid = 1
    changes = {
        'wrong_uid': dict(uid=2),
        'bool_uid': dict(uid=True),
        'wrong_sample': dict(sample_timestamp=99.99),
        'stale_write': dict(sent_at=100.14),
        'after_old_deadline': dict(sent_at=100.251),
        'future_write': dict(sent_at=100.265),
        'zero_write': dict(rpm=0.),
        'missing_receipt': dict(receipt=None),
        'nonfinite_rpm': dict(rpm=float('nan')),
        'invalid_rpm': dict(rpm=None),
        'invalid_write_time': dict(sent_at='recent'),
    }
    if change == 'missing_uid':
        expected_uid = None
    elif change == 'bool_expected_uid':
        expected_uid = True
    else:
        anchor = replace(anchor, **changes[change])
    _, result = recovered(1.813, proof=anchor, expected_uid=expected_uid)
    assert result.output_rpm == 29
    assert not result.depth_expiry_completed_anchor_used


@pytest.mark.parametrize('distance', [1.48, 1.74])
def test_near_target_cannot_borrow_old_high_request(distance):
    _, wheel_only = recovered(distance)
    _, with_write = recovered(distance, proof=GOOD_WRITE)
    assert with_write.output_rpm == wheel_only.output_rpm
    assert not with_write.depth_expiry_completed_anchor_used


@pytest.mark.parametrize('reason', ['identity_lost', 'emergency_stop',
                                    'lateral_depth_ineligible'])
def test_explicit_non_expiry_withdrawal_forbids_completed_write_recovery(reason):
    _, result = recovered(1.813, proof=GOOD_WRITE, old_withdrawal=reason)
    assert not result.depth_expiry_recovery_used
    assert not result.depth_expiry_completed_anchor_used
    assert result.output_rpm <= 17.5


def test_new_depth_braking_cap_wins_even_with_recent_positive_packet():
    _, result = recovered(1.813, proof=GOOD_WRITE, rate=-1.5)
    assert result.cap_rpm == result.output_rpm == 0
    assert not result.depth_expiry_completed_anchor_used


def test_stronger_fresh_depth_step_never_multiplies_old_packet_credit():
    old_fast = replace(GOOD_WRITE, rpm=120.)
    _, measured_only = recovered(1.813, measured=0., proof=None, step=.15)
    _, with_write = recovered(1.813, measured=0., proof=old_fast, step=.15)
    assert with_write.output_rpm == measured_only.output_rpm
    assert with_write.output_rpm <= min(with_write.cap_rpm, 240.*.15)
    assert not with_write.depth_expiry_completed_anchor_used


def test_stronger_fresh_depth_step_still_obeys_immediate_braking_zero():
    _, result = recovered(1.813, measured=0., proof=GOOD_WRITE,
                          rate=-1.5, step=.15)
    assert result.cap_rpm == result.output_rpm == 0


def test_completed_old_target_cannot_raise_new_small_distance_request():
    def run(proof):
        pi = DistancePiController(DistancePiConfig(
            kp_per_sec=.8, physical_ttl_sec=.25))
        args = dict(target_distance_m=1.4, deadband_m=.03,
                    max_output_rpm=200., rise_rpm_per_sec=240.,
                    fall_rpm_per_sec=300., raw_closure_valid=True,
                    range_rate_m_s=0., raw_distance_m=1.813)
        pi.update(1.813, sample_timestamp=100., execution_now=100.,
                  ego_forward_rpm=34., **args)
        return pi.update(
            1.813, sample_timestamp=100.23, execution_now=100.264,
            ego_forward_rpm=17.5, depth_expiry_recovery_step_sec=.05,
            depth_expiry_execution_anchor=proof,
            depth_expiry_expected_uid=1, **args)

    measured_only = run(None)
    with_write = run(GOOD_WRITE)
    assert 17.5 < with_write.output_rpm < 29.5
    assert with_write.output_rpm == measured_only.output_rpm
    assert not with_write.depth_expiry_completed_anchor_used
