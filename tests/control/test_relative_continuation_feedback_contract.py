"""Fresh relative envelopes tolerate bounded sampling tails, not lost authority."""
from types import SimpleNamespace

import pytest

from car_control_modular.depth_continuation import (
    ContinuationMotionEvidence,
    RELATIVE_CONTINUATION_FEEDBACK_MAX_AGE_SEC,
    RELATIVE_CONTINUATION_REVERSE_TAIL_RPM,
    continuation_feedback_speeds,
    continuation_speed_cap,
    relative_continuation_speed_cap,
)


CIRCUMFERENCE = .816814
STAMP = 100.


def inputs(*, age=.084, feedback_age=.102, left=0., right=0., **changes):
    values = dict(distance=2.142, stop_distance=1.2,
        original_speed_bound=68*CIRCUMFERENCE/60., sample_age=age,
        feedback=SimpleNamespace(timestamp=STAMP+age-feedback_age, trustworthy=True,
            left_forward_rpm=left, right_forward_rpm=right), now=STAMP+age,
        circumference=CIRCUMFERENCE, max_rpm=200., deceleration=.4,
        response_delay=.2, motion=ContinuationMotionEvidence(
            1, STAMP, .333475, .1, .16, 3), target_id=1, sample_timestamp=STAMP)
    values.update(changes)
    return values


def feedback_speeds(args, **changes):
    values = {key: args[key] for key in ('feedback', 'now', 'circumference',
                                        'max_rpm', 'original_speed_bound')}
    values.update(feedback_max_age_sec=RELATIVE_CONTINUATION_FEEDBACK_MAX_AGE_SEC,
                  reverse_tail_rpm=RELATIVE_CONTINUATION_REVERSE_TAIL_RPM)
    values.update(changes)
    return continuation_feedback_speeds(**values)


def static_cap(args):
    return continuation_speed_cap(**{key: value for key, value in args.items()
        if key not in ('motion', 'target_id', 'sample_timestamp')})


def test_cap172_102ms_feedback_does_not_withdraw_fresh_relative_depth():
    args = inputs()
    rpm, reason = relative_continuation_speed_cap(**args)
    assert 0 < rpm <= 68
    assert reason == 'same_grant_relative_braking_cap'
    # The static/legacy caller remains exactly on the old100ms contract.
    assert static_cap(args) == (0., 'continuation_feedback_invalid')


@pytest.mark.parametrize('feedback_age', [.151, .20, .251, -.001])
def test_relative_feedback_deadline_and_future_timestamp_still_reject(feedback_age):
    args = inputs(feedback_age=feedback_age)
    assert relative_continuation_speed_cap(**args) == (0., 'continuation_feedback_invalid')


@pytest.mark.parametrize('left,right', [(-2., 0.), (-1., -1.), (0., -3.), (-3., 2.)])
def test_small_signed_pair_tail_keeps_absolute_outer_momentum(left, right):
    args = inputs(feedback_age=.01, left=left, right=right)
    body, outer, reason = feedback_speeds(args)
    assert reason == 'continuation_feedback_valid'
    assert body == max(0., .5*(left+right))*CIRCUMFERENCE/60.
    assert outer == max(abs(left), abs(right))*CIRCUMFERENCE/60.
    assert relative_continuation_speed_cap(**args)[0] > 0
    assert static_cap(args) == (0., 'continuation_feedback_invalid')


@pytest.mark.parametrize('left,right', [(0., -37.), (-1., 20.), (20., -1.),
                                       (-3.01, 0.), (-3.01, -3.01)])
def test_one_small_negative_wheel_cannot_hide_significant_rotation(left, right):
    args = inputs(feedback_age=.01, left=left, right=right)
    assert feedback_speeds(args) == (None, None, 'continuation_feedback_invalid')
    assert relative_continuation_speed_cap(**args) == (0., 'continuation_feedback_invalid')


def test_near_zero_signed_tail_is_not_zero_braking_momentum():
    # Five millimetres of available margin cannot stop3RPM with this delay.
    # A zero-clamped outer speed would wrongly grant the positive command.
    args = inputs(age=0., feedback_age=0., left=-3., right=0., distance=1.24,
                  original_speed_bound=.1, motion=ContinuationMotionEvidence(
                      1, STAMP, 0., -.01, .16, 3))
    assert feedback_speeds(args)[1] == pytest.approx(3*CIRCUMFERENCE/60.)
    assert relative_continuation_speed_cap(**args) == (0., 'braking_margin')


@pytest.mark.parametrize('feedback_age', [.001, .099, .102, .149])
def test_full_150ms_is_reserved_regardless_of_actual_encoder_age(feedback_age):
    baseline = relative_continuation_speed_cap(**inputs(feedback_age=.001))[0]
    assert relative_continuation_speed_cap(**inputs(feedback_age=feedback_age))[0] == baseline


@pytest.mark.parametrize('target', [-.2, 0., .4, 1.])
def test_age_budget_and_cap_remain_monotone_with_alternating_feedback(target):
    caps = []
    for index in range(251):
        args = inputs(age=index/1000., feedback_age=.149 if index % 2 else .001,
            motion=ContinuationMotionEvidence(1, STAMP, target, .1, .16, 3))
        caps.append(relative_continuation_speed_cap(**args)[0])
    assert all(new <= old+1e-9 for old, new in zip(caps, caps[1:]))


def test_feedback_allowance_does_not_extend_the_physical_depth_deadline():
    assert relative_continuation_speed_cap(**inputs(age=.250))[0] > 0
    assert relative_continuation_speed_cap(**inputs(age=.301)) == (0., 'invalid_braking_model')


@pytest.mark.parametrize('change', [
    {'timestamp': float('nan')}, {'timestamp': True}, {'left_forward_rpm': float('inf')},
    {'left_forward_rpm': True}, {'right_forward_rpm': 201.}, {'trustworthy': False},
])
def test_nonfinite_unsigned_and_untrusted_feedback_are_not_relaxed(change):
    args = inputs(feedback_age=.01)
    for key, value in change.items():
        setattr(args['feedback'], key, value)
    rpm, reason = relative_continuation_speed_cap(**args)
    assert rpm == 0
    assert reason in {'continuation_feedback_missing', 'continuation_feedback_invalid'}


@pytest.mark.parametrize('change', [
    {'feedback_max_age_sec': .151}, {'feedback_max_age_sec': 0.},
    {'feedback_max_age_sec': True}, {'feedback_max_age_sec': float('nan')},
    {'reverse_tail_rpm': 3.01}, {'reverse_tail_rpm': -1.},
    {'reverse_tail_rpm': True}, {'reverse_tail_rpm': float('inf')},
])
def test_optional_feedback_contract_cannot_exceed_bounded_relative_policy(change):
    assert feedback_speeds(inputs(), **change) == (None, None, 'invalid_braking_model')


def test_cap32_prior_written_command_budget_keeps_new_deceleration_request():
    args = inputs(age=.0534, feedback_age=.001, distance=1.912, left=21., right=21.,
        original_speed_bound=20*CIRCUMFERENCE/60., motion=ContinuationMotionEvidence(
            1, STAMP, .22961327576573343, .03221655909906668, .232306142, 2))
    assert relative_continuation_speed_cap(**args) == (0., 'continuation_speed_exceeds_bound')
    # Runtime must prove/capture the previous43/33RPM write; the pure helper
    # cannot invent it from a later feedback or increase the approved18RPM.
    args['original_speed_bound'] = 43*CIRCUMFERENCE/60.
    rpm, reason = relative_continuation_speed_cap(**args)
    assert reason == 'same_grant_relative_braking_cap'
    assert min(18., rpm) == 18.


def test_fresh_relative_feedback_still_checks_real_high_outer_momentum():
    args = inputs(distance=2.56, feedback_age=.149, left=10., right=190.,
                  original_speed_bound=116*CIRCUMFERENCE/60.)
    assert relative_continuation_speed_cap(**args) == (0., 'braking_margin')
