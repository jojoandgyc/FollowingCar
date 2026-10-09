"""Same-grant relative braking is not new depth or an absolute stop guarantee."""
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import pytest

from car_control_modular.depth_continuation import (
    ContinuationMotionEvidence, continuation_speed_cap,
    relative_braking_budget, relative_continuation_speed_cap,
)


CIRCUMFERENCE = .816814
STAMP = 100.


def evidence(**changes):
    return replace(ContinuationMotionEvidence(1, STAMP, .7641, .546, .16, 3), **changes)


def inputs(age=.181, *, left=16., right=16., **changes):
    args = dict(distance=2.5648, stop_distance=1.2,
                original_speed_bound=116.*CIRCUMFERENCE/60., sample_age=age,
                feedback=SimpleNamespace(timestamp=STAMP+age-.01,
                    left_forward_rpm=left, right_forward_rpm=right, trustworthy=True),
                now=STAMP+age, circumference=CIRCUMFERENCE, max_rpm=200.,
                deceleration=.4, response_delay=.2, motion=evidence(), target_id=1,
                sample_timestamp=STAMP)
    args.update(changes)
    return args


def cap(**kwargs):
    return relative_continuation_speed_cap(**inputs(**kwargs))


def budget_from(args):
    return relative_braking_budget(
        distance_m=args['distance'], stop_distance_m=args['stop_distance'],
        speed_bound_m_s=args['original_speed_bound'], age_sec=args['sample_age'],
        outer_speed_m_s=max(abs(args['feedback'].left_forward_rpm),
                            abs(args['feedback'].right_forward_rpm))*CIRCUMFERENCE/60.,
        target_speed_m_s=args['motion'].target_speed_bound_m_s,
        deceleration_m_s2=args['deceleration'],
        response_delay_sec=args['response_delay'],
        wheel_circumference_m=args['circumference'], max_rpm=args['max_rpm'])


def test_cap165_measured_momentum_requires_immediate_braking():
    # Physical raw Depth and encoder values from the 2026-10-04 run. The
    # modeled stop margin is already smaller than the measured stopping need;
    # an upstream PI request of 84 RPM cannot make it safe to keep driving.
    args = inputs(age=.1675, distance=2.35472, left=85., right=84.,
                  original_speed_bound=1.225221, deceleration=.7,
                  motion=evidence(target_speed_bound_m_s=.3781920536,
                                  sample_timestamp=STAMP))
    args['feedback'].timestamp = args['now']-.0372
    budget = budget_from(args)
    assert budget.margin_m < budget.required_stop_m
    assert 0 < budget.margin_m < 1.
    assert budget.max_allowed_rpm == 0.
    assert relative_continuation_speed_cap(**args) == (0., 'braking_margin')


def test_same_stopping_budget_provides_nonzero_early_cap_before_emergency():
    # The same physical speed has time to decelerate while farther away;
    # callers can use this cap to start reducing speed *before* a zero veto.
    args = inputs(age=.08, distance=2.4, left=85., right=84.,
                  original_speed_bound=1.225221, deceleration=.7,
                  motion=evidence(target_speed_bound_m_s=.3781920536))
    budget = budget_from(args)
    assert budget.margin_m > budget.required_stop_m
    assert 0 < budget.max_allowed_rpm < 90.
    assert relative_continuation_speed_cap(**args)[0] == pytest.approx(
        budget.max_allowed_rpm)


def test_stationary_target_and_fast_approach_tighten_budget():
    moving = budget_from(inputs(age=.12, distance=3., left=85., right=84.,
                                original_speed_bound=1.225221, deceleration=.7,
                                motion=evidence(target_speed_bound_m_s=.6)))
    stationary = budget_from(inputs(age=.12, distance=3., left=85., right=84.,
                                    original_speed_bound=1.225221, deceleration=.7,
                                    motion=evidence(target_speed_bound_m_s=0.)))
    approaching = budget_from(inputs(age=.12, distance=3., left=85., right=84.,
                                     original_speed_bound=1.225221, deceleration=.7,
                                     motion=evidence(target_speed_bound_m_s=-.3)))
    assert moving.target_credit_m_s > stationary.target_credit_m_s > approaching.target_credit_m_s
    assert moving.required_stop_m < stationary.required_stop_m < approaching.required_stop_m
    assert moving.max_allowed_rpm >= stationary.max_allowed_rpm >= approaching.max_allowed_rpm


def test_expired_feedback_cannot_reuse_a_positive_budget():
    args = inputs(age=.12, distance=3., left=30., right=30.,
                  deceleration=.7, motion=evidence(target_speed_bound_m_s=.3))
    assert budget_from(args).max_allowed_rpm > 0.
    args['feedback'].timestamp = args['now']-.151
    assert relative_continuation_speed_cap(**args) == (
        0., 'continuation_feedback_invalid')


def test_cap238_relative_model_reduces_the_stationary_model_cliff():
    args = inputs()
    rpm, reason = relative_continuation_speed_cap(**args)
    stationary, _ = continuation_speed_cap(**{
        k: v for k, v in args.items() if k not in ('motion', 'target_id', 'sample_timestamp')})
    assert reason == 'same_grant_relative_braking_cap'
    assert stationary+20. < rpm <= 116.


def test_no_discrete_switch_between_179_and_181_ms():
    earlier, _ = cap(age=.179)
    later, _ = cap(age=.181)
    assert 0 <= earlier-later < 1.


@pytest.mark.parametrize('target', [-1., -.2, 0., .2, .7641, 2., 6.])
def test_fixed_grant_cap_is_nonincreasing_for_every_age(target):
    outputs = [cap(age=i/1000., motion=evidence(target_speed_bound_m_s=target))[0]
               for i in range(251)]
    assert all(right <= left+1e-9 for left, right in zip(outputs, outputs[1:]))
    assert all(0 <= rpm <= 116.+1e-9 for rpm in outputs)


def test_deadline_does_not_extend_by_replaying_original_motion():
    assert cap(age=.25)[0] > 0
    assert cap(age=.300001) == (0., 'invalid_braking_model')
    assert cap(age=.251, now=STAMP+.25) == (0., 'invalid_braking_model')


@pytest.mark.parametrize('left,right', [(0., 0.), (16., 16.), (30., 40.), (50., 50.)])
def test_faster_or_newer_usable_encoder_does_not_change_cap(left, right):
    baseline = cap()[0]
    args = inputs(left=left, right=right)
    args['feedback'].timestamp = args['now']
    assert relative_continuation_speed_cap(**args)[0] == pytest.approx(baseline)


def test_original_travel_bound_cannot_be_replaced_by_current_slow_wheels():
    baseline = cap(left=1., right=1.)[0]
    higher_original = cap(left=1., right=1.,
                          original_speed_bound=150.*CIRCUMFERENCE/60.)[0]
    assert higher_original < baseline


def test_outer_wheel_momentum_cannot_be_hidden_by_body_average():
    assert cap(left=40., right=40.)[0] > 0
    # Mean 100 RPM still fits the original 116 RPM bound. Outer-wheel
    # momentum nevertheless exceeds the relative stopping budget.
    assert cap(left=10., right=190.) == (0., 'braking_margin')


def test_body_acceleration_beyond_original_bound_still_vetoes():
    assert cap(left=117., right=117.) == (0., 'continuation_speed_exceeds_bound')


def test_negative_target_velocity_is_never_upgraded_to_stationary():
    static = cap(motion=evidence(target_speed_bound_m_s=0.))[0]
    toward = cap(motion=evidence(target_speed_bound_m_s=-.2))[0]
    assert toward < static
    assert cap(motion=evidence(target_speed_bound_m_s=-2.))[0] == 0.


def test_positive_target_credit_decays_to_stationary_not_infinite_retreat():
    positive = cap(age=.2, motion=evidence(target_speed_bound_m_s=.3))[0]
    stationary = cap(age=.2, motion=evidence(target_speed_bound_m_s=0.))[0]
    assert positive == stationary


@pytest.mark.parametrize('change', [
    {'uid': 2}, {'uid': True}, {'uid': 1.}, {'sample_timestamp': STAMP-.01},
    {'sample_timestamp': True}, {'target_speed_bound_m_s': float('nan')},
    {'target_speed_bound_m_s': float('inf')}, {'target_speed_bound_m_s': True},
    {'target_speed_bound_m_s': 6.1}, {'range_rate_m_s': 3.1},
    {'range_rate_m_s': True}, {'span_sec': .301}, {'span_sec': .024},
    {'span_sec': True}, {'sample_count': 1}, {'sample_count': 18},
    {'sample_count': True}, {'sample_count': 3.},
])
def test_invalid_motion_does_not_create_permission(change):
    assert cap(motion=evidence(**change)) == (0., 'invalid_motion_evidence')


@pytest.mark.parametrize('target_id', [True, 1., 0, -1, None, '1'])
def test_uid_contract_is_strict(target_id):
    assert cap(target_id=target_id) == (0., 'invalid_motion_evidence')


@pytest.mark.parametrize('motion', [None, {}, object()])
def test_missing_motion_is_not_synthesized(motion):
    assert cap(motion=motion) == (0., 'invalid_motion_evidence')


@pytest.mark.parametrize('name', [
    'distance', 'stop_distance', 'original_speed_bound', 'sample_age', 'now',
    'circumference', 'max_rpm', 'deceleration', 'response_delay', 'sample_timestamp',
])
@pytest.mark.parametrize('value', [True, float('nan'), float('inf'), '1'])
def test_all_numeric_inputs_reject_bool_nonfinite_and_strings(name, value):
    args = inputs()
    args[name] = value
    assert relative_continuation_speed_cap(**args) == (0., 'invalid_braking_model')


@pytest.mark.parametrize('change', [
    {'trustworthy': False}, {'left_forward_rpm': -1.}, {'right_forward_rpm': 201.},
    {'left_forward_rpm': True}, {'timestamp': STAMP+.181-.151},
    {'timestamp': STAMP+.182}, {'right_forward_rpm': float('nan')},
])
def test_original_feedback_safety_checks_remain_required(change):
    args = inputs()
    for key, value in change.items():
        setattr(args['feedback'], key, value)
    rpm, reason = relative_continuation_speed_cap(**args)
    assert rpm == 0.
    assert reason in {'continuation_feedback_missing', 'continuation_feedback_invalid'}


def test_feedback_missing_and_no_distance_margin_are_not_relative_exceptions():
    assert cap(feedback=None) == (0., 'continuation_feedback_missing')
    assert cap(distance=1.2) == (0., 'braking_margin')


def test_motion_evidence_is_immutable_and_never_refreshes_itself():
    motion = evidence()
    for age in (.05, .18, .24):
        cap(age=age, motion=motion)
        assert motion.sample_timestamp == STAMP
    with pytest.raises(FrozenInstanceError):
        motion.sample_timestamp = STAMP+.1


def test_negative_result_is_clamped_and_absolute_rpm_limit_is_kept():
    assert cap(motion=evidence(target_speed_bound_m_s=-1.))[0] == 0.
    rpm, _ = cap(distance=100., max_rpm=60.)
    assert rpm == 60.


def test_extreme_finite_values_cannot_escape_as_nonfinite_cap():
    assert cap(response_delay=1e308) == (0., 'invalid_braking_model')
