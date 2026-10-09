from dataclasses import replace
from types import SimpleNamespace
import pytest

import request_0513_modular as runtime
from car_control_modular.control_types import SteeringFeedback, ControlDecision, ControlAction
from car_control_modular.depth_continuation import continuation_feedback_speeds, continuation_speed_cap
from test_distance_pi_runtime import pi_owner, extended_pi_owner
from test_lateral_zero_runtime import owner, NOW
from test_longitudinal_authority_runtime import _frame


def cap(**changes):
    args = dict(distance=2.662, stop_distance=1.43, original_speed_bound=100*.816814/60,
                sample_age=.201, feedback=SteeringFeedback(timestamp=10., trustworthy=True,
                  left_forward_rpm=34., right_forward_rpm=32.), now=10., circumference=.816814,
                max_rpm=200, deceleration=.4, response_delay=.2)
    args.update(changes)
    return continuation_speed_cap(**args)


def test_cap329_can_reduce_instead_of_revoke_when_measured_momentum_fits():
    rpm, reason = cap()
    assert 45 < rpm < 65
    assert reason == "same_grant_reduced_braking_cap"


def test_rpm_cap_only_decreases_with_age_even_if_feedback_newer():
    values = [cap(sample_age=t, now=10+t-.201,
                  feedback=SteeringFeedback(timestamp=10+t-.201, trustworthy=True,
                    left_forward_rpm=34, right_forward_rpm=32))[0] for t in (.181, .201, .225, .25)]
    assert values == sorted(values, reverse=True)
    assert cap(sample_age=.251)[0] == 0


@pytest.mark.parametrize("kwargs", [dict(distance=1.5), dict(deceleration=0), dict(circumference=0),
    dict(original_speed_bound=float("nan")), dict(feedback=None),
    dict(feedback=SteeringFeedback(timestamp=10, trustworthy=True, left_forward_rpm=100, right_forward_rpm=100)),
    dict(feedback=SteeringFeedback(timestamp=9.8, trustworthy=True)),
    dict(feedback=SteeringFeedback(timestamp=10.1, trustworthy=True)),
    dict(feedback=SteeringFeedback(timestamp=10, trustworthy=True, left_forward_rpm=-2)),
    dict(feedback=SteeringFeedback(timestamp=10, trustworthy=False))])
def test_insufficient_margin_or_invalid_feedback_still_zero(kwargs):
    assert cap(**kwargs)[0] == 0


def test_real_reader_reduces_without_renewing_or_updating_pid(extended_pi_owner, monkeypatch):
    o = extended_pi_owner
    monkeypatch.setattr(runtime, "ASTRA_DEPTH_CONTINUATION_SPEED_CAP_ENABLE", True)
    monkeypatch.setattr(runtime, "TARGET_DISTANCE", 1.4)
    monkeypatch.setattr(runtime, "DISTANCE_APPROACH_DECELERATION_M_S2", .4)
    monkeypatch.setattr(runtime, "DISTANCE_APPROACH_RESPONSE_DELAY_SEC", .2)
    linear = ("forward", 50, 1, NOW-.02)
    o._depth30_linear_snapshot = linear
    o._depth30_linear_timing = replace(o._depth30_linear_timing, snapshot=linear,
        continuation_distance_m=2.662, continuation_speed_bound_m_s=100*.816814/60)
    o._action_runtime = SimpleNamespace(get_steering_feedback=lambda: SteeringFeedback(
        timestamp=NOW+.19, trustworthy=True, left_forward_rpm=34, right_forward_rpm=32))
    timing, approvals = o._depth30_linear_timing, list(o.approvals)
    value = o._fresh_depth_linear_snapshot(1, now=NOW+.19)
    assert value and 0 < value[1] < 50 and value[3] == linear[3]
    assert o._depth30_linear_snapshot == linear and o._depth30_linear_timing is timing
    assert o.approvals == approvals
    # Later evidence must use the SAME reduced-cap policy, not the old bool-only
    # veto immediately undoing what the motor reader just accepted.
    actions, accepted = o._commit_depth_linear_decision(
        ControlDecision(actions=[ControlAction.forward(50, "late")]),
        _frame(NOW-.005, distance=2.65), 1, is_fresh_depth=True)
    assert accepted and 0 < actions[0].speed_percent <= value[1]
    assert o._depth30_linear_timing.depth_expires_at == timing.depth_expires_at
    assert o.approvals == approvals
    # Original physical deadline and one-way veto remain authoritative.
    assert o._fresh_depth_linear_snapshot(1, now=NOW+.231) is None


def test_danger_veto_not_revivable_by_new_feedback(extended_pi_owner, monkeypatch):
    o = extended_pi_owner
    monkeypatch.setattr(runtime, "ASTRA_DEPTH_CONTINUATION_SPEED_CAP_ENABLE", True)
    o._explicit_stop_requested = True
    assert o._fresh_depth_linear_snapshot(1) is None
    o._explicit_stop_requested = False
    assert o._fresh_depth_linear_snapshot(1) is None


def test_turning_body_budget_does_not_treat_outer_wheel_as_acceleration():
    feedback = SteeringFeedback(timestamp=10, trustworthy=True,
                                left_forward_rpm=50, right_forward_rpm=70)
    rpm, reason = cap(distance=3.2, original_speed_bound=60*.816814/60,
                      sample_age=.2, feedback=feedback)
    assert rpm == pytest.approx(60)
    assert reason == "same_grant_reduced_braking_cap"


def test_turning_body_average_cannot_hide_outer_wheel_braking_need():
    # At 2.78m the body-only stopping calculation fits. The unchanged,
    # conservative outer-wheel budget does NOT fit and must still stop.
    feedback = SteeringFeedback(timestamp=10, trustworthy=True,
                                left_forward_rpm=50, right_forward_rpm=70)
    assert cap(distance=2.78, original_speed_bound=60*.816814/60,
               sample_age=.2, feedback=feedback) == (0., "braking_margin")


def test_cap1628_actual_feedback_still_exceeds_body_bound_and_braking_budget():
    now = 19781.653485323 + .1818
    feedback = SteeringFeedback(timestamp=19781.802578045, trustworthy=True,
                                left_forward_rpm=52, right_forward_rpm=65)
    kwargs = dict(distance=2.2572448824867326, original_speed_bound=58*.816814/60,
                  sample_age=.1818, feedback=feedback, now=now)
    assert cap(**kwargs) == (0., "continuation_speed_exceeds_bound")
    # Even hypothetically raising B to the measured body speed cannot make
    # this log segment pass the independent (outer-wheel) braking envelope.
    kwargs["original_speed_bound"] = 58.5*.816814/60
    assert cap(**kwargs) == (0., "braking_margin")


@pytest.mark.parametrize("left,right", [(-1, 1), (0, 201), (float("nan"), 20),
                                       (float("inf"), 20), (61, 61)])
def test_body_average_never_hides_bad_individual_wheels(left, right):
    feedback = SteeringFeedback(timestamp=10, trustworthy=True,
                                left_forward_rpm=left, right_forward_rpm=right)
    body, outer, _reason = continuation_feedback_speeds(
        feedback=feedback, now=10., circumference=.816814, max_rpm=200,
        original_speed_bound=60*.816814/60)
    assert body is None and outer is None


def prepare_continuation(o, monkeypatch, *, distance=3.2, bound_rpm=60, percent=30,
                         feedback=None, enabled=True):
    monkeypatch.setattr(runtime, "ASTRA_DEPTH_CONTINUATION_SPEED_CAP_ENABLE", enabled)
    monkeypatch.setattr(runtime, "TARGET_DISTANCE", 1.4)
    monkeypatch.setattr(runtime, "DISTANCE_APPROACH_DECELERATION_M_S2", .4)
    monkeypatch.setattr(runtime, "DISTANCE_APPROACH_RESPONSE_DELAY_SEC", .2)
    linear = ("forward", percent, 1, NOW-.02)
    o._depth30_linear_snapshot = linear
    o._depth30_linear_timing = replace(o._depth30_linear_timing, snapshot=linear,
        continuation_distance_m=distance, continuation_speed_bound_m_s=bound_rpm*.816814/60)
    o._action_runtime = SimpleNamespace(get_steering_feedback=lambda: feedback)
    return linear


@pytest.mark.parametrize("enabled", [False, True])
def test_existing_full_speed_path_keeps_sufficient_turning_room(extended_pi_owner, monkeypatch, enabled):
    o = extended_pi_owner
    feedback = SteeringFeedback(timestamp=NOW+.19, trustworthy=True,
                                left_forward_rpm=50, right_forward_rpm=70)
    linear = prepare_continuation(o, monkeypatch, feedback=feedback, enabled=enabled)
    timing, approvals = o._depth30_linear_timing, list(o.approvals)
    assert o._fresh_depth_linear_snapshot(1) == linear
    assert o._depth30_linear_timing is timing and o.approvals == approvals


@pytest.mark.parametrize("enabled", [False, True])
def test_full_speed_path_checks_outer_braking_not_only_body(extended_pi_owner, monkeypatch, enabled):
    o = extended_pi_owner
    feedback = SteeringFeedback(timestamp=NOW+.19, trustworthy=True,
                                left_forward_rpm=50, right_forward_rpm=70)
    linear = prepare_continuation(o, monkeypatch, distance=2.78, feedback=feedback, enabled=enabled)
    assert o._fresh_depth_linear_snapshot(1) is None
    assert o._depth30_continuation_veto == (linear[2], linear[3])


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("feedback", [None,
    SteeringFeedback(timestamp=NOW+.19, trustworthy=False),
    SteeringFeedback(timestamp=NOW, trustworthy=True),
    SteeringFeedback(timestamp=NOW+.20, trustworthy=True),
    SteeringFeedback(timestamp=NOW+.19, trustworthy=True, left_forward_rpm=-1, right_forward_rpm=1),
    SteeringFeedback(timestamp=NOW+.19, trustworthy=True, left_forward_rpm=0, right_forward_rpm=201),
    SteeringFeedback(timestamp=NOW+.19, trustworthy=True, left_forward_rpm=61, right_forward_rpm=61),
])
def test_far_full_speed_path_cannot_skip_current_feedback_checks(extended_pi_owner, monkeypatch, enabled, feedback):
    o = extended_pi_owner
    prepare_continuation(o, monkeypatch, distance=10., feedback=feedback, enabled=enabled)
    assert o._fresh_depth_linear_snapshot(1) is None


def test_reduced_cap_switch_off_does_not_add_permission(extended_pi_owner, monkeypatch):
    o = extended_pi_owner
    feedback = SteeringFeedback(timestamp=NOW+.19, trustworthy=True,
                                left_forward_rpm=34, right_forward_rpm=32)
    linear = prepare_continuation(o, monkeypatch, distance=2.662, bound_rpm=100,
                                 percent=50, feedback=feedback, enabled=False)
    assert o._depth_forward_continuation_limit(linear, o._depth30_linear_timing, NOW+.19) == (0, "braking_margin")
    monkeypatch.setattr(runtime, "ASTRA_DEPTH_CONTINUATION_SPEED_CAP_ENABLE", True)
    percent, _reason = o._depth_forward_continuation_limit(linear, o._depth30_linear_timing, NOW+.19)
    assert 0 < percent < linear[1]


def test_feedback_veto_cannot_be_repaired_without_new_depth(extended_pi_owner, monkeypatch):
    o = extended_pi_owner
    linear = prepare_continuation(o, monkeypatch, feedback=None)
    assert o._fresh_depth_linear_snapshot(1) is None
    o._action_runtime.get_steering_feedback = lambda: SteeringFeedback(
        timestamp=NOW+.19, trustworthy=True, left_forward_rpm=50, right_forward_rpm=70)
    assert o._fresh_depth_linear_snapshot(1) is None
    assert o._depth30_continuation_veto == (linear[2], linear[3])


def test_one_feedback_snapshot_per_limit_evaluation(extended_pi_owner, monkeypatch):
    o = extended_pi_owner
    feedback = SteeringFeedback(timestamp=NOW+.19, trustworthy=True,
                                left_forward_rpm=34, right_forward_rpm=32)
    linear = prepare_continuation(o, monkeypatch, distance=2.662, bound_rpm=100, percent=50)
    samples = iter([feedback])
    o._action_runtime.get_steering_feedback = lambda: next(samples)
    percent, _reason = o._depth_forward_continuation_limit(linear, o._depth30_linear_timing, NOW+.19)
    assert 0 < percent < 50


def test_original_evidence_still_uses_faster_wheel_and_closure_not_mean(pi_owner):
    o = pi_owner
    observed = _frame(NOW-.02, distance=3.)
    observed = replace(observed,
        distance_state=replace(observed.distance_state, raw_distance_m=2.9),
        steering_feedback=SteeringFeedback(timestamp=NOW, trustworthy=True,
            left_forward_rpm=40, right_forward_rpm=80))
    distance, bound = o._depth_continuation_evidence(observed, 1, 30, NOW)
    assert distance == 2.9
    assert bound == pytest.approx(80*runtime.VISION_MMWAVE_FUSION_ENCODER_WHEEL_CIRCUMFERENCE_M/60)
    o._follow_controller.last_distance_pid_result = SimpleNamespace(approach_closing_m_s=2.)
    assert o._depth_continuation_evidence(observed, 1, 30, NOW) == (2.9, 2.)
