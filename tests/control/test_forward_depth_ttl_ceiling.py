"""300 ms support ceiling does not change fresh evidence or configured leases."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from car_control_modular.depth_authority_timing import MAX_FORWARD_DEPTH_TTL_SEC
from car_control_modular.controllers import FollowPolicyConfig, FollowSafetyController
from car_control_modular.depth_continuation import (
    ContinuationMotionEvidence, continuation_speed_cap, relative_continuation_speed_cap,
)
from car_control_modular.distance_pi import DistancePiConfig, DistancePiController
from car_control_modular.longitudinal_execution import ForwardExecutionAnchor, ForwardRecoveryAnchor
from car_control_modular.sample_braking import SampleBrakingAssessment


def _pi_update(controller, stamp=100., now=100.):
    return controller.update(3., 1.4, sample_timestamp=stamp, execution_now=now,
        deadband_m=.03, max_output_rpm=200., rise_rpm_per_sec=240.,
        ego_forward_rpm=30., raw_distance_m=3., range_rate_m_s=0., raw_closure_valid=True)


def _assessment():
    return SampleBrakingAssessment(1, 100., 100.05, 5., 30., 20., 100.04,
                                   0., 1.1, .816814, 1., .15, 200.)


def _cap_inputs(age):
    return dict(distance=5., stop_distance=1.1, original_speed_bound=30.*.816814/60.,
        sample_age=age, feedback=SimpleNamespace(timestamp=100.+age-.01, trustworthy=True,
            left_forward_rpm=20., right_forward_rpm=20.), now=100.+age,
        circumference=.816814, max_rpm=200., deceleration=1., response_delay=.15)


def _relative_inputs(age):
    return dict(_cap_inputs(age), motion=ContinuationMotionEvidence(1, 100., 0., -.2, .1, 3),
                target_id=1, sample_timestamp=100.)


@pytest.mark.parametrize("ttl", [.18, .25, .30])
def test_pi_config_support_and_own_expiry_remain_distinct(ttl):
    controller = DistancePiController(DistancePiConfig(physical_ttl_sec=ttl))
    _pi_update(controller)
    result = _pi_update(controller, now=100.+ttl+.001)
    assert result.status == "stale_sample" and result.output_rpm == 0
    assert controller._execution_suspended
    assert DistancePiConfig().physical_ttl_sec == .18
    assert MAX_FORWARD_DEPTH_TTL_SEC == .30


@pytest.mark.parametrize("ttl", [.300001, .5, float("inf"), float("nan"), True, 0.])
def test_pi_rejects_unsupported_or_invalid_ttl(ttl):
    with pytest.raises(ValueError, match="physical_ttl_sec"):
        DistancePiConfig(physical_ttl_sec=ttl)


@pytest.mark.parametrize("ttl,age,expired", [
    (.18, .181, True), (.25, .28, True), (.30, .28, False), (.30, .301, True),
])
def test_controller_uses_configured_lease_for_withdrawal_classification(ttl, age, expired):
    controller = FollowSafetyController(FollowPolicyConfig(
        distance_pid_enable=True, distance_control_mode="distance_pi",
        depth_longitudinal_sample_max_age_sec=ttl,
    ))
    controller.active_target_id = 1
    controller._distance_pid_last_sample_timestamp = 100.
    controller._distance_pi_admitted_grant = (1, 100.)
    reason = "lateral_zero_no_qualified_depth:revoke:expired"
    controller.suspend_longitudinal_authority(100.+age, reason)
    assert controller._distance_pi_grant_withdrawal == (
        1, 100., "physical_depth_expired" if expired else reason)
    assert controller._distance_pid_last_sample_timestamp == 100.


@pytest.mark.parametrize("age", [.181, .25, .28, .30])
def test_300ms_ceiling_never_makes_old_depth_a_new_pi_update(age):
    controller = DistancePiController(DistancePiConfig(physical_ttl_sec=.30))
    result = _pi_update(controller, now=100.+age)
    assert result.status == "continuation_only" and result.output_rpm == 0
    assert controller._last_sample_ts is controller._last_execution_ts is controller.last_result is None
    assert controller.integral_m_s == 0


def test_new_pi_measurement_window_and_feedback_pairing_are_not_extended():
    with pytest.raises(ValueError, match="fresh_update_max_age_sec"):
        DistancePiConfig(physical_ttl_sec=.30, fresh_update_max_age_sec=.181)
    with pytest.raises(ValueError, match="max_integration_gap_sec"):
        DistancePiConfig(physical_ttl_sec=.30, max_integration_gap_sec=.181)
    with pytest.raises(ValueError, match="braking assessment"):
        replace(_assessment(), checked_at=100.181, feedback_timestamp=100.18)
    with pytest.raises(ValueError, match="braking assessment"):
        replace(_assessment(), feedback_timestamp=99.899)


def test_shared_budget_can_tighten_to_300ms_but_cannot_extend_fresh_assessment():
    evidence = _assessment()
    values = [evidence.budget(100.+age, 20., authorized_rpm=20.)
              for age in (.25, .275, .30)]
    assert all(value.cap_rpm == 20. for value in values)
    assert all(a.margin_m > b.margin_m for a, b in zip(values, values[1:]))
    assert evidence.budget(100.300001, 20., authorized_rpm=20.).cap_rpm == 0
    assert evidence.budget(100.28, 20.).reason == "shared_braking_plan_changed"
    assert evidence.sample_timestamp == 100. and evidence.checked_at == 100.05


@pytest.mark.parametrize("cap,inputs", [
    (continuation_speed_cap, _cap_inputs), (relative_continuation_speed_cap, _relative_inputs),
])
def test_continuation_models_support_300ms_only_with_existing_safe_budget(cap, inputs):
    outputs = [cap(**inputs(age))[0] for age in (.25, .275, .30)]
    assert all(0 < value <= 30. for value in outputs)
    assert outputs == sorted(outputs, reverse=True)
    assert cap(**inputs(.300001))[0] == 0
    assert cap(**inputs(.5))[0] == 0


@pytest.mark.parametrize("cap,inputs,feedback_age", [
    (continuation_speed_cap, _cap_inputs, .100001),
    (relative_continuation_speed_cap, _relative_inputs, .150001),
])
def test_extended_continuation_still_requires_original_encoder_freshness(cap, inputs, feedback_age):
    args = inputs(.28)
    args["feedback"].timestamp = args["now"]-feedback_age
    assert cap(**args)[0] == 0


@pytest.mark.parametrize("sent_age,valid", [(.25, True), (.28, True), (.30, True), (.300001, False)])
def test_recovery_anchor_accepts_only_receipts_completed_inside_supported_lease(sent_age, valid):
    receipt = object()
    anchor = ForwardExecutionAnchor(1, 100., 20., 100.+sent_age, receipt)
    proof = ForwardRecoveryAnchor(anchor, receipt, 1, 100.32)
    assert proof.valid_for(1, 100., 100.31, 100.32, .35) is valid
    assert not replace(proof, current_receipt=object()).valid_for(1, 100., 100.31, 100.32, .35)


def test_recovery_memory_and_new_sample_age_are_not_extended_by_300ms_support():
    receipt = object()
    anchor = ForwardExecutionAnchor(1, 100., 20., 100.28, receipt)
    proof = ForwardRecoveryAnchor(anchor, receipt, 1, 100.351)
    assert not proof.valid_for(1, 100., 100.34, 100.351, .35)
    proof = replace(proof, checked_at=100.32)
    assert not proof.valid_for(1, 100., 100.10, 100.32, .35)
