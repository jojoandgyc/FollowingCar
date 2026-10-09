"""New-depth recovery uses actual writes, not a fixed low-speed restart."""
from dataclasses import replace

import pytest

from car_control_modular.distance_pi import DistancePiConfig, DistancePiController
from car_control_modular.longitudinal_execution import ForwardExecutionAnchor, ForwardRecoveryAnchor
from car_control_modular.mssd_motor import MotorSpeedReceipt
from car_control_modular.sample_braking import SampleBrakingAssessment


def replay(cap, *, expired=True, fault=None, proof_enabled=True):
    if cap == 208:
        old, stamp = 7960.006343713, 7960.168133041
        actual, raw = 3.1810851302156262, 3.1560916888494526
        travel, outer, ego, late = 119., 33., 32.5, .0013
        rpm, sent = 114., old+.09  # current actual command, not a synthetic demand
    else:
        old, stamp = 7961.383739124, 7961.577621432
        actual, raw = 2.595857142857143, 2.5721217400714202
        travel, outer, ego, late = 92., 0., 0., .0023
        rpm, sent = 84., old+.2523-.062064143
    cfg = DistancePiConfig(
        kp_per_sec=3., physical_ttl_sec=.25, deceleration_m_s2=1.,
        response_delay_sec=.15, launch_request_rpm=180., launch_full_error_m=.5,
        use_target_motion=False, stationary_stop_preview_distance_m=1.1)
    pi = DistancePiController(cfg)
    now = old+.25+late if expired else old+.249
    receipt = MotorSpeedReceipt(1, int(rpm), -int(rpm), sent)
    proof = ForwardRecoveryAnchor(ForwardExecutionAnchor(1, old, rpm, sent, receipt),
                                  receipt, 0, now)

    def step(t, s, d, b, o, e, **kw):
        assessment = (SampleBrakingAssessment(
            1, s, t, d, b, o, t-.03, 0., 1.1, .816814, 1., .15, 200.,
            outer_allowance_rpm=10.) if 0 <= t-s <= .18 else None)
        if fault == "missing_assessment" and s != old:
            assessment = None
        return pi.update(actual, 1.4, sample_timestamp=s, execution_now=t,
            deadband_m=.03, max_output_rpm=200., rise_rpm_per_sec=240.,
            ego_forward_rpm=e, raw_distance_m=d, braking_assessment=assessment,
            preview_outer_forward_rpm=o, preview_feedback_timestamp=t-.03, **kw)

    first = step(old+.06, old, raw, travel, travel, travel)
    if expired:
        pi.suspend(now, "no_live_grant_before_pi", reset_execution=True)
    if fault == "stop": pi.suspend(now, "explicit_stop", reset_execution=True)
    elif fault == "park": pi.set_normal_parking(True)
    elif fault == "rejected": pi.reject_output(old)
    elif fault == "zero": proof = replace(proof, current_receipt=MotorSpeedReceipt(2, 0, 0, now))
    elif fault == "uid": proof = replace(proof, executed=replace(proof.executed, uid=2))
    elif fault == "future_write": proof = replace(proof, executed=replace(proof.executed, sent_at=now+.001))
    elif fault == "unissued": proof = replace(proof, executed=replace(proof.executed, receipt=None))
    elif fault == "replay": stamp = old
    elif fault == "older": stamp = old-.001
    elif fault == "reverse": ego = -1.
    elif fault == "near": actual = raw = 1.4
    elif fault == "long_gap": now = old+.351; stamp = now-.03; proof = replace(proof, checked_at=now)
    result = step(now, stamp, raw, travel, outer, ego,
        depth_expiry_recovery_step_sec=.05 if expired else 0.,
        depth_expiry_execution_anchor=proof.executed if cap == 235 and fault is None else None,
        depth_expiry_expected_uid=1,
        execution_recovery_proof=proof if proof_enabled else None,
        execution_recovery_uid=1)
    return pi, result, first


@pytest.mark.parametrize("cap,old_output", [(208, 44), (235, 24)])
def test_short_expiry_no_longer_restarts_at_fixed_measured_speed_steps(cap, old_output):
    _, before, _ = replay(cap, expired=False)
    _, old, _ = replay(cap, proof_enabled=False)
    _, after, _ = replay(cap)
    assert old.output_rpm == old_output
    assert after.status == "execution_continuity"
    assert after.execution_recovery_anchor_used
    assert abs(after.output_rpm-before.output_rpm) <= 1  # Current cap still tightens with age.
    assert after.output_rpm <= after.cap_rpm
    assert after.sample_dt_sec == 0  # No integration across an expired lease.
    assert after.depth_expiry_recovery_step_sec == 0


@pytest.mark.parametrize("fault", ["stop", "park", "rejected", "zero", "uid", "future_write",
                                  "unissued", "reverse", "near", "long_gap", "missing_assessment"])
def test_completed_command_memory_cannot_override_stop_or_invalid_proof(fault):
    _, result, _ = replay(235, fault=fault)
    assert not result.execution_recovery_anchor_used
    if fault == "near": assert result.output_rpm == 0


@pytest.mark.parametrize("fault", ["replay", "older"])
def test_repeated_or_older_depth_cannot_spend_completed_command_ramp(fault):
    # An expired reused depth is independently rejected before any continuity
    # proof can advance the timeline or create a new command.
    _, result, _ = replay(235, fault=fault)
    assert not result.execution_recovery_anchor_used
    assert result.output_rpm == 0


def test_valid_command_memory_still_obeys_new_tighter_braking_envelope():
    _, result, _ = replay(235)
    assert result.output_rpm == 91
    assert result.cap_rpm < 93
    assert result.final_limit_reason == "braking_envelope"


@pytest.mark.parametrize("bad", [True, float("nan"), -1, None])
def test_untyped_or_nonfinite_recovery_proof_values_cannot_supply_ramp(bad):
    receipt = object()
    original = ForwardRecoveryAnchor(ForwardExecutionAnchor(1, 10., 80., 10.1, receipt),
                                     receipt, 0, 10.26)
    assert not replace(original, checked_at=bad).valid_for(1, 10., 10.2, 10.26, .35)


def test_reusing_a_completed_packet_does_not_stack_recovery_step_credit():
    pi, first, _ = replay(235)
    source, sent = 7961.383739124, 7961.383739124+.2523-.062064143
    receipt = MotorSpeedReceipt(1, 84, -84, sent)
    executed = ForwardExecutionAnchor(1, source, 84., sent, receipt)
    for elapsed in (.27, .29, .32, .34):
        now, stamp = source+elapsed, source+elapsed-.03
        pi.suspend(now, "physical_depth_expired", reset_execution=True)
        proof = ForwardRecoveryAnchor(executed, receipt, 0, now)
        assessment = SampleBrakingAssessment(1, stamp, now, 3., 92., 0., now,
            0., 1.1, .816814, 1., .15, 200., outer_allowance_rpm=10.)
        result = pi.update(3., 1.4, sample_timestamp=stamp, execution_now=now,
            deadband_m=.03, max_output_rpm=200., rise_rpm_per_sec=240.,
            ego_forward_rpm=0., raw_distance_m=3., braking_assessment=assessment,
            preview_outer_forward_rpm=0., preview_feedback_timestamp=now,
            execution_recovery_proof=proof, execution_recovery_uid=1)
        assert result.execution_recovery_anchor_used
        # Bound is measured from the ORIGINAL actual write, not from the last
        # recovered request. More calls cannot manufacture extra acceleration.
        assert result.output_rpm <= min(result.cap_rpm, 84.+240.*(now-sent))
