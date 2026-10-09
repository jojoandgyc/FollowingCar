"""New forward admission at obsolete-yaw/zero commits; fake motor I/O only."""
import pytest

from test_unified_forward_snapshot import feedback, inject, writer
from test_zero_publication_lifecycle import churn_at_three_terminal_checks


@pytest.mark.parametrize("old_yaw,new_yaw", [(-4., 0.), (4., 0.), (-4., 4.), (4., -4.)])
@pytest.mark.parametrize("new_base", [50., 70.])
def test_retry_exhaustion_discards_obsolete_yaw_but_rechecks_latest_forward(
        monkeypatch, old_yaw, new_yaw, new_base):
    rt, owner, driver, clock, publish, _ = writer(monkeypatch, base=60., yaw=old_yaw)
    calls = churn_at_three_terminal_checks(monkeypatch, rt, clock, publish,
                                          base=new_base, yaw=new_yaw)
    rt._service_follow_wheels()
    assert len(calls) == 4  # Three fused plans, at most one independent straight plan.
    assert driver.pairs == [(int(60+old_yaw), -int(60-old_yaw)),
                            (int(new_base), -int(new_base))]
    assert not driver.stops
    assert owner._depth30_linear_timing.depth_expires_at == pytest.approx(
        owner._depth30_linear_snapshot[3]+.25)


@pytest.mark.parametrize("yaw", [-4., 0., 4.])
@pytest.mark.parametrize("stage", ["before", "guard", "commit", "terminal"])
def test_ten_rpm_common_increment_is_fused_without_zero(monkeypatch, yaw, stage):
    rt, owner, driver, clock, publish, _ = writer(monkeypatch, base=40., yaw=yaw)
    if stage == "before":
        publish(50., yaw)
    else:
        inject(monkeypatch, rt, stage, lambda: publish(50., yaw))
    rt._service_follow_wheels()
    assert driver.pairs == [(int(40+yaw), -int(40-yaw)),
                            (int(50+yaw), -int(50-yaw))]
    assert not driver.stops


@pytest.mark.parametrize("old_yaw", [-4., 0., 4.])
@pytest.mark.parametrize("loss_mode", ["decelerating", "pivot", "zero"])
def test_new_depth_arriving_at_old_zero_commit_replaces_obsolete_stop(
        monkeypatch, old_yaw, loss_mode):
    rt, owner, driver, clock, publish, _ = writer(monkeypatch, base=60., yaw=old_yaw)
    rt.config.follow_forward_loss_handoff_enable = loss_mode != "zero"
    rt._forward_loss_handoff.note_sent((60+old_yaw, 60-old_yaw), 10.)
    if loss_mode == "pivot":
        clock[0] = 10.2
        publish(60., old_yaw or 4., new_depth=False)
    clock[0] = 10.251
    rt._steering_feedback = feedback(clock[0], 24., 24.)
    inject(monkeypatch, rt, "commit", lambda: publish(54., 0.))
    rt._service_follow_wheels()
    assert len(driver.pairs) == 2
    assert driver.pairs[-1] == (54, -54)
    assert not driver.stops and all(all(pair) for pair in driver.pairs)


@pytest.mark.parametrize("fault", ["stop", "hazard", "uid", "depth", "feedback",
                                   "reverse", "receipt", "full_reverse"])
def test_independent_straight_does_not_bypass_terminal_safety(monkeypatch, fault):
    rt, owner, driver, clock, publish, _ = writer(monkeypatch, base=60., yaw=4.)
    churn_at_three_terminal_checks(monkeypatch, rt, clock, publish, base=70., yaw=-4.)
    original = rt._plan_ordinary_snapshot_forward
    injected = []

    def change(*args, **kwargs):
        if kwargs.get("straight_fallback") and not injected:
            injected.append(True)
            if fault == "stop":
                owner._explicit_stop_requested = True
            elif fault == "hazard":
                rt.hard_stop_check = lambda _: True
            elif fault == "uid":
                owner._follow_controller.active_target_id = 2
            elif fault == "depth":
                clock[0] += .251
                rt._steering_feedback = feedback(clock[0], 24., 24.)
            elif fault == "feedback":
                rt._steering_feedback = feedback(clock[0]-.151, 24., 24.)
            elif fault == "reverse":
                rt._steering_feedback = feedback(clock[0], -8., -8.)
            elif fault == "receipt":
                rt.backend.send_targets(20, -20, "another-writer")
            else:
                rt._visible_wheel_guard.pending_full_reverse = True
        return original(*args, **kwargs)

    monkeypatch.setattr(rt, "_plan_ordinary_snapshot_forward", change)
    rt._service_follow_wheels()
    assert injected
    own_pairs = driver.pairs[2:] if fault == "receipt" else driver.pairs[1:]
    assert not any(left > 0 and right < 0 for left, right in own_pairs)


def test_first_budget_runs_outside_serial_lock_and_terminal_budget_inside(monkeypatch):
    rt, owner, driver, clock, publish, _ = writer(monkeypatch, base=60., yaw=4.)
    publish(70., 4.)
    original = owner._depth_forward_continuation_limit
    lock_states = []

    def budget(*args, **kwargs):
        lock_states.append(owner.motor_io_lock.locked())
        return original(*args, **kwargs)

    owner._depth_forward_continuation_limit = budget
    rt._service_follow_wheels()
    assert lock_states == [False, True]
    assert driver.pairs[-1] == (74, -66)


def test_negative_encoder_update_while_waiting_for_commit_is_not_ignored(monkeypatch):
    rt, owner, driver, clock, publish, _ = writer(monkeypatch, base=60., yaw=4.)
    publish(70., 4.)

    def reversing_feedback():
        clock[0] += .001
        rt._steering_feedback = feedback(clock[0], -8., -8.)

    inject(monkeypatch, rt, "commit", reversing_feedback)
    rt._service_follow_wheels()
    assert not any(left > 0 and right < 0 for left, right in driver.pairs[1:])


def removed_feedback_grant(monkeypatch):
    rt, owner, driver, clock, publish, _ = writer(monkeypatch, base=60., yaw=0.)
    controller = owner._follow_controller
    controller._distance_pi_admitted_grant = (1, 10.)
    controller._distance_pi_grant_withdrawal = (
        1, 10., "lateral_depth:continuation_feedback_stale")
    owner._depth30_linear_snapshot = owner._depth30_linear_timing = None
    return rt, owner, driver, clock, controller


@pytest.mark.parametrize("reason", ["stale", "unavailable"])
def test_removed_feedback_grant_only_exposes_memory_not_old_authority(monkeypatch, reason):
    rt, owner, driver, clock, controller = removed_feedback_grant(monkeypatch)
    controller._distance_pi_grant_withdrawal = (
        1, 10., "lateral_depth:continuation_feedback_"+reason)
    proof = rt.recovery_forward_execution_anchor(1, clock[0])
    assert proof is not None
    assert rt.forward_recovery_anchor_valid(1, proof, clock[0])
    assert owner._fresh_depth_linear_snapshot(1) is None
    assert owner._depth30_linear_snapshot is None
    assert driver.pairs == [(60, -60)] and not driver.stops


@pytest.mark.parametrize("fault", [
    "no_admission", "newer_admission", "wrong_uid", "wrong_withdrawal_sample",
    "no_withdrawal", "danger", "visibility", "physical_expiry", "zero_reason",
    "zero_packet", "stop_packet", "same_pair_other_writer", "reverse_packet",
    "identity_changed", "explicit_stop", "long_gap",
])
def test_removed_source_memory_cannot_cross_other_withdrawals_or_motor_writes(monkeypatch, fault):
    rt, owner, driver, clock, controller = removed_feedback_grant(monkeypatch)
    if fault == "no_admission":
        controller._distance_pi_admitted_grant = None
    elif fault == "newer_admission":
        controller._distance_pi_admitted_grant = (1, 10.01)
        controller._distance_pi_grant_withdrawal = (
            1, 10.01, "lateral_depth:continuation_feedback_stale")
    elif fault == "wrong_uid":
        controller._distance_pi_admitted_grant = (2, 10.)
        controller._distance_pi_grant_withdrawal = (
            2, 10., "lateral_depth:continuation_feedback_stale")
    elif fault == "wrong_withdrawal_sample":
        controller._distance_pi_grant_withdrawal = (
            1, 9.99, "lateral_depth:continuation_feedback_stale")
    elif fault == "no_withdrawal":
        controller._distance_pi_grant_withdrawal = None
    elif fault in {"danger", "visibility", "physical_expiry", "zero_reason"}:
        reason = {"danger": "momentum_stop", "visibility": "lateral_depth:visibility_expired",
                  "physical_expiry": "lateral_depth:physical_expired", "zero_reason": "zero_approved"}[fault]
        controller._distance_pi_grant_withdrawal = (1, 10., reason)
    elif fault == "zero_packet":
        rt.backend.send_targets(0, 0, "ZERO")
    elif fault == "stop_packet":
        rt.backend.send_stop("STOP", mode="emergency")
    elif fault == "same_pair_other_writer":
        rt.backend.send_targets(60, -60, "OTHER_WRITER")
    elif fault == "reverse_packet":
        rt.backend.send_targets(-60, 60, "REVERSE")
    elif fault == "identity_changed":
        controller.active_target_id = 2
    elif fault == "explicit_stop":
        owner._explicit_stop_requested = True
    else:
        clock[0] = 10.351
    assert rt.recovery_forward_execution_anchor(1, clock[0]) is None


@pytest.mark.parametrize("change", ["withdrawal", "admission", "stop"])
def test_removed_source_rechecks_revocation_after_state_readers(monkeypatch, change):
    rt, owner, driver, clock, controller = removed_feedback_grant(monkeypatch)
    original = rt._visible_wheel_control_active

    def changing_reader():
        result = original()
        if change == "withdrawal":
            controller._distance_pi_grant_withdrawal = (1, 10., "identity_conflict")
        elif change == "admission":
            controller._distance_pi_admitted_grant = (1, 10.01)
        else:
            rt.backend.send_stop("CONCURRENT_STOP", mode="emergency")
        return result

    monkeypatch.setattr(rt, "_visible_wheel_control_active", changing_reader)
    assert rt.recovery_forward_execution_anchor(1, clock[0]) is None
