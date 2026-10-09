"""Admission selection changes provenance, never creates motion permission."""
from dataclasses import replace

import pytest

from car_control_modular.control_types import DepthLinearTiming
from car_control_modular.forward_write_snapshot import select_forward_write_snapshot
from car_control_modular.sample_braking import SampleBrakingAssessment
from test_unified_forward_snapshot import feedback, inject, writer


def grant(stamp, cap, uid=1):
    raw = ("forward", cap, uid, stamp)
    assessment = SampleBrakingAssessment(uid, stamp, stamp, 10., 24., 24.,
        stamp, 0., 1.1, .8168, 1., .15, 100., outer_allowance_rpm=10.)
    return raw, DepthLinearTiming(raw, stamp, stamp+.25, braking_assessment=assessment)


def test_latest_positive_grant_contracts_pair_and_keeps_real_sample_and_deadline():
    old, new = grant(10., 80.), grant(10.1, 50.)
    result = select_forward_write_snapshot(1, old, new, (76, 84), 10.12, 100.)
    assert result.pair == (46, 54)
    assert result.linear is new[0] and result.grant is new
    assert result.grant[1].depth_expires_at == 10.35


@pytest.mark.parametrize("fault", ["older", "future", "expired", "uid", "zero", "reverse",
    "missing", "assessment", "assessment_uid", "nan_cap", "nan_now", "pair_reverse",
    "pair_zero", "pair_nan", "too_small_for_yaw"])
def test_replacement_provenance_and_bounds_are_not_optional(fault):
    old, new = grant(10., 80.), grant(10.1, 50.)
    pair, now = (76, 84), 10.12
    if fault == "older":
        new = grant(9.99, 50.)
    elif fault == "future":
        new = grant(10.2, 50.)
    elif fault == "expired":
        now = 10.351
    elif fault == "uid":
        new = grant(10.1, 50., uid=2)
    elif fault == "zero":
        new = grant(10.1, 0.)
    elif fault == "reverse":
        raw = ("backward", 50., 1, 10.1)
        new = raw, replace(new[1], snapshot=raw)
    elif fault == "missing":
        new = None
    elif fault == "assessment":
        new = new[0], replace(new[1], braking_assessment=None)
    elif fault == "assessment_uid":
        new = new[0], replace(new[1], braking_assessment=grant(10.1, 50., uid=2)[1].braking_assessment)
    elif fault == "nan_cap":
        new = grant(10.1, float("nan"))
    elif fault == "nan_now":
        now = float("nan")
    elif fault == "pair_reverse":
        pair = (-4, 40)
    elif fault == "pair_zero":
        pair = (0, 0)
    elif fault == "pair_nan":
        pair = (float("nan"), 40)
    else:
        new = grant(10.1, 1.)
    assert select_forward_write_snapshot(1, old, new, pair, now, 100.) is None


@pytest.mark.parametrize("pair", [(50, 50), (46, 54), (54, 46)])
def test_final_admission_uses_higher_cap_without_increasing_guarded_pair(pair):
    old, new = grant(10., 50.), grant(10.1, 90.)
    result = select_forward_write_snapshot(1, old, new, pair, 10.12, 100.)
    assert result.pair == pair
    assert result.linear == new[0]


@pytest.mark.parametrize("fault", ["full_reverse", "commanded_reverse", "read_veto",
                                    "explicit_stop", "shutdown", "faulted_feedback"])
def test_new_grant_at_terminal_cannot_hide_explicit_veto(monkeypatch, fault):
    rt, owner, driver, clock, publish, _ = writer(monkeypatch, base=80., yaw=-4.)

    def update():
        publish(70., -4.)
        if fault == "full_reverse":
            rt._visible_wheel_guard.pending_full_reverse = True
        elif fault == "commanded_reverse":
            rt._visible_wheel_guard.commanded_reverse = True
        elif fault == "read_veto":
            owner._depth30_read_veto = (1, clock[0], "explicit_sample_rejection")
        elif fault == "explicit_stop":
            owner._explicit_stop_requested = True
        elif fault == "shutdown":
            owner._runtime_shutdown_requested = True
        else:
            rt._steering_feedback = replace(feedback(clock[0], 24., 24.), trustworthy=False)

    inject(monkeypatch, rt, "terminal", update)
    rt._service_follow_wheels()
    assert all(pair == (0, 0) for pair in driver.pairs[1:])


@pytest.mark.parametrize("yaw", [-4., 4.])
def test_twenty_rising_grants_at_final_gate_preserve_forward_arc_then_catch_up(monkeypatch, yaw):
    from test_follow_continuity_acceptance import qualified_writer

    rt, owner, driver, clock, publish = qualified_writer(monkeypatch, base=20., yaw=yaw)
    original = rt._linear_packet_write_limit
    published = []

    def rising(*args, **kwargs):
        clock[0] += .001
        base = owner._depth30_linear_snapshot[1] + 3.
        published.append(base)
        publish(base, yaw)
        return original(*args, **kwargs)

    monkeypatch.setattr(rt, "_linear_packet_write_limit", rising)
    for _ in range(20):
        clock[0] += .05
        rt._steering_feedback = feedback(clock[0], 24., 24.)
        count, writes = len(published), len(driver.pairs)
        rt._service_follow_wheels()
        assert len(published) == count + 1  # No retries caused by a higher cap.
        assert len(driver.pairs) == writes + 1
        left, right = driver.pairs[-1]
        assert left > 0 > right and .5 * (left + right) == yaw
        assert .5 * (left - right) == pytest.approx(published[-1] - 3.)
        assert rt._forward_execution_anchor.sample_timestamp == clock[0]
        assert owner._depth30_linear_timing.depth_expires_at == pytest.approx(clock[0] + .25)
        assert not owner.motor_io_lock.locked() and not driver.stops

    monkeypatch.setattr(rt, "_linear_packet_write_limit", original)
    clock[0] += .05
    rt._steering_feedback = feedback(clock[0], 24., 24.)
    rt._service_follow_wheels()
    assert driver.pairs[-1] == (int(80. + yaw), -int(80. - yaw))
    assert all(left > 0 > right for left, right in driver.pairs)


def test_completed_packet_ledger_uses_terminal_feedback_not_planning_feedback(monkeypatch):
    rt, owner, driver, clock, publish, _ = writer(monkeypatch, base=60., yaw=-4.)
    publish(60., -4.)
    original_budget = owner._depth_forward_continuation_limit
    original_note = rt._note_follow_packet
    budgets, completed_feedback = [], []
    latest = [None]

    def change_feedback(*args, **kwargs):
        result = original_budget(*args, **kwargs)
        budgets.append(True)
        if len(budgets) == 2:
            clock[0] += .001
            latest[0] = rt._steering_feedback = feedback(clock[0], 20., 20.)
        return result

    def record(uid, applied, linear, final_feedback, *args, **kwargs):
        completed_feedback.append(final_feedback)
        return original_note(uid, applied, linear, final_feedback, *args, **kwargs)

    monkeypatch.setattr(owner, "_depth_forward_continuation_limit", change_feedback)
    monkeypatch.setattr(rt, "_note_follow_packet", record)
    rt._service_follow_wheels()
    assert budgets == [True, True]
    assert completed_feedback == [latest[0]]
    assert rt._follow_write_feedback is latest[0]
    assert driver.pairs[-1] == (56, -64)
