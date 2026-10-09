"""Past-interval command proof, without motor hardware or inferred motion."""
from dataclasses import replace

import pytest

from car_control_modular import action_runtime as module
from car_control_modular.executed_speed_budget import (
    EXECUTED_SPEED_MEMORY_SEC, EXECUTED_SPEED_HISTORY_SEC,
    executed_speed_bound_rpm, executed_interval_speed_bound_rpm,
    observe_completed_speed_response, record_completed_speed,
)
from car_control_modular.mssd_motor import MotorSpeedReceipt
from test_executed_speed_budget import record, response
from test_forward_execution_anchor import execution_runtime, write_forward
from test_unified_forward_snapshot import writer
from test_zero_publication_lifecycle import churn_at_three_terminal_checks


def bound(history, sample=10.05, now=10.25, **kwargs):
    args = dict(uid=1, sample_timestamp=sample, now=now, receipt=history[-1].receipt)
    args.update(kwargs)
    return executed_interval_speed_bound_rpm(history, **args)


def retired_history():
    history = record(pair=(80, 60))
    history = record(history, pair=(40, 30), stamp=10.1, seq=2)
    for stamp, left in [(10.15, 35.), (10.20, 30.)]:
        history = observe_completed_speed_response(history,
            feedback=response(stamp, left, 25.), now=stamp+.005, receipt=history[-1].receipt)
    return history


def test_retirement_after_capture_does_not_erase_motion_before_response():
    history = retired_history()
    assert history[0].retired_at == 10.2
    assert executed_speed_bound_rpm(history, uid=1, now=10.205) == 40.
    assert bound(history, sample=10.12, now=10.205) == 80.
    assert bound(history, sample=10.1999, now=10.205) == 80.
    assert bound(history, sample=10.2, now=10.205) == 40.
    assert bound(history, sample=10.201, now=10.205) == 40.
    frozen = history
    for _ in range(5):
        assert bound(history, sample=10.12, now=10.205) == 80.
    assert history is frozen and history[0].receipt.completed_at == 10.


def test_original_memory_expiry_inside_interval_does_not_drop_past_cost():
    history = record(pair=(80, 60))
    history = record(history, pair=(40, 30), stamp=10.5, seq=2)
    history = record(history, pair=(40, 30), stamp=10.7, seq=3)
    assert EXECUTED_SPEED_MEMORY_SEC == .65
    assert EXECUTED_SPEED_HISTORY_SEC == pytest.approx(.95)
    assert executed_speed_bound_rpm(history, uid=1, now=10.7) == 40.
    assert bound(history, sample=10.6, now=10.7) == 80.
    assert bound(history, sample=10.651, now=10.7) == 80.


@pytest.mark.parametrize("retired_when,expected", [(None, 80.), (10.75, 40.), (10.85, 80.)])
def test_700_to_900ms_old_command_needs_measured_retirement_for_interval_proof(retired_when, expected):
    history = record(pair=(80, 60))
    history = record(history, pair=(40, 30), stamp=10.5, seq=2)
    if retired_when is not None:
        for stamp in (retired_when-.05, retired_when):
            history = observe_completed_speed_response(history,
                feedback=response(stamp, 30., 25.), now=stamp+.005,
                receipt=history[-1].receipt)
    history = record(history, pair=(40, 30), stamp=10.9, seq=3)
    assert history[0].receipt.completed_at == 10.
    assert executed_speed_bound_rpm(history, uid=1, now=10.9) == 40.
    assert bound(history, sample=10.8, now=10.9) == expected


@pytest.mark.parametrize("sample,now,expected", [
    (9.999, 10.2, None), (10., 10.2, 60.), (10.05, 10.2, 60.),
    (10.1, 10.2, 60.), (10.2, 10.2, 60.), (10., 10.3001, None),
    (10.21, 10.2, None), (float("nan"), 10.2, None), (True, 10.2, None),
    (10.05, float("inf"), None), (10.05, True, None),
])
def test_coverage_clock_and_window_are_explicit(sample, now, expected):
    history = record()
    history = record(history, pair=(40, 30), stamp=10.1, seq=2)
    assert bound(history, sample=sample, now=now) == expected


@pytest.mark.parametrize("interruption", ["stop", "equal_other_writer", "sequence_gap"])
def test_chain_break_cannot_cover_an_earlier_sample_but_new_anchor_can(interruption):
    history = record()
    previous = history[-1].receipt
    seq = 2
    if interruption == "stop":
        previous = None
    elif interruption == "equal_other_writer":
        previous = replace(previous)
    else:
        seq = 3
    history = record(history, pair=(40, 30), stamp=10.2, seq=seq, previous=previous)
    assert bound(history, sample=10.15) is None
    assert bound(history, sample=10.21) == 60.  # Existing high MAX remains conservative.


@pytest.mark.parametrize("fault", [
    "empty", "not_tuple", "bad_entry", "wrong_uid", "bool_uid", "sequence_hole",
    "reordered_time", "receipt_replaced", "stop_receipt", "ack_not_accounted",
    "future_receipt", "retired_in_future", "retired_before_write", "wrong_outer",
])
def test_missing_or_inconsistent_execution_proof_fails_closed(fault):
    history = record()
    history = record(history, pair=(40, 30), stamp=10.1, seq=2)
    receipt = history[-1].receipt
    if fault == "empty": history = ()
    elif fault == "not_tuple": history = list(history)
    elif fault == "bad_entry": history = (object(),)
    elif fault == "wrong_uid": history = (history[0], replace(history[1], uid=2))
    elif fault == "bool_uid": history = (history[0], replace(history[1], uid=True))
    elif fault == "sequence_hole":
        receipt = replace(receipt, sequence=3)
        history = (history[0], replace(history[1], receipt=receipt))
    elif fault == "reordered_time":
        receipt = replace(receipt, completed_at=9.99)
        history = (history[0], replace(history[1], receipt=receipt))
    elif fault == "receipt_replaced": receipt = replace(receipt)
    elif fault == "stop_receipt": receipt = None
    elif fault == "ack_not_accounted": receipt = MotorSpeedReceipt(3, 90, -90, 10.2)
    elif fault == "future_receipt":
        receipt = replace(receipt, completed_at=10.3)
        history = (history[0], replace(history[1], receipt=receipt))
    elif fault == "retired_in_future": history = (replace(history[0], retired_at=10.3), history[1])
    elif fault == "retired_before_write": history = (replace(history[0], retired_at=9.9), history[1])
    else: history = (replace(history[0], outer_rpm=1.), history[1])
    assert executed_interval_speed_bound_rpm(history, uid=1,
        sample_timestamp=10.05, now=10.25, receipt=receipt) is None


def test_runtime_reader_is_cache_only_and_zero_receipt_is_data_not_permission(monkeypatch):
    rt, owner, driver, _, clock, _ = execution_runtime(monkeypatch)
    write_forward(rt, owner)
    clock[0] = 10.05
    rt.get_steering_feedback = lambda: pytest.fail("interval reader must not query feedback")
    before = (tuple(driver.pairs), tuple(driver.stops), rt._continuation_executed_speed_history)
    with owner.motor_io_lock:
        assert rt.braking_interval_speed_bound_rpm(1, 10., clock[0]) == 50.
    assert rt.braking_interval_speed_bound_rpm(1, 9.99, clock[0]) is None
    assert before == (tuple(driver.pairs), tuple(driver.stops), rt._continuation_executed_speed_history)
    rt.backend.send_targets(0, 0, "UNTRACKED_ZERO")
    assert rt.braking_interval_speed_bound_rpm(1, 10., clock[0]) is None
    assert rt.continuation_executed_speed_bound_rpm(1, clock[0]) == 50.


@pytest.mark.parametrize("race", ["history", "receipt", "stop", "generation", "uid"])
def test_runtime_rechecks_proof_after_pure_read(monkeypatch, race):
    rt, owner, driver, _, clock, _ = execution_runtime(monkeypatch)
    write_forward(rt, owner)
    clock[0] = 10.05
    original = module.executed_interval_speed_bound_rpm

    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        assert result == 50.
        if race == "history":
            rt._continuation_executed_speed_history = tuple(list(rt._continuation_executed_speed_history))
        elif race == "receipt":
            rt.backend.send_targets(50, -30, "OTHER_WRITER")
        elif race == "stop": rt.backend.send_stop("STOP", mode="emergency")
        elif race == "generation": rt.backend.stop_write_generation += 1
        else: owner._follow_controller.active_target_id = 2
        return result

    monkeypatch.setattr(module, "executed_interval_speed_bound_rpm", changed)
    assert rt.braking_interval_speed_bound_rpm(1, 10., clock[0]) is None


@pytest.mark.parametrize("yaw", [-10., 10.])
def test_terminal_straight_fallback_does_not_erase_valid_single_wheel_reversal(monkeypatch, yaw):
    rt, owner, driver, clock, publish, _ = writer(monkeypatch, base=60., yaw=4.)

    def legal_limit(raw, timing, now, *, feedback, quiet):
        model = timing.braking_assessment
        budget = model.budget(now, max(feedback.left_forward_rpm, feedback.right_forward_rpm),
            authorized_rpm=raw[1], execution_bound_rpm=max(model.travel_bound_rpm,
                raw[1]+model.outer_allowance_rpm))
        return budget.cap_rpm, budget.reason

    owner._depth_forward_continuation_limit = legal_limit
    churn_at_three_terminal_checks(monkeypatch, rt, clock, publish, base=70., yaw=-4.,
                                  after_last=lambda: publish(5., yaw))
    rt._service_follow_wheels()
    assert driver.pairs == [(64, -56), (0, 0)]
    assert not driver.stops


@pytest.mark.parametrize("reader_name", ["braking_interval_speed_bound_rpm",
                                        "braking_interval_continuation_bound_rpm"])
@pytest.mark.parametrize("after_lower_write", [False, True])
def test_untracked_high_packet_cannot_be_hidden_by_a_later_lower_packet(
        monkeypatch, reader_name, after_lower_write):
    rt, owner, driver, _, clock, _ = execution_runtime(monkeypatch)
    write_forward(rt, owner)
    clock[0] = 10.04
    rt.backend.send_targets(110, -110, "UNTRACKED_HIGH")
    if after_lower_write:
        previous = rt.backend.last_speed_receipt
        clock[0] = 10.07
        rt.backend.send_targets(40, -40, "LATER_NORMAL_LOW")
        rt._continuation_executed_speed_history = record_completed_speed(
            rt._continuation_executed_speed_history, uid=1, applied=(40, 40),
            signs=(1, -1), receipt=rt.backend.last_speed_receipt,
            previous_receipt=previous, now=clock[0], packet_written=True)
    assert getattr(rt, reader_name)(1, 10., clock[0]) is None
    assert rt.continuation_executed_speed_bound_rpm(1, clock[0]) == 50.


@pytest.mark.parametrize("race", ["untracked_high", "stop", "generation", "uid"])
def test_continuation_rechecks_actual_writer_ownership_after_interval_query(monkeypatch, race):
    rt, owner, driver, _, clock, _ = execution_runtime(monkeypatch)
    write_forward(rt, owner)
    clock[0] = 10.05
    original = module.executed_interval_speed_bound_rpm

    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        assert result == 50.
        if race == "untracked_high":
            rt.backend.send_targets(110, -110, "UNTRACKED_HIGH")
        elif race == "stop": rt.backend.send_stop("STOP", mode="emergency")
        elif race == "generation": rt.backend.stop_write_generation += 1
        else: owner._follow_controller.active_target_id = 2
        return result

    monkeypatch.setattr(module, "executed_interval_speed_bound_rpm", changed)
    assert rt.braking_interval_continuation_bound_rpm(1, 10., clock[0]) is None


@pytest.mark.parametrize("reader_name,expected", [
    ("braking_interval_speed_bound_rpm", None),
    ("braking_interval_continuation_bound_rpm", 50.),
])
def test_feedback_retirement_during_continuation_retains_older_higher_cost(
        monkeypatch, reader_name, expected):
    rt, owner, driver, _, clock, _ = execution_runtime(monkeypatch)
    write_forward(rt, owner)
    clock[0] = 10.02
    with owner.motor_io_lock:
        rt._send_follow_wheel_targets(30, -30, "LOW", visible_required=True)
    clock[0] = 10.09
    writes = tuple(driver.pairs)
    receipt = rt.backend.last_speed_receipt
    original = module.executed_interval_speed_bound_rpm

    def retire_during_query(*args, **kwargs):
        result = original(*args, **kwargs)
        assert result == 50.
        rt._observe_executed_speed_response(response(10.04, 25., 20.))
        rt._observe_executed_speed_response(response(10.08, 24., 19.))
        assert rt.continuation_executed_speed_bound_rpm(1, clock[0]) == 30.
        return result

    monkeypatch.setattr(module, "executed_interval_speed_bound_rpm", retire_during_query)
    with owner.motor_io_lock:
        assert getattr(rt, reader_name)(1, 10.085, clock[0]) == expected
    assert rt.backend.last_speed_receipt is receipt
    assert tuple(driver.pairs) == writes
