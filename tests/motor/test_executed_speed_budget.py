"""Physical command-memory provenance; fake clocks and drivers only."""
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import pytest

from car_control_modular.executed_speed_budget import (
    EXECUTED_SPEED_MEMORY_SEC, MAX_EXECUTED_SPEED_RECORDS,
    executed_speed_bound_rpm, record_completed_speed,
)
from car_control_modular.mssd_motor import MotorSpeedReceipt
from test_forward_execution_anchor import execution_runtime, write_forward
from test_visible_wheel_continuity import feedback


_AUTOMATIC_PREVIOUS_RECEIPT = object()


def record(history=(), *, uid=1, pair=(60, 40), stamp=10., seq=1,
           previous=_AUTOMATIC_PREVIOUS_RECEIPT, now=None, **kwargs):
    receipt = MotorSpeedReceipt(seq, pair[0], -pair[1], stamp)
    if previous is _AUTOMATIC_PREVIOUS_RECEIPT:
        previous = history[-1].receipt if history else None
    return record_completed_speed(
        history, uid=uid, applied=pair, signs=(1, -1), receipt=receipt,
        previous_receipt=previous, now=stamp if now is None else now,
        packet_written=True, **kwargs)


def test_low_and_zero_receipts_retain_original_high_until_original_expiry():
    history = record()
    history = record(history, pair=(20, 10), stamp=10.3, seq=2,
                     previous=history[-1].receipt)
    history = record(history, pair=(0, 0), stamp=10.6, seq=3,
                     previous=history[-1].receipt)
    assert executed_speed_bound_rpm(history, uid=1, now=10.64) == 60.
    assert executed_speed_bound_rpm(history, uid=1, now=10.66) == 20.
    assert executed_speed_bound_rpm(history, uid=1, now=10.96) == 0.
    assert executed_speed_bound_rpm(history, uid=1, now=11.26) is None
    assert EXECUTED_SPEED_MEMORY_SEC == .65


@pytest.mark.parametrize("pair,expected", [
    ((10, 50), 50), ((-12, 12), 12), ((-15, -25), 25), ((0, 0), 0),
])
def test_outer_absolute_not_average_is_physical_budget(pair, expected):
    assert executed_speed_bound_rpm(record(pair=pair), uid=1, now=10.) == expected


def test_target_identity_isolation_and_new_target_clears_old_entries():
    history = record()
    assert executed_speed_bound_rpm(history, uid=2, now=10.) is None
    history = record(history, uid=2, pair=(10, 20), stamp=10.1, seq=2)
    assert len(history) == 1 and history[0].uid == 2
    assert executed_speed_bound_rpm(history, uid=1, now=10.1) is None
    assert executed_speed_bound_rpm(history, uid=2, now=10.1) == 20.


@pytest.mark.parametrize("change", [
    "not_written", "same_object", "same_sequence", "older_sequence",
    "missing", "untyped", "future", "old", "nan_stamp", "inf_stamp",
    "zero_stamp", "nan_now", "bool_now", "wrong_sign", "wrong_pair",
    "nonfinite_pair", "bool_pair", "invalid_uid", "bool_uid", "float_uid",
    "bad_sign", "bool_sequence", "zero_sequence", "bad_previous",
    "out_of_order_time",
])
def test_invalid_uncompleted_or_unproven_receipt_does_not_mutate_history(change):
    history = record()
    receipt = MotorSpeedReceipt(2, 30, -20, 10.1)
    args = dict(uid=1, applied=(30, 20), signs=(1, -1), receipt=receipt,
                previous_receipt=history[-1].receipt, now=10.1, packet_written=True)
    if change == "not_written": args["packet_written"] = False
    elif change == "same_object": args["receipt"] = history[-1].receipt
    elif change == "same_sequence": args["receipt"] = replace(receipt, sequence=1)
    elif change == "older_sequence":
        args["previous_receipt"] = replace(receipt, sequence=3)
    elif change == "missing": args["receipt"] = None
    elif change == "untyped": args["receipt"] = object()
    elif change == "future": args["receipt"] = replace(receipt, completed_at=10.1001)
    elif change == "old": args["now"] = 10.751
    elif change == "nan_stamp": args["receipt"] = replace(receipt, completed_at=float("nan"))
    elif change == "inf_stamp": args["receipt"] = replace(receipt, completed_at=float("inf"))
    elif change == "zero_stamp": args["receipt"] = replace(receipt, completed_at=0.)
    elif change == "nan_now": args["now"] = float("nan")
    elif change == "bool_now": args["now"] = True
    elif change == "wrong_sign": args["signs"] = (1, 1)
    elif change == "wrong_pair": args["applied"] = (31, 20)
    elif change == "nonfinite_pair": args["applied"] = (float("nan"), 20)
    elif change == "bool_pair": args["applied"] = (True, 20)
    elif change == "invalid_uid": args["uid"] = 0
    elif change == "bool_uid": args["uid"] = True
    elif change == "float_uid": args["uid"] = 1.
    elif change == "bad_sign": args["signs"] = (1, 2)
    elif change == "bool_sequence": args["receipt"] = replace(receipt, sequence=True)
    elif change == "zero_sequence": args["receipt"] = replace(receipt, sequence=0)
    elif change == "bad_previous": args["previous_receipt"] = object()
    elif change == "out_of_order_time": args["receipt"] = replace(receipt, completed_at=9.99)
    assert record_completed_speed(history, **args) is history
    assert executed_speed_bound_rpm(history, uid=1, now=10.1) == 60.


@pytest.mark.parametrize("uid,now", [
    (0, 10.), (True, 10.), (1., 10.), (1, float("nan")),
    (1, float("inf")), (1, True), (1, 9.999), (1, 10.650001),
])
def test_reader_rejects_invalid_identity_time_or_expired_history(uid, now):
    history = record()
    assert executed_speed_bound_rpm(history, uid=uid, now=now) is None
    assert len(history) == 1


def test_history_is_immutable_copy_on_write_and_bounded():
    history = record()
    original = history
    with pytest.raises(FrozenInstanceError): history[0].outer_rpm = 1.
    for seq in range(2, 90):
        history = record(history, stamp=10.+seq*.001, seq=seq)
    assert isinstance(history, tuple) and len(history) == MAX_EXECUTED_SPEED_RECORDS
    assert original[0].receipt.sequence == 1 and len(original) == 1
    before = history
    for _ in range(5): assert executed_speed_bound_rpm(history, uid=1, now=10.1) == 60.
    assert history is before


def test_real_follow_writer_records_actual_completed_applied_packet(monkeypatch):
    runtime, owner, driver, _, clock, _ = execution_runtime(monkeypatch)
    assert runtime.continuation_executed_speed_bound_rpm(1, clock[0]) is None
    write_forward(runtime, owner)
    assert driver.pairs == [(50, -30)]
    entries = runtime._continuation_executed_speed_history
    assert entries[-1].receipt is runtime.backend.last_speed_receipt
    assert entries[-1].outer_rpm == 50.
    assert runtime.continuation_executed_speed_bound_rpm(1, clock[0]) == 50.
    # A direct backend writer is outside normal follow and cannot add evidence.
    runtime.backend.send_targets(150, -150, "OTHER_WRITER")
    assert runtime._continuation_executed_speed_history is entries
    assert runtime.continuation_executed_speed_bound_rpm(1, clock[0]) == 50.


def test_fake_driver_ack_completion_owns_timestamp(monkeypatch):
    runtime, owner, driver, _, clock, _ = execution_runtime(monkeypatch)
    original = driver.set_left_speed
    def completed(value):
        clock[0] += .005
        original(value)
    monkeypatch.setattr(driver, "set_left_speed", completed)
    write_forward(runtime, owner)
    assert runtime._continuation_executed_speed_history[-1].receipt.completed_at == 10.005


def test_planned_positive_but_guard_zero_records_zero_not_requested_speed(monkeypatch):
    runtime, owner, driver, _, clock, _ = execution_runtime(monkeypatch)
    owner._fresh_depth_linear_snapshot = lambda uid, now=None: None
    runtime.get_steering_feedback = lambda: feedback(clock[0], 20, 20)
    with owner.motor_io_lock:
        runtime._send_follow_wheel_targets(50, -30, "STEER", visible_required=True)
    assert driver.pairs == [(0, 0)]
    assert runtime.continuation_executed_speed_bound_rpm(1, clock[0]) == 0.


def test_failed_dual_write_adds_no_history_and_does_not_renew_old_entry(monkeypatch):
    runtime, owner, driver, _, clock, _ = execution_runtime(monkeypatch)
    write_forward(runtime, owner)
    history = runtime._continuation_executed_speed_history
    def fail_left(value): raise RuntimeError("fake acknowledgement failure")
    monkeypatch.setattr(driver, "set_left_speed", fail_left)
    clock[0] += .01
    with owner.motor_io_lock, pytest.raises(RuntimeError, match="fake acknowledgement"):
        runtime._send_follow_wheel_targets(45, -35, "STEER", visible_required=True)
    assert runtime._continuation_executed_speed_history is history
    assert history[-1].receipt.completed_at == 10.


def test_normal_zero_does_not_clear_physical_history_and_reader_has_no_authority(monkeypatch):
    runtime, owner, driver, _, clock, _ = execution_runtime(monkeypatch)
    write_forward(runtime, owner)
    clock[0] += .02
    with owner.motor_io_lock:
        runtime._send_follow_wheel_targets(0, 0, "FOLLOW_ZERO", visible_required=True)
    assert driver.pairs == [(50, -30), (0, 0)]
    assert runtime.continuation_executed_speed_bound_rpm(1, clock[0]) == 50.
    owner._depth30_linear_snapshot = None
    owner._depth30_continuation_veto = (1, 9.84)
    writes = list(driver.pairs)
    assert runtime.continuation_executed_speed_bound_rpm(1, clock[0]) == 50.
    assert owner._depth30_linear_snapshot is None
    assert owner._depth30_continuation_veto == (1, 9.84)
    assert driver.pairs == writes
    owner._follow_controller.active_target_id = 2
    assert runtime.continuation_executed_speed_bound_rpm(1, clock[0]) is None
    assert runtime.continuation_executed_speed_bound_rpm(2, clock[0]) is None


def test_backend_skip_cannot_append_same_receipt(monkeypatch):
    runtime, owner, driver, _, clock, _ = execution_runtime(monkeypatch)
    write_forward(runtime, owner)
    history = runtime._continuation_executed_speed_history
    monkeypatch.setattr(runtime.backend, "send_targets", lambda *args, **kwargs: None)
    clock[0] += .01
    write_forward(runtime, owner)
    assert runtime._continuation_executed_speed_history is history
    assert len(driver.pairs) == 1


def response(stamp, left, right, trustworthy=True):
    return SimpleNamespace(timestamp=stamp, left_forward_rpm=left,
                           right_forward_rpm=right, trustworthy=trustworthy)


def test_confirmed_lower_response_retires_old_high_for_future_reads_only():
    history = record(pair=(80, 60))
    frozen_bound = executed_speed_bound_rpm(history, uid=1, now=10.)
    history = record(history, pair=(40, 30), stamp=10.10, seq=2)
    history = record(history, pair=(40, 30), stamp=10.16, seq=3,
                     feedback=response(10.15, 35, 30))
    assert executed_speed_bound_rpm(history, uid=1, now=10.16) == 80
    history = record(history, pair=(40, 30), stamp=10.22, seq=4,
                     feedback=response(10.21, 32, 28))
    assert executed_speed_bound_rpm(history, uid=1, now=10.22) == 40
    retired = next(item for item in history if item.receipt.sequence == 1)
    assert retired.retired_at == 10.21  # Past-interval proof retains the record.
    assert frozen_bound == 80  # Existing grants do not acquire the later value.


@pytest.mark.parametrize("case", [
    "same_sample", "before_lower_write", "rising", "above_lower_target",
    "stale", "future", "untrusted", "gap", "later_higher_write", "out_of_order",
])
def test_unproven_lower_response_must_preserve_old_high(case):
    history = record(pair=(80, 60))
    history = record(history, pair=(40, 30), stamp=10.1, seq=2)
    history = record(history, pair=(40, 30), stamp=10.16, seq=3,
                     feedback=response(10.15, 35, 30))
    fb = response(10.21, 32, 28)
    pair, stamp = (40, 30), 10.22
    if case == "same_sample": fb = response(10.15, 35, 30)
    elif case == "before_lower_write": fb = response(10.09, 32, 28)
    elif case == "rising": fb = response(10.21, 36, 30)
    elif case == "above_lower_target": fb = response(10.21, 45, 30)
    elif case == "stale": stamp = 10.4
    elif case == "future": fb = response(10.23, 32, 28)
    elif case == "untrusted": fb = response(10.21, 32, 28, False)
    elif case == "gap": fb, stamp = response(10.32, 32, 28), 10.33
    elif case == "later_higher_write": pair = (50, 30)
    elif case == "out_of_order": fb = response(10.14, 32, 28)
    history = record(history, pair=pair, stamp=stamp, seq=4, feedback=fb)
    assert executed_speed_bound_rpm(history, uid=1, now=stamp) == 80


def test_cap693_values_with_two_writer_observations_keep_unconfirmed_38():
    # Numeric boundary fixture, NOT a replay of the real writer schedule:
    # add a fake write at .90 so both saved 29/27 feedback values are consumed.
    # The real trace's sparse writes need not observe both of these samples.
    history = record(pair=(45, 35), stamp=29122.558305)
    history = record(history, pair=(38, 22), stamp=29122.8587, seq=2)
    history = record(history, pair=(38, 22), stamp=29122.90, seq=3,
                     feedback=response(29122.8707, 29, 17))
    history = record(history, pair=(30, 30), stamp=29122.971579, seq=4,
                     feedback=response(29122.9059, 27, 13))
    assert executed_speed_bound_rpm(history, uid=1, now=29123.0) == 38
    history = record(history, pair=(22, 22), stamp=29123.0257, seq=5,
                     feedback=response(29123.007877, 30, 25))
    assert executed_speed_bound_rpm(history, uid=1, now=29123.0257) == 38


@pytest.mark.parametrize("pair,first,second", [
    ((43, 33), (20, 18), (21, 21)),  # CAP32, still responding to high write.
    ((44, 24), (24, 21), (31, 25)),  # CAP834, both encoder readings rising.
])
def test_cap32_and_cap834_rising_response_preserves_high(pair, first, second):
    history = record(pair=pair)
    history = record(history, pair=(18, 18), stamp=10.01, seq=2)
    history = record(history, pair=(18, 18), stamp=10.07, seq=3,
                     feedback=response(10.06, *first))
    history = record(history, pair=(18, 18), stamp=10.13, seq=4,
                     feedback=response(10.12, *second))
    assert executed_speed_bound_rpm(history, uid=1, now=10.13) == max(pair)


def test_zero_requires_two_actual_zero_feedback_samples_not_zero_command():
    history = record()
    history = record(history, pair=(0, 0), stamp=10.1, seq=2)
    history = record(history, pair=(0, 0), stamp=10.16, seq=3,
                     feedback=response(10.15, 0, 0))
    assert executed_speed_bound_rpm(history, uid=1, now=10.16) == 60
    history = record(history, pair=(0, 0), stamp=10.22, seq=4,
                     feedback=response(10.21, 0, 0))
    assert executed_speed_bound_rpm(history, uid=1, now=10.22) == 0


def test_writer_retirement_uses_existing_feedback_without_extra_reads(monkeypatch):
    runtime, owner, driver, _, clock, _ = execution_runtime(monkeypatch)
    write_forward(runtime, owner)
    # Only this normal writer can publish completed evidence. It records the
    # feedback it already read; the speed-budget reader performs no new read.
    clock[0] = 10.02
    with owner.motor_io_lock:
        runtime._send_follow_wheel_targets(30, -30, "LOW", visible_required=True)
    runtime.get_steering_feedback = lambda: feedback(10.025, 25, 23)
    clock[0] = 10.03
    with owner.motor_io_lock:
        runtime._send_follow_wheel_targets(30, -30, "LOW", visible_required=True)
    assert runtime.continuation_executed_speed_bound_rpm(1, clock[0]) == 50
    runtime.get_steering_feedback = lambda: feedback(10.035, 24, 21)
    clock[0] = 10.04
    with owner.motor_io_lock:
        runtime._send_follow_wheel_targets(30, -30, "LOW", visible_required=True)
    assert runtime.continuation_executed_speed_bound_rpm(1, clock[0]) == 30
    calls = list(driver.pairs)
    def must_not_read(): raise AssertionError("budget read must not read encoder")
    runtime.get_steering_feedback = must_not_read
    assert runtime.continuation_executed_speed_bound_rpm(1, clock[0]) == 30
    assert driver.pairs == calls


@pytest.mark.parametrize("bad", ["older", "same_stamp_same_outer_other_wheel", "same_stamp_changed_outer"])
def test_conflicting_or_older_feedback_clears_proof_until_two_new_samples(bad):
    history = record(pair=(80, 60))
    history = record(history, pair=(40, 30), stamp=10.1, seq=2)
    history = record(history, pair=(40, 30), stamp=10.16, seq=3,
                     feedback=response(10.15, 35, 30))
    fb = (response(10.14, 34, 28) if bad == "older" else
          response(10.15, 35, 25) if bad == "same_stamp_same_outer_other_wheel" else
          response(10.15, 34, 30))
    history = record(history, pair=(40, 30), stamp=10.17, seq=4, feedback=fb)
    history = record(history, pair=(40, 30), stamp=10.22, seq=5,
                     feedback=response(10.21, 32, 28))
    assert executed_speed_bound_rpm(history, uid=1, now=10.22) == 80
    history = record(history, pair=(40, 30), stamp=10.28, seq=6,
                     feedback=response(10.27, 30, 26))
    assert executed_speed_bound_rpm(history, uid=1, now=10.28) == 40


@pytest.mark.parametrize("interruption", ["stop", "replaced_receipt", "sequence_gap"])
def test_receipt_discontinuity_requires_two_samples_after_a_new_low_write(interruption):
    history = record(pair=(80, 60))
    history = record(history, pair=(40, 30), stamp=10.1, seq=2)
    history = record(history, pair=(40, 30), stamp=10.16, seq=3,
                     feedback=response(10.15, 35, 30))
    before = history
    previous = history[-1].receipt
    seq = 4
    if interruption == "stop":
        previous = None  # STOP/current-mode I/O invalidates the backend receipt.
    elif interruption == "replaced_receipt":
        previous = replace(previous)  # Equal payload does not establish identity.
    else:
        seq = 5  # Another completed packet followed the captured prior receipt.
    history = record(history, pair=(40, 30), stamp=10.22, seq=seq,
                     previous=previous, feedback=response(10.21, 32, 28))
    assert executed_speed_bound_rpm(history, uid=1, now=10.22) == 80
    assert history[0].receipt is before[0].receipt
    assert executed_speed_bound_rpm(history, uid=1, now=10.650001) == 40
    # Clearing sample tuples alone would reuse the old low anchor and combine
    # .21 (before the new low write) with .27 to retire the high too soon.
    history = record(history, pair=(40, 30), stamp=10.28, seq=seq+1,
                     feedback=response(10.27, 30, 26))
    assert executed_speed_bound_rpm(history, uid=1, now=10.28) == 80
    history = record(history, pair=(40, 30), stamp=10.34, seq=seq+2,
                     feedback=response(10.33, 29, 25))
    assert executed_speed_bound_rpm(history, uid=1, now=10.34) == 40
    assert executed_speed_bound_rpm(before, uid=1, now=10.34) == 80
    assert before[0].receipt.completed_at == 10.


@pytest.mark.parametrize("interruption", ["reverse", "same_pair"])
def test_real_nonfollow_write_cannot_complete_an_old_low_response(monkeypatch, interruption):
    runtime, owner, driver, _, clock, _ = execution_runtime(monkeypatch)
    write_forward(runtime, owner)
    frozen_grant = owner._depth30_linear_snapshot

    def low(now, sample_stamp, left, right):
        clock[0] = now
        runtime.get_steering_feedback = lambda: feedback(sample_stamp, left, right)
        with owner.motor_io_lock:
            runtime._send_follow_wheel_targets(30, -30, "LOW", visible_required=True)

    low(10.02, 10.015, 28, 26)
    low(10.03, 10.025, 25, 23)
    before = runtime._continuation_executed_speed_history
    assert runtime.continuation_executed_speed_bound_rpm(1, clock[0]) == 50
    clock[0] = 10.032
    if interruption == "reverse":
        runtime.send_percent_backward(30)
        assert driver.pairs[-1] == (-60, 60)
    else:
        with owner.motor_io_lock:
            runtime.backend.send_targets(30, -30, "OTHER_WRITER")
    assert runtime._continuation_executed_speed_history is before
    assert runtime.backend.last_speed_receipt.sequence == 4
    low(10.04, 10.035, 24, 21)
    assert runtime.continuation_executed_speed_bound_rpm(1, clock[0]) == 50
    low(10.05, 10.045, 23, 20)
    assert runtime.continuation_executed_speed_bound_rpm(1, clock[0]) == 50
    low(10.06, 10.055, 22, 19)
    assert runtime.continuation_executed_speed_bound_rpm(1, clock[0]) == 30
    assert owner._depth30_linear_snapshot is frozen_grant
    assert executed_speed_bound_rpm(before, uid=1, now=clock[0]) == 50
    assert before[0].receipt.completed_at == 10.
