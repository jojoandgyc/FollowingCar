"""Instruction-boundary readers and STOP ownership; fake clocks/serial only.

The trace hook observes actual backend assignments, without hard-coded source
line numbers or fabricated receipts. It is synchronous fault injection, not a
claim about how often these very short windows occur in the running system.
"""
from contextlib import contextmanager
import sys

import pytest

from test_distance_tracking_response import setup
from test_lateral_zero_runtime import owner
from test_depth_authority_250 import authority, advance
from test_feedback_interval_execution import execution_case, admit


@contextmanager
def observe_publication(backend, phase, inspect):
    """Inspect exactly one real assignment window, restoring any prior trace."""
    code = backend._write_speed_pair.__func__.__code__
    previous_trace = sys.gettrace()
    seen = []

    def trace(frame, event, arg):
        if event != "line" or frame.f_code is not code or seen:
            return trace
        submitted = frame.f_locals.get("submitted")
        published = backend.last_speed_write
        if submitted is None or published is None:
            return trace
        pending_window = (published is submitted
                          and backend.last_speed_receipt is submitted.previous_receipt)
        completed_window = (published is not submitted
                            and published.completed_receipt is not None
                            and backend.last_speed_receipt is None)
        if ((phase == "pending_before_receipt_clear" and pending_window)
                or (phase == "ack_before_receipt_publish" and completed_window)):
            seen.append(phase)
            inspect(submitted, published)
        return trace

    sys.settrace(trace)
    try:
        yield seen
    finally:
        sys.settrace(previous_trace)


@pytest.mark.parametrize("phase", [
    "pending_before_receipt_clear", "ack_before_receipt_publish",
])
def test_each_field_publication_window_keeps_budget_not_fake_receipt(execution_case, phase):
    a = execution_case()
    admit(a)
    old_receipt = a.backend.last_speed_receipt
    old_history = a.action._continuation_executed_speed_history
    old_grant = a.owner._fresh_depth_linear_snapshot(1)

    def inspect(submitted, published):
        assert submitted.previous_receipt is old_receipt
        assert submitted.completed_receipt is None
        assert a.action._continuation_executed_speed_history is old_history
        if phase == "pending_before_receipt_clear":
            assert a.backend.last_speed_receipt is old_receipt
            assert published is submitted
        else:
            assert a.backend.last_speed_receipt is None
            assert published.completed_receipt.sequence == old_receipt.sequence + 1
            assert published.completed_receipt.completed_at == a.clock.now
        assert a.action.braking_interval_continuation_bound_rpm(1, a.stamp, a.clock.now) == 76.
        assert a.action.continuation_executed_speed_bound_rpm(1, a.clock.now) == 76.
        assert a.owner._fresh_depth_linear_snapshot(1) == old_grant
        # Reads may create temporary accounting views, never completed ledger
        # entries, changed physical sample clocks, or extra driver commands.
        assert a.action._continuation_executed_speed_history is old_history
        assert a.owner._depth30_linear_timing is a.timing
        assert a.timing.depth_expires_at == pytest.approx(a.stamp + .30)

    with observe_publication(a.backend, phase, inspect) as seen:
        a.action._service_follow_wheels()
    assert seen == [phase]
    assert a.backend.driver.pairs == [(76, -76), (76, -76)]
    receipt = a.backend.last_speed_receipt
    assert receipt is not old_receipt
    assert receipt is a.backend.last_speed_write.completed_receipt
    assert receipt is a.action._continuation_executed_speed_history[-1].receipt
    assert a.owner._fresh_depth_linear_snapshot(1) == old_grant


def test_two_normal_writes_keep_contiguous_receipts_without_renewing_depth(execution_case):
    a = execution_case()
    admit(a)
    old_grant = a.owner._fresh_depth_linear_snapshot(1)
    a.action._service_follow_wheels()
    first = a.backend.last_speed_receipt
    advance(a, a.clock.now + .055)
    a.action._service_follow_wheels()
    second = a.backend.last_speed_receipt
    history = a.action._continuation_executed_speed_history
    assert second is not first
    assert [item.receipt.sequence for item in history] == [1, 2, 3]
    assert history[-2].receipt is first and history[-1].receipt is second
    assert a.backend.last_speed_write.previous_receipt is first
    assert a.backend.last_speed_write.completed_receipt is second
    assert all(item.response_anchor_valid for item in history)
    assert len(a.backend.driver.pairs) == 3
    assert all(left > 0 > right for left, right in a.backend.driver.pairs)
    assert a.action.braking_interval_continuation_bound_rpm(1, a.stamp, a.clock.now) == 76.
    current = a.owner._fresh_depth_linear_snapshot(1)
    assert current is not None and current[2:] == old_grant[2:]
    assert 0 < current[1] <= old_grant[1]
    assert a.owner._depth30_linear_timing is a.timing
    assert a.timing.depth_expires_at == pytest.approx(a.stamp + .30)


@pytest.mark.parametrize("phase", ["before_ledger", "after_ledger"])
@pytest.mark.parametrize("mode", ["emergency", "free"])
def test_stop_after_dual_ack_cannot_be_undone_by_late_bookkeeping(
        execution_case, monkeypatch, phase, mode):
    a = execution_case()
    admit(a)
    old_history = a.action._continuation_executed_speed_history
    old_generation = a.backend.stop_write_generation
    original_note = a.action._note_follow_packet
    original_stop = a.backend.driver.stop_all
    stopped = []
    receipts = []

    def stop_driver(stop_mode=0):
        # Invalidation precedes even the first physical STOP transaction.
        assert a.backend.stop_write_generation > old_generation
        assert a.backend.last_speed_receipt is None
        assert a.backend.last_speed_write is None
        assert a.action.braking_interval_continuation_bound_rpm(1, a.stamp, a.clock.now) is None
        stopped.append(stop_mode)
        original_stop(stop_mode)

    def stop_after_ack():
        receipt = a.backend.last_speed_receipt
        assert receipt is not None
        assert a.backend.last_speed_write.completed_receipt is receipt
        receipts.append(receipt)
        a.backend.send_stop("publication_test_stop", mode=mode, preserve_zero=True)
        assert a.owner._fresh_depth_linear_snapshot(1) is None
        # The physical STOP must already invalidate accounting, independently
        # of the later control-state publication that owns the held stop.
        a.owner._explicit_stop_requested = True

    def note(*args, **kwargs):
        if phase == "before_ledger":
            stop_after_ack()
        result = original_note(*args, **kwargs)
        if phase == "after_ledger":
            stop_after_ack()
        return result

    monkeypatch.setattr(a.backend.driver, "stop_all", stop_driver)
    monkeypatch.setattr(a.action, "_note_follow_packet", note)
    a.action._service_follow_wheels()
    assert stopped and len(receipts) == 1
    if phase == "before_ledger":
        assert a.action._continuation_executed_speed_history is old_history
    else:
        assert a.action._continuation_executed_speed_history[-1].receipt is receipts[0]
    assert a.backend.last_speed_receipt is None
    assert a.backend.last_speed_write is None
    assert a.owner._fresh_depth_linear_snapshot(1) is None
    monkeypatch.setattr(a.action, "_note_follow_packet", original_note)
    pair_count = len(a.backend.driver.pairs)
    a.action._service_follow_wheels()
    assert a.backend.driver.pairs[pair_count:] == []  # No speed-mode zero either.
    assert a.timing.depth_expires_at == pytest.approx(a.stamp + .30)
