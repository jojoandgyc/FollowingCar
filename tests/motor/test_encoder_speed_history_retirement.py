"""New encoder evidence can retire a responded command without new motor I/O."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from car_control_modular.executed_speed_budget import (
    executed_speed_bound_rpm, observe_completed_speed_response,
)
from test_executed_speed_budget import record
from test_forward_execution_anchor import execution_runtime, write_forward
from test_visible_wheel_continuity import feedback


def observe(history, stamp, left, right, *, receipt=None):
    return observe_completed_speed_response(
        history, feedback=feedback(stamp, left, right), now=stamp+.005,
        receipt=history[-1].receipt if receipt is None else receipt)


def test_two_post_lower_write_samples_retire_without_new_write_or_clock_renewal():
    old = record(pair=(74, 60))
    low = record(old, pair=(40, 30), stamp=10.1, seq=2)
    first = observe(low, 10.15, 35, 30)
    assert executed_speed_bound_rpm(first, uid=1, now=10.155) == 74
    second = observe(first, 10.20, 25, 20)
    assert executed_speed_bound_rpm(second, uid=1, now=10.205) == 40
    assert second[-1].receipt is low[-1].receipt
    assert second[-1].receipt.completed_at == 10.1
    assert executed_speed_bound_rpm(low, uid=1, now=10.205) == 74


@pytest.mark.parametrize('kind', ['duplicate', 'older', 'rising', 'too_fast', 'stale', 'bad'])
def test_inadequate_feedback_cannot_retire_history(kind):
    low = record(record(pair=(74, 60)), pair=(40, 30), stamp=10.1, seq=2)
    first = observe(low, 10.15, 35, 30)
    stamp, left, right = 10.20, 25, 20
    if kind == 'duplicate': stamp, left, right = 10.15, 35, 30
    if kind == 'older': stamp = 10.14
    if kind == 'rising': left = 38
    if kind == 'too_fast': left = 50
    fb = feedback(stamp, left, right)
    if kind == 'bad': fb.trustworthy = False
    latest = observe_completed_speed_response(
        first, feedback=fb, now=stamp+(.151 if kind == 'stale' else .005),
        receipt=first[-1].receipt)
    assert executed_speed_bound_rpm(latest, uid=1, now=10.3) == 74


@pytest.mark.parametrize('kind', ['stop', 'untracked_equal', 'higher_write'])
def test_intervening_command_breaks_response_proof(kind):
    low = record(record(pair=(74, 60)), pair=(40, 30), stamp=10.1, seq=2)
    first = observe(low, 10.15, 35, 30)
    receipt = None if kind == 'stop' else replace(first[-1].receipt)
    if kind == 'higher_write':
        first = record(first, pair=(80, 80), stamp=10.17, seq=3)
        receipt = first[-1].receipt
    changed = observe_completed_speed_response(
        first, feedback=feedback(10.20, 25, 20), now=10.205, receipt=receipt)
    assert executed_speed_bound_rpm(changed, uid=1, now=10.205) >= 74


def test_runtime_feedback_progress_without_packets_leaves_frozen_grant_unchanged(monkeypatch):
    runtime, owner, driver, _, clock, _ = execution_runtime(monkeypatch)
    write_forward(runtime, owner)
    clock[0] = 10.02
    with owner.motor_io_lock:
        runtime._send_follow_wheel_targets(30, -30, 'LOW', visible_required=True)
    writes = tuple(driver.pairs)
    grant, timing = owner._depth30_linear_snapshot, owner._depth30_linear_timing
    clock[0] = 10.035
    runtime._observe_executed_speed_response(feedback(10.03, 25, 23))
    assert runtime.continuation_executed_speed_bound_rpm(1, clock[0]) == 50
    clock[0] = 10.075
    runtime._observe_executed_speed_response(feedback(10.07, 24, 21))
    assert runtime.continuation_executed_speed_bound_rpm(1, clock[0]) == 30
    assert tuple(driver.pairs) == writes
    assert owner._depth30_linear_snapshot is grant
    assert owner._depth30_linear_timing is timing


def test_encoder_worker_publishes_response_without_triggering_motor_write(monkeypatch):
    runtime, owner, driver, _, clock, _ = execution_runtime(monkeypatch)
    runtime.config.steering_feedback_poll_interval_sec = .05
    runtime.config.steering_feedback_log_interval_sec = .5
    runtime.config.steering_feedback_left_body_deg_per_encoder_deg = .5225
    runtime.config.steering_feedback_right_body_deg_per_encoder_deg = .5424
    write_forward(runtime, owner)
    clock[0] = 10.02
    with owner.motor_io_lock:
        runtime._send_follow_wheel_targets(30, -30, 'LOW', visible_required=True)
    writes = tuple(driver.pairs)
    calls = []

    class StopAfterTwo:
        count = 0

        def is_set(self):
            return self.count == 2

        def wait(self, delay):
            self.count += 1

    event = owner.action_stop_event = StopAfterTwo()

    def read_motor_status(side):
        clock[0] = (10.03, 10.07)[event.count]
        calls.append(side)
        pair = ((25, -23), (24, -21))[event.count]
        return SimpleNamespace(speed_rpm=pair[0 if side == 'left' else 1],
                               error_code=0, position_degree=0)

    driver.read_motor_status = read_motor_status
    runtime._steering_feedback_loop()
    assert calls == ['left', 'right', 'left', 'right']
    assert runtime._steering_feedback.timestamp == 10.07
    assert runtime.continuation_executed_speed_bound_rpm(1, clock[0]) == 30
    assert tuple(driver.pairs) == writes
