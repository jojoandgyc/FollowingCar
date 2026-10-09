"""A detector-only lease appearing at the yaw write lock blocks old reverse."""

from contextlib import contextmanager

import pytest

from test_detector_reverse_guard import runtime


@pytest.mark.parametrize("raw", [True, False])
def test_reverse_zero_percent_yaw_rechecks_detector_lease_under_motor_lock(raw):
    action, sent = runtime(raw=raw)
    action.owner._current_steer_correction_rpm = 5
    action._yaw_revision_write_allowed = lambda *_: True
    if raw:
        # Only the guard is under test; the normal visible wheel writer has
        # its own unrelated feedback and Depth setup.
        action._send_follow_wheel_targets = lambda left, right, label, **kw: sent.append(
            (left, right, label))

    @contextmanager
    def motor_lock():
        # Published after send_percent_backward's unlocked entry check.
        action.owner._detector_identity_lease = False
        yield

    action.owner.motor_io_lock = motor_lock()
    assert action.send_percent_backward(0) is None
    assert sent == [(0, 0, "REVERSE_DETECTOR_IDENTITY_GUARD")]


@pytest.mark.parametrize("raw", [True, False])
def test_reverse_yaw_rechecks_after_revision_callback(raw):
    action, sent = runtime(raw=raw)
    action.owner._current_steer_correction_rpm = 5
    if raw:
        action._send_follow_wheel_targets = lambda left, right, label, **kw: sent.append(
            (left, right, label))

    def revision_check(*_):
        action.owner._detector_identity_lease = False
        return True

    action._yaw_revision_write_allowed = revision_check
    assert action.send_percent_backward(0) is None
    assert sent == [(0, 0, "REVERSE_DETECTOR_IDENTITY_GUARD")]


@pytest.mark.parametrize("raw", [True, False])
def test_full_identity_keeps_reverse_zero_percent_yaw(raw):
    action, sent = runtime(raw=raw)
    action.owner._current_steer_correction_rpm = 5
    action._yaw_revision_write_allowed = lambda *_: True
    if raw:
        action._send_follow_wheel_targets = lambda left, right, label, **kw: sent.append(
            (left, right, label))

    assert action.send_percent_backward(0) == "YAW_ONLY"
    assert len(sent) == 1 and sent[0][-1] == "YAW_ONLY"
